"""Run multiple DDIM evaluations and aggregate them into one JSON payload."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CHECKPOINT = ROOT / "runs" / "run_stable01" / "step-30000.pt"
DEFAULT_SAMPLING_STEPS = [250, 125, 50, 10, 5]
EVALUATION_SCRIPT = HERE / "evaluation.py"


@dataclass(frozen=True)
class SweepConfig:
    checkpoint: Path
    output_json: Path
    device: str
    sampling_steps: list[int]
    ddim_eta: float
    max_features_eval: int
    max_rows_eval: int
    new_instances_eval: int
    n_splits: int
    n_repeats: int
    random_state: int
    datasets: str | None
    use_cache: bool


def parse_sampling_steps(raw: str) -> list[int]:
    steps = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not steps:
        raise ValueError("sampling steps must not be empty")
    if any(step <= 0 for step in steps):
        raise ValueError(f"sampling steps must be positive, got {steps}")
    return steps


def tail_summary(text: str, *, max_lines: int = 40, max_chars: int = 4000) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    summary = "\n".join(lines)
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def build_eval_command(
    config: SweepConfig,
    *,
    num_sampling_steps: int,
    output_json: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATION_SCRIPT.resolve()),
        "--checkpoint",
        str(config.checkpoint),
        "--device",
        config.device,
        "--num-sampling-steps",
        str(int(num_sampling_steps)),
        "--sampling-method",
        "ddim",
        "--ddim-eta",
        str(float(config.ddim_eta)),
        "--max-features-eval",
        str(int(config.max_features_eval)),
        "--max-rows-eval",
        str(int(config.max_rows_eval)),
        "--new-instances-eval",
        str(int(config.new_instances_eval)),
        "--n-splits",
        str(int(config.n_splits)),
        "--n-repeats",
        str(int(config.n_repeats)),
        "--random-state",
        str(int(config.random_state)),
        "--output-json",
        str(output_json),
    ]
    if config.datasets:
        command.extend(["--datasets", config.datasets])
    if not config.use_cache:
        command.append("--no-cache")
    return command


def make_failure_record(
    *,
    num_sampling_steps: int,
    command: list[str],
    returncode: int | None,
    stdout: str,
    stderr: str,
    error: str,
) -> dict[str, Any]:
    return {
        "num_sampling_steps": int(num_sampling_steps),
        "status": "failed",
        "command": command,
        "returncode": returncode,
        "stdout_tail": tail_summary(stdout),
        "stderr_tail": tail_summary(stderr),
        "error": error,
    }


def make_success_record(
    *,
    num_sampling_steps: int,
    command: list[str],
    stdout: str,
    stderr: str,
    evaluation_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "num_sampling_steps": int(num_sampling_steps),
        "status": "success",
        "command": command,
        "stdout_tail": tail_summary(stdout),
        "stderr_tail": tail_summary(stderr),
        "evaluation": evaluation_payload,
    }


def run_sweep(config: SweepConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "output_json": str(config.output_json),
            "sampling_method": "ddim",
        },
        "results": [],
    }
    write_json(config.output_json, payload)

    with tempfile.TemporaryDirectory(prefix="tabicl_regression03_ddim_") as temp_dir:
        temp_root = Path(temp_dir)
        for step in config.sampling_steps:
            temp_output = temp_root / f"{config.checkpoint.stem}_ddim_steps{step}.json"
            command = build_eval_command(
                config,
                num_sampling_steps=step,
                output_json=temp_output,
            )

            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

            if completed.returncode != 0:
                record = make_failure_record(
                    num_sampling_steps=step,
                    command=command,
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    error="evaluation.py returned a non-zero exit code",
                )
            else:
                try:
                    with temp_output.open("r", encoding="utf-8") as f:
                        evaluation_payload = json.load(f)
                except FileNotFoundError:
                    record = make_failure_record(
                        num_sampling_steps=step,
                        command=command,
                        returncode=completed.returncode,
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                        error=f"evaluation.py did not create {temp_output}",
                    )
                except json.JSONDecodeError as exc:
                    record = make_failure_record(
                        num_sampling_steps=step,
                        command=command,
                        returncode=completed.returncode,
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                        error=f"evaluation.py wrote invalid JSON: {exc}",
                    )
                else:
                    record = make_success_record(
                        num_sampling_steps=step,
                        command=command,
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                        evaluation_payload=evaluation_payload,
                    )

            payload["results"].append(record)
            write_json(config.output_json, payload)

    return payload


def parse_args() -> SweepConfig:
    parser = argparse.ArgumentParser(
        description="Run a DDIM evaluation sweep and aggregate results into one JSON file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Checkpoint to evaluate.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Combined output JSON path. Defaults next to the checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--sampling-steps",
        type=str,
        default="250,125,50,10,5",
        help="Comma-separated DDIM sampling-step counts.",
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="DDIM eta passed through to evaluation.py.",
    )
    parser.add_argument(
        "--max-features-eval",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-rows-eval",
        type=int,
        default=1000,
        help="Skip datasets with more than this many rows; <=0 disables row filtering.",
    )
    parser.add_argument(
        "--new-instances-eval",
        type=int,
        default=0,
        help="Maximum rows per dataset passed to evaluation.py; <=0 uses all rows.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset subset to pass to evaluation.py.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable dataset cache in evaluation.py.",
    )

    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")

    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else checkpoint.with_name(f"{checkpoint.stem}_ddim_sweep.json")
    )

    return SweepConfig(
        checkpoint=checkpoint,
        output_json=output_json,
        device=str(args.device),
        sampling_steps=parse_sampling_steps(args.sampling_steps),
        ddim_eta=float(args.ddim_eta),
        max_features_eval=int(args.max_features_eval),
        max_rows_eval=int(args.max_rows_eval),
        new_instances_eval=int(args.new_instances_eval),
        n_splits=int(args.n_splits),
        n_repeats=int(args.n_repeats),
        random_state=int(args.random_state),
        datasets=args.datasets.strip() if args.datasets and args.datasets.strip() else None,
        use_cache=not bool(args.no_cache),
    )


def main() -> dict[str, Any]:
    config = parse_args()
    print(
        "Running DDIM evaluation sweep for "
        f"{config.checkpoint.name} with steps {config.sampling_steps}",
        flush=True,
    )
    payload = run_sweep(config)
    print(f"Saved combined results to {config.output_json}", flush=True)
    return payload


if __name__ == "__main__":
    main()
