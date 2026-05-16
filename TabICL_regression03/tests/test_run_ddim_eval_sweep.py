from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabicl_style.run_ddim_eval_sweep import SweepConfig, run_sweep


def test_run_sweep_builds_ddim_commands_and_aggregates_results(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "step-30000.pt"
    checkpoint.write_bytes(b"checkpoint")
    output_json = tmp_path / "step-30000_ddim_sweep.json"

    config = SweepConfig(
        checkpoint=checkpoint,
        output_json=output_json,
        device="cpu",
        sampling_steps=[250, 125, 50, 10, 5],
        ddim_eta=0.0,
        max_features_eval=32,
        new_instances_eval=20,
        n_splits=1,
        n_repeats=1,
        random_state=0,
        datasets="boston",
        use_cache=True,
    )

    seen_steps: list[int] = []

    def fake_run(command, capture_output, text, check):
        assert capture_output is True
        assert text is True
        assert check is False
        assert command[0] == sys.executable
        assert "--sampling-method" in command
        assert command[command.index("--sampling-method") + 1] == "ddim"

        step = int(command[command.index("--num-sampling-steps") + 1])
        seen_steps.append(step)
        per_run_output = Path(command[command.index("--output-json") + 1])

        if step == 50:
            return subprocess.CompletedProcess(
                command,
                returncode=2,
                stdout="started\n",
                stderr="boom\n",
            )

        payload = {
            "config": {
                "sampling_method": "ddim",
                "num_sampling_steps": step,
            },
            "overall_metrics": {"R2": step / 100.0},
            "datasets": {
                "boston": {"metrics": {"R2": step / 100.0, "RMSE": 1.0, "MAE": 1.0}}
            },
        }
        per_run_output.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=f"ok {step}\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    payload = run_sweep(config)

    assert seen_steps == [250, 125, 50, 10, 5]
    assert [record["num_sampling_steps"] for record in payload["results"]] == [250, 125, 50, 10, 5]

    for record in payload["results"]:
        assert record["command"][record["command"].index("--sampling-method") + 1] == "ddim"

    failed = payload["results"][2]
    assert failed["status"] == "failed"
    assert failed["returncode"] == 2
    assert failed["stderr_tail"] == "boom"

    success_steps = [250, 125, 10, 5]
    success_records = [record for record in payload["results"] if record["status"] == "success"]
    assert [record["num_sampling_steps"] for record in success_records] == success_steps
    for record in success_records:
        assert record["evaluation"]["config"]["sampling_method"] == "ddim"
        assert record["evaluation"]["config"]["num_sampling_steps"] == record["num_sampling_steps"]

    written_payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert written_payload == payload
