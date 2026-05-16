"""Sweep TabICL_regression03-compatible checkpoints with the evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabicl_style.evaluation import (
    DEFAULT_MAX_FEATURES_EVAL,
    DEFAULT_NEW_INSTANCES_EVAL,
    NDPRegressorWrapper,
    TABPFN_REGRESSION_DATASETS,
    _load_checkpoint,
    eval_model,
    get_regression_datasets,
)


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"step-(\d+)", path.stem)
    if match:
        return int(match.group(1))
    tail_digits = re.findall(r"(\d+)", path.stem)
    return int(tail_digits[-1]) if tail_digits else -1


def _sorted_checkpoints(checkpoints_dir: Path) -> List[Path]:
    checkpoints = sorted(checkpoints_dir.glob("step-*.pt"), key=_checkpoint_step)
    if checkpoints:
        return checkpoints
    return sorted(checkpoints_dir.glob("*.pt"), key=_checkpoint_step)


def _sanitize_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _format_metrics(metrics: Dict[str, float]) -> Dict[str, Dict[str, float | None]]:
    datasets: Dict[str, Dict[str, float | None]] = {}
    rmse_values: List[float] = []
    mae_values: List[float] = []
    r2_values: List[float] = []
    for dataset_name in TABPFN_REGRESSION_DATASETS:
        r2_key = f"{dataset_name}/R2"
        rmse_key = f"{dataset_name}/RMSE"
        mae_key = f"{dataset_name}/MAE"
        if r2_key not in metrics and rmse_key not in metrics and mae_key not in metrics:
            continue
        r2_value = _sanitize_float(metrics.get(r2_key))
        rmse_value = _sanitize_float(metrics.get(rmse_key))
        mae_value = _sanitize_float(metrics.get(mae_key))
        datasets[dataset_name] = {
            "R2": r2_value,
            "RMSE": rmse_value,
            "MAE": mae_value,
        }
        if r2_value is not None:
            r2_values.append(r2_value)
        if rmse_value is not None:
            rmse_values.append(rmse_value)
        if mae_value is not None:
            mae_values.append(mae_value)

    overall = {
        "R2": _sanitize_float(metrics.get("R2")),
        "RMSE": _sanitize_float(metrics.get("RMSE")),
        "MAE": _sanitize_float(metrics.get("MAE")),
    }
    if overall["R2"] is None and r2_values:
        overall["R2"] = float(np.mean(r2_values))
    if overall["RMSE"] is None and rmse_values:
        overall["RMSE"] = float(np.mean(rmse_values))
    if overall["MAE"] is None and mae_values:
        overall["MAE"] = float(np.mean(mae_values))

    return {"overall": overall, "datasets": datasets}


def _write_results(output_json: Path, payload: Dict) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _plot_metric(
    results: List[Dict],
    *,
    metric_name: str,
    output_path: Path,
) -> None:
    points = []
    for record in results:
        if record.get("error"):
            continue
        step = record.get("step")
        metric_value = (
            record.get("overall", {}).get(metric_name)
            if isinstance(record.get("overall"), dict)
            else None
        )
        if step is None or metric_value is None:
            continue
        points.append((int(step), float(metric_value)))

    if not points:
        print(f"No valid points for {metric_name}; skipping plot.")
        return

    points.sort(key=lambda item: item[0])
    steps = [item[0] for item in points]
    values = [item[1] for item in points]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, "-o")
    plt.xlabel("Checkpoint step")
    plt.ylabel(metric_name)
    plt.title(f"Checkpoint sweep: {metric_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {metric_name} plot to {output_path}")


def sweep_checkpoints(
    checkpoints_dir: Path,
    *,
    device: torch.device,
    num_sampling_steps: int,
    sampling_method: str,
    max_features_eval: int,
    new_instances_eval: int,
    n_splits: int,
    random_state: int,
    use_cache: bool,
    dataset_names: List[str] | None,
    output_json: Path,
) -> Dict:
    checkpoints = _sorted_checkpoints(checkpoints_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {checkpoints_dir}")

    print("Loading regression benchmark datasets once for the full sweep...", flush=True)
    datasets = get_regression_datasets(
        max_features_eval=max_features_eval,
        new_instances_eval=new_instances_eval,
        random_state=random_state,
        verbose=True,
        use_cache=use_cache,
        dataset_names=dataset_names,
    )
    print(f"Loaded {len(datasets)} dataset(s) for sweep.\n", flush=True)

    payload = {
        "config": {
            "checkpoints_dir": str(checkpoints_dir),
            "device": str(device),
            "num_sampling_steps": int(num_sampling_steps),
            "sampling_method": str(sampling_method),
            "max_features_eval": int(max_features_eval),
            "new_instances_eval": int(new_instances_eval),
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "use_cache": bool(use_cache),
            "datasets": dataset_names,
        },
        "results": [],
    }

    for checkpoint_path in checkpoints:
        step = _checkpoint_step(checkpoint_path)
        print(
            f"Evaluating checkpoint {checkpoint_path.name} (step={step})",
            flush=True,
        )
        record = {
            "checkpoint": str(checkpoint_path),
            "step": int(step),
            "overall": {},
            "datasets": {},
            "error": None,
        }

        try:
            model, process, _config = _load_checkpoint(checkpoint_path, device)
            regressor = NDPRegressorWrapper(
                model=model,
                process=process,
                device=device,
                num_sampling_steps=num_sampling_steps,
                sampling_method=sampling_method,
            )
            metrics = eval_model(
                regressor,
                datasets,
                n_splits=n_splits,
                random_state=random_state,
                plot_dataset=None,
                plot_dir=HERE / "evaluation_plots",
            )
            formatted = _format_metrics(metrics)
            record["overall"] = formatted["overall"]
            record["datasets"] = formatted["datasets"]
            print(
                "  "
                f"R2={record['overall'].get('R2')} | "
                f"RMSE={record['overall'].get('RMSE')} | "
                f"MAE={record['overall'].get('MAE')}",
                flush=True,
            )
        except Exception as exc:
            record["error"] = str(exc)
            print(f"  Failed: {exc}", flush=True)

        payload["results"].append(record)
        _write_results(output_json, payload)
        print(f"  Saved intermediate results to {output_json}\n", flush=True)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep TabICL_regression03-compatible checkpoints with the evaluation pipeline."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Directory containing step-*.pt checkpoints.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save checkpoint sweep results JSON.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save R2/RMSE/MAE sweep plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=50,
        help="Number of reverse diffusion steps per evaluation.",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
    )
    parser.add_argument(
        "--max-features-eval",
        type=int,
        default=DEFAULT_MAX_FEATURES_EVAL,
    )
    parser.add_argument("--new-instances-eval", type=int, default=DEFAULT_NEW_INSTANCES_EVAL)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated subset of dataset names to evaluate.",
    )
    parser.add_argument("--no-cache", action="store_true")

    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else checkpoints_dir / "evaluation" / "sweep_results.json"
    )
    plot_dir = (
        Path(args.plot_dir).resolve()
        if args.plot_dir
        else checkpoints_dir / "evaluation"
    )
    dataset_names = None
    if args.datasets:
        dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]

    payload = sweep_checkpoints(
        checkpoints_dir,
        device=torch.device(args.device),
        num_sampling_steps=int(args.num_sampling_steps),
        sampling_method=str(args.sampling_method),
        max_features_eval=int(args.max_features_eval),
        new_instances_eval=int(args.new_instances_eval),
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        use_cache=not bool(args.no_cache),
        dataset_names=dataset_names,
        output_json=output_json,
    )

    _plot_metric(
        payload["results"],
        metric_name="R2",
        output_path=plot_dir / "sweep_r2.png",
    )
    _plot_metric(
        payload["results"],
        metric_name="RMSE",
        output_path=plot_dir / "sweep_rmse.png",
    )
    _plot_metric(
        payload["results"],
        metric_name="MAE",
        output_path=plot_dir / "sweep_mae.png",
    )

    print(f"Saved final results to {output_json}", flush=True)


if __name__ == "__main__":
    main()
