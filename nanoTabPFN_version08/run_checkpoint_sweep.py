from __future__ import annotations

"""
Sweep evaluation over many training checkpoints for nanoTabPFN_version07
using the new discrete D3PM diffusion.

This script:
- Finds all `model_step_*.pt` checkpoints under a W&B run directory.
- For each checkpoint, evaluates benchmark datasets and sampled prior tasks.
- Logs ROC-AUC and related metrics to JSON and plots ROC-AUC vs. training steps.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent

from config import Config  # type: ignore
from evaluation import (  # type: ignore
    NanoTabPFNClassifier,
    compute_metrics,
    load_benchmark_datasets,
    load_checkpoint,
    sample_prior_tasks,
)


def _get_run_paths() -> Dict[str, Path]:
    """Return key paths for the W&B run and evaluation outputs."""
    # Adjust the run id below as needed when sweeping a different run.
    run_dir = HERE / "wandb" / "run-20251215_011406-rlqkac8z"
    files_dir = run_dir / "files"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return {"run_dir": run_dir, "files_dir": files_dir, "eval_dir": eval_dir}


def _sorted_checkpoints(files_dir: Path) -> List[Path]:
    """Return all model_step_*.pt checkpoints sorted by step."""
    ckpts = sorted(
        files_dir.glob("model_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return ckpts


def _load_existing(eval_dir: Path) -> tuple[List[Dict], List[Dict]]:
    """
    Load previously saved benchmark/prior records if they exist.
    Returns (benchmark_records, prior_records). Absent files yield empty lists.
    """
    bench_path = eval_dir / "benchmarks_metrics.json"
    prior_path = eval_dir / "prior_tasks_metrics.json"

    benchmark_records: List[Dict] = []
    prior_records: List[Dict] = []

    if bench_path.exists():
        try:
            with bench_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                benchmark_records = data.get("checkpoints", [])
        except Exception as e:
            print(f"Warning: failed to load {bench_path}: {e}")

    if prior_path.exists():
        try:
            with prior_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                prior_records = data.get("checkpoints", [])
        except Exception as e:
            print(f"Warning: failed to load {prior_path}: {e}")

    return benchmark_records, prior_records


def _build_classifier(ckpt_path: Path, device: torch.device) -> NanoTabPFNClassifier:
    """Load a checkpoint and build a NanoTabPFNClassifier with eval settings."""
    model, process, config_dict = load_checkpoint(ckpt_path, device)

    # Use evaluation settings from the training config if present, but override
    # the number of DDIM steps to 500 for this sweep.
    eval_cfg = config_dict.get("training", {})
    sampling_method = eval_cfg.get("eval_sampling_method", "ddim")
    num_steps = 500

    classifier = NanoTabPFNClassifier(
        model=model,
        process=process,
        device=device,
        num_classes=config_dict["model"]["num_outputs"],
        sampling_method=sampling_method,
        num_sampling_steps=num_steps,
        show_progress=True,  # show tqdm bars inside sampling
    )
    return classifier


def _evaluate_on_datasets(
    classifier: NanoTabPFNClassifier,
    datasets: List,
    step: int,
    *,
    kind: str,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate classifier on a list of (name, X_train, X_test, y_train, y_test) datasets.

    Returns mapping: dataset_name -> metrics dict (including "roc_auc").
    """
    results: Dict[str, Dict[str, float]] = {}

    for name, X_train, X_test, y_train, y_test in datasets:
        print(f"  [{kind}] dataset={name}")
        classifier.fit(X_train, y_train)
        y_logits = classifier.predict_logits(
            X_test,
            desc=f"{kind}:{name}@step={step}",
        )
        y_proba = classifier.logits_to_proba(y_logits)
        y_pred = y_proba.argmax(axis=1)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        results[name] = metrics

    return results


def _flush_json(
    eval_dir: Path,
    paths: Dict[str, Path],
    benchmark_records: List[Dict],
    prior_records: List[Dict],
    h5_path: Path,
) -> None:
    """
    Persist current benchmark and prior-task metrics to JSON.

    This is called after each checkpoint so partial results are not lost
    if the sweep is interrupted.
    """
    benchmarks_json = {
        "run_dir": str(paths["run_dir"]),
        "checkpoints": benchmark_records,
    }
    prior_json = {
        "run_dir": str(paths["run_dir"]),
        "prior_h5": str(h5_path),
        "checkpoints": prior_records,
    }

    with (eval_dir / "benchmarks_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(benchmarks_json, f, indent=2)

    with (eval_dir / "prior_tasks_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(prior_json, f, indent=2)


def main() -> None:
    paths = _get_run_paths()
    files_dir = paths["files_dir"]
    eval_dir = paths["eval_dir"]

    ckpts = _sorted_checkpoints(files_dir)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {files_dir}")

    # Load any previous sweep results to support resume.
    benchmark_records, prior_records = _load_existing(eval_dir)
    processed_steps = {int(rec["step"]) for rec in benchmark_records if "step" in rec}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Benchmark datasets: evaluate all four tasks.
    benchmark_dataset_keys = [
        "breast_cancer",
        "diabetes",
        "blood_transfusion",
        "amazon_employee_access",
    ]
    benchmark_datasets = load_benchmark_datasets(benchmark_dataset_keys)

    # Prior tasks are skipped for this sweep.
    prior_datasets = []
    h5_path = HERE / "NO_PRIOR_USED"

    steps: List[int] = []
    # Seed curves from any existing records (for resume plotting)
    benchmark_roc_curves: Dict[str, List[float]] = {name: [] for name, *_ in benchmark_datasets}
    prior_roc_curves: Dict[str, List[float]] = {name: [] for name, *_ in prior_datasets}

    def _add_existing_to_curves(records: List[Dict], curves: Dict[str, List[float]]):
        for rec in records:
            for ds_name, metrics in rec.get("datasets", {}).items():
                if "roc_auc" in metrics:
                    curves.setdefault(ds_name, []).append(metrics["roc_auc"])

    _add_existing_to_curves(benchmark_records, benchmark_roc_curves)
    _add_existing_to_curves(prior_records, prior_roc_curves)
    steps.extend([int(rec["step"]) for rec in benchmark_records if "step" in rec])

    for ckpt_path in ckpts:
        step = int(ckpt_path.stem.split("_")[-1])
        if step in processed_steps:
            print(f"\nSkipping checkpoint {ckpt_path.name} (step={step}) already processed.")
            continue
        print(f"\nEvaluating checkpoint {ckpt_path.name} (step={step})")

        steps.append(step)
        classifier = _build_classifier(ckpt_path, device)

        # Benchmark datasets
        bench_metrics = _evaluate_on_datasets(
            classifier,
            benchmark_datasets,
            step,
            kind="benchmark",
        )
        benchmark_records.append(
            {
                "step": step,
                "checkpoint": str(ckpt_path),
                "datasets": bench_metrics,
            }
        )
        for ds_name, metrics in bench_metrics.items():
            benchmark_roc_curves.setdefault(ds_name, []).append(metrics["roc_auc"])

        # Prior tasks (training prior H5)
        # Skip prior tasks.

        # Flush JSON logs after each checkpoint evaluation.
        _flush_json(eval_dir, paths, benchmark_records, prior_records, h5_path)

    # Plot ROC-AUC vs steps for benchmarks (4 curves)
    order = np.argsort(steps)
    sorted_steps = [steps[i] for i in order]

    plt.figure(figsize=(8, 5))
    for ds_name, auc_values in benchmark_roc_curves.items():
        if not auc_values:
            continue
        ys = [auc_values[i] for i in order if i < len(auc_values)]
        plt.plot(sorted_steps[: len(ys)], ys, marker="o", label=ds_name)
    plt.xlabel("Training steps")
    plt.ylabel("ROC-AUC")
    plt.title("Benchmark ROC-AUC vs training steps (version07 discrete)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_dir / "benchmarks_roc_auc.png", dpi=150)
    plt.close()

    # Prior-task plotting skipped (no prior tasks evaluated).


if __name__ == "__main__":
    main()
