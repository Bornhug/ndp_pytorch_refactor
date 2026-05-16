"""Evaluate task-specific finetuned checkpoints on saved fixed outer splits."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetuning.common import (
    aggregate_metric_dicts,
    evaluate_fixed_split,
    load_task_split_json,
    parse_dataset_names,
    write_json,
)
from tabicl_style.evaluation import (
    _load_checkpoint,
    _save_normalized_prediction_plot,
)


def _load_summary_payload(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("summary.json is not a JSON object")
        return payload
    except Exception as exc:
        print(
            f"Warning: failed to read summary.json at {summary_path}: {exc}",
            flush=True,
        )
        return None


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _is_task_dir(path: Path) -> bool:
    return path.is_dir() and (path / "last.pt").is_file() and (path / "split.json").is_file()


def _scan_fold_directories(dataset_root: Path) -> list[Path]:
    fold_dirs: list[Path] = []
    for child in sorted(dataset_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or not child.name.startswith("fold_"):
            continue
        if _is_task_dir(child):
            fold_dirs.append(child)
    return fold_dirs


def _read_dataset_name_from_split(task_dir: Path) -> str:
    split_path = task_dir / "split.json"
    if not split_path.is_file():
        return task_dir.name
    try:
        with split_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        dataset_name = str(payload.get("dataset_name", task_dir.name)).strip()
        return dataset_name or task_dir.name
    except Exception:
        return task_dir.name


def _scan_dataset_task_dirs(finetune_root: Path) -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = {}

    if _is_task_dir(finetune_root):
        dataset_name = _read_dataset_name_from_split(finetune_root)
        discovered[dataset_name] = [finetune_root]
        return discovered

    direct_fold_dirs = _scan_fold_directories(finetune_root)
    if direct_fold_dirs:
        dataset_name = _read_dataset_name_from_split(direct_fold_dirs[0])
        discovered[dataset_name] = direct_fold_dirs
        return discovered

    for child in sorted(finetune_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if _is_task_dir(child):
            dataset_name = _read_dataset_name_from_split(child)
            discovered[dataset_name] = [child]
            continue
        fold_dirs = _scan_fold_directories(child)
        if fold_dirs:
            dataset_name = _read_dataset_name_from_split(fold_dirs[0])
            discovered[dataset_name] = fold_dirs
    return discovered


def _discover_dataset_task_dirs(
    finetune_root: Path,
) -> tuple[dict[str, list[Path]], dict[str, Any] | None]:
    summary_payload = _load_summary_payload(finetune_root / "summary.json")
    discovered = _scan_dataset_task_dirs(finetune_root)
    return discovered, summary_payload


def _order_discovered_dataset_names(
    dataset_task_dirs: dict[str, list[Path]],
    summary_payload: dict[str, Any] | None,
) -> list[str]:
    discovered_names = list(dataset_task_dirs.keys())
    if summary_payload is None:
        return discovered_names

    ordered: list[str] = []
    resolved = summary_payload.get("resolved_datasets")
    if isinstance(resolved, list):
        ordered.extend(
            _dedupe_preserve_order(
                [str(name) for name in resolved if str(name).strip() and str(name) in dataset_task_dirs]
            )
        )
    dataset_name = summary_payload.get("dataset_name")
    if isinstance(dataset_name, str) and dataset_name.strip() in dataset_task_dirs:
        ordered.append(dataset_name.strip())

    ordered = _dedupe_preserve_order(ordered)
    ordered.extend([name for name in discovered_names if name not in set(ordered)])
    return ordered


def _select_dataset_names(
    discovered: list[str],
    requested: list[str] | None,
) -> tuple[list[str], list[str]]:
    if requested is None:
        return discovered, []
    missing = [name for name in requested if name not in discovered]
    selected = [name for name in requested if name in discovered]
    return selected, missing


def _compute_normalized_predictions(
    y_context: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_context = np.asarray(y_context, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mean = float(y_context.mean())
    std = float(max(y_context.std(), 1e-6))
    return (y_true - mean) / std, (y_pred - mean) / std


def _aggregate_overall_metrics(
    dataset_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fold_means: list[dict[str, float]] = []
    fold_values: dict[int, dict[str, list[float]]] = {}
    for dataset_result in dataset_results.values():
        for fold_result in dataset_result.get("folds", []):
            fold_index = int(fold_result.get("fold_index", 1))
            metric_dict = fold_result.get("metrics", {})
            bucket = fold_values.setdefault(fold_index, {"R2": [], "RMSE": [], "MAE": []})
            for key in ("R2", "RMSE", "MAE"):
                if key in metric_dict:
                    bucket[key].append(float(metric_dict[key]))

    for fold_index in sorted(fold_values):
        metric_dict: dict[str, float] = {}
        for key in ("R2", "RMSE", "MAE"):
            values = fold_values[fold_index][key]
            if values:
                metric_dict[key] = float(np.mean(values))
        if metric_dict:
            fold_means.append(metric_dict)
    return aggregate_metric_dicts(fold_means)


def _evaluate_task_checkpoint(
    *,
    dataset_name: str,
    task_dir: Path,
    device: torch.device,
    num_sampling_steps: int,
    sampling_method: str,
    ddim_eta: float,
    n_repeats: int,
    use_cache: bool,
) -> dict[str, Any]:
    checkpoint_path = task_dir / "last.pt"
    split_path = task_dir / "split.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    split_payload, task_split = load_task_split_json(
        split_path,
        use_cache=use_cache,
        verbose=False,
    )
    payload_dataset = str(split_payload.get("dataset_name", dataset_name))
    if payload_dataset != dataset_name:
        raise ValueError(
            f"split.json dataset mismatch: directory={dataset_name}, "
            f"payload={payload_dataset}"
        )

    model, process, _config = _load_checkpoint(checkpoint_path, device)
    result = evaluate_fixed_split(
        model=model,
        process=process,
        device=device,
        task_split=task_split,
        num_sampling_steps=int(num_sampling_steps),
        sampling_method=str(sampling_method),
        ddim_eta=float(ddim_eta),
        random_state=int(split_payload.get("random_state", 0)),
        n_repeats=int(n_repeats),
    )

    y_true = np.asarray(result["y_true"], dtype=np.float32)
    y_pred_mean = np.asarray(result["y_pred_mean"], dtype=np.float32)
    y_true_norm, y_pred_norm = _compute_normalized_predictions(
        task_split.y_outer_context,
        y_true,
        y_pred_mean,
    )

    return {
        "dataset_name": dataset_name,
        "fold_index": int(split_payload.get("fold_index", 1)),
        "n_splits": int(split_payload.get("n_splits", 1)),
        "outer_splitter": str(split_payload.get("outer_splitter", "shuffle_split")),
        "metrics": result["metrics"],
        "y_true": result["y_true"],
        "y_pred_repetitions": result["y_pred_repetitions"],
        "y_pred_mean": result["y_pred_mean"],
        "y_true_normalized": y_true_norm.astype(np.float32).tolist(),
        "y_pred_normalized": y_pred_norm.astype(np.float32).tolist(),
        "n_repeats": int(result["n_repeats"]),
        "num_sampling_steps": int(result["num_sampling_steps"]),
        "sampling_method": str(result["sampling_method"]),
        "ddim_eta": float(result.get("ddim_eta", 0.0)),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "split_path": str(split_path.resolve()),
    }


def _concatenate_repeat_predictions(fold_results: list[dict[str, Any]]) -> list[list[float]]:
    if not fold_results:
        return []
    repeat_count = int(fold_results[0]["n_repeats"])
    chunks: list[list[np.ndarray]] = [[] for _ in range(repeat_count)]
    for fold_result in fold_results:
        fold_repetitions = fold_result["y_pred_repetitions"]
        if len(fold_repetitions) != repeat_count:
            raise ValueError("Inconsistent repeat counts across fold evaluations.")
        for rep_index, values in enumerate(fold_repetitions):
            chunks[rep_index].append(np.asarray(values, dtype=np.float32))
    return [
        np.concatenate(values, axis=0).astype(np.float32).tolist()
        for values in chunks
    ]


def _aggregate_dataset_result(
    *,
    dataset_name: str,
    fold_results: list[dict[str, Any]],
    plot_dataset: str | None,
    plot_dir: Path,
) -> dict[str, Any]:
    ordered_folds = sorted(
        fold_results,
        key=lambda item: (
            int(item.get("fold_index", 1)),
            str(item.get("checkpoint_path", "")),
        ),
    )
    metrics = aggregate_metric_dicts(
        [fold["metrics"] for fold in ordered_folds if "metrics" in fold]
    )
    y_true = np.concatenate(
        [np.asarray(fold["y_true"], dtype=np.float32) for fold in ordered_folds],
        axis=0,
    ).astype(np.float32)
    y_pred_mean = np.concatenate(
        [np.asarray(fold["y_pred_mean"], dtype=np.float32) for fold in ordered_folds],
        axis=0,
    ).astype(np.float32)
    y_true_norm = np.concatenate(
        [np.asarray(fold["y_true_normalized"], dtype=np.float32) for fold in ordered_folds],
        axis=0,
    ).astype(np.float32)
    y_pred_norm = np.concatenate(
        [np.asarray(fold["y_pred_normalized"], dtype=np.float32) for fold in ordered_folds],
        axis=0,
    ).astype(np.float32)

    plot_path: Path | None = None
    if plot_dataset is not None and dataset_name == plot_dataset:
        plot_path = _save_normalized_prediction_plot(
            dataset_name=dataset_name,
            y_true_norm=y_true_norm,
            y_pred_norm=y_pred_norm,
            plot_dir=plot_dir,
        )

    return {
        "dataset_name": dataset_name,
        "n_splits": int(max((fold.get("n_splits", 1) for fold in ordered_folds), default=1)),
        "completed_fold_indices": [int(fold.get("fold_index", 1)) for fold in ordered_folds],
        "metrics": metrics,
        "folds": ordered_folds,
        "y_true": y_true.tolist(),
        "y_pred_repetitions": _concatenate_repeat_predictions(ordered_folds),
        "y_pred_mean": y_pred_mean.tolist(),
        "y_true_normalized": y_true_norm.tolist(),
        "y_pred_normalized": y_pred_norm.tolist(),
        "n_repeats": int(ordered_folds[0]["n_repeats"]),
        "num_sampling_steps": int(ordered_folds[0]["num_sampling_steps"]),
        "sampling_method": str(ordered_folds[0]["sampling_method"]),
        "ddim_eta": float(ordered_folds[0].get("ddim_eta", 0.0)),
        "plot_path": str(plot_path.resolve()) if plot_path is not None else None,
    }


def main() -> dict[str, Any]:
    import argparse

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Evaluate task-specific TabICL_regression03 finetuned checkpoints."
    )
    parser.add_argument(
        "--finetune-root",
        type=str,
        required=True,
        help="Root directory produced by run_all_tasks_finetune.py.",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=500,
        help="Number of reverse diffusion steps.",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
        help="Sampling method: ddpm (stochastic) or ddim (deterministic).",
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="DDIM eta; ignored unless --sampling-method ddim.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable dataset caching.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset subset to evaluate.",
    )
    parser.add_argument(
        "--plot-dataset",
        type=str,
        default=None,
        help="Dataset name to save one normalized GT vs prediction plot for.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=str(HERE / "evaluation_plots"),
        help="Directory to save normalized prediction plot.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save detailed evaluation results JSON.",
    )

    args = parser.parse_args()
    finetune_root = Path(args.finetune_root).resolve()
    if not finetune_root.is_dir():
        raise SystemExit(f"Finetune root does not exist or is not a directory: {finetune_root}")
    if int(args.n_repeats) <= 0:
        raise SystemExit(f"n_repeats must be positive, got {args.n_repeats}")

    requested_datasets = parse_dataset_names(args.datasets)
    dataset_task_dirs, summary_payload = _discover_dataset_task_dirs(finetune_root)
    discovered_datasets = _order_discovered_dataset_names(
        dataset_task_dirs,
        summary_payload,
    )
    selected_datasets, missing_requested = _select_dataset_names(
        discovered_datasets,
        requested_datasets,
    )

    failures: list[dict[str, Any]] = []
    for dataset_name in missing_requested:
        failures.append(
            {
                "dataset_name": dataset_name,
                "error": "Requested dataset was not discovered under finetune-root.",
                "checkpoint_path": None,
                "split_path": None,
            }
        )

    if not selected_datasets and not failures:
        failures.append(
            {
                "dataset_name": None,
                "error": "No evaluable task directories were found under finetune-root.",
                "checkpoint_path": None,
                "split_path": None,
            }
        )

    print(f"Discovered {len(discovered_datasets)} task(s) under {finetune_root}", flush=True)
    if requested_datasets is not None:
        print(f"Selected {len(selected_datasets)} requested task(s).", flush=True)

    device = torch.device(args.device)
    dataset_results: dict[str, dict[str, Any]] = {}
    plot_dir = Path(args.plot_dir)
    iterator = selected_datasets
    if tqdm is not None:
        iterator = tqdm(selected_datasets, desc="Evaluating finetuned tasks", unit="task")

    for dataset_name in iterator:
        task_dirs = dataset_task_dirs.get(dataset_name, [])
        fold_results: list[dict[str, Any]] = []
        for task_dir in task_dirs:
            try:
                fold_results.append(
                    _evaluate_task_checkpoint(
                        dataset_name=dataset_name,
                        task_dir=task_dir,
                        device=device,
                        num_sampling_steps=int(args.num_sampling_steps),
                        sampling_method=str(args.sampling_method),
                        ddim_eta=float(args.ddim_eta),
                        n_repeats=int(args.n_repeats),
                        use_cache=not bool(args.no_cache),
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "dataset_name": dataset_name,
                        "fold_index": (
                            int(task_dir.name.split("_", 1)[1])
                            if task_dir.name.startswith("fold_")
                            and task_dir.name.split("_", 1)[1].isdigit()
                            else None
                        ),
                        "error": str(exc),
                        "checkpoint_path": str((task_dir / "last.pt").resolve()),
                        "split_path": str((task_dir / "split.json").resolve()),
                        "traceback": traceback.format_exc(),
                    }
                )
        if not fold_results:
            failures.append(
                {
                    "dataset_name": dataset_name,
                    "error": "No folds evaluated successfully for this dataset.",
                    "checkpoint_path": None,
                    "split_path": None,
                }
            )
            continue
        dataset_results[dataset_name] = _aggregate_dataset_result(
            dataset_name=dataset_name,
            fold_results=fold_results,
            plot_dataset=args.plot_dataset,
            plot_dir=plot_dir,
        )

    overall_metrics = _aggregate_overall_metrics(dataset_results)
    metrics: dict[str, float] = {}
    for dataset_name, result in dataset_results.items():
        metrics[f"{dataset_name}/R2"] = float(result["metrics"]["R2"])
        metrics[f"{dataset_name}/RMSE"] = float(result["metrics"]["RMSE"])
        metrics[f"{dataset_name}/MAE"] = float(result["metrics"]["MAE"])
    metrics.update(
        {
            key: value
            for key, value in overall_metrics.items()
            if isinstance(value, (float, int))
        }
    )

    if args.output_json is not None:
        payload = {
            "config": {
                "finetune_root": str(finetune_root),
                "device": args.device,
                "num_sampling_steps": int(args.num_sampling_steps),
                "sampling_method": args.sampling_method,
                "ddim_eta": float(args.ddim_eta),
                "n_repeats": int(args.n_repeats),
                "use_cache": not bool(args.no_cache),
                "datasets_requested": requested_datasets,
                "datasets_discovered": discovered_datasets,
                "datasets_loaded": list(dataset_results.keys()),
                "n_splits": {
                    dataset_name: int(result.get("n_splits", 1))
                    for dataset_name, result in dataset_results.items()
                },
                "checkpoint_name": "last.pt",
                "summary_json": (
                    str((finetune_root / "summary.json").resolve())
                    if (finetune_root / "summary.json").exists()
                    else None
                ),
            },
            "overall_metrics": overall_metrics,
            "datasets": dataset_results,
            "failures": failures,
        }
        write_json(args.output_json, payload)
        print(f"Saved evaluation details to {Path(args.output_json).resolve()}", flush=True)

    if not dataset_results:
        print("No datasets were evaluated successfully.", flush=True)
    else:
        print("\n" + "=" * 70)
        print(f"{'Dataset':<45} {'R2':>8} {'RMSE':>10} {'MAE':>10}")
        print("-" * 70)
        for dataset_name in selected_datasets:
            if dataset_name not in dataset_results:
                continue
            result = dataset_results[dataset_name]["metrics"]
            print(
                f"{dataset_name:<45} "
                f"{float(result['R2']):>8.4f} "
                f"{float(result['RMSE']):>10.4f} "
                f"{float(result['MAE']):>10.4f}"
            )
        print("-" * 70)
        print(f"{'Mean R2':<45} {float(overall_metrics.get('R2', float('nan'))):>8.4f}")
        print(f"{'Mean RMSE':<45} {float(overall_metrics.get('RMSE', float('nan'))):>8.4f}")
        print(f"{'Mean MAE':<45} {float(overall_metrics.get('MAE', float('nan'))):>8.4f}")
        print(
            f"{'Mean R2 +/- SE':<45} "
            f"{float(overall_metrics.get('R2', float('nan'))):>8.4f} +/- "
            f"{float(overall_metrics.get('R2_SE', float('nan'))):>8.4f}"
        )
        print(
            f"{'Mean RMSE +/- SE':<45} "
            f"{float(overall_metrics.get('RMSE', float('nan'))):>8.4f} +/- "
            f"{float(overall_metrics.get('RMSE_SE', float('nan'))):>8.4f}"
        )
        print(
            f"{'Mean MAE +/- SE':<45} "
            f"{float(overall_metrics.get('MAE', float('nan'))):>8.4f} +/- "
            f"{float(overall_metrics.get('MAE_SE', float('nan'))):>8.4f}"
        )
        print("=" * 70)

    if failures:
        print("\nFailures:", flush=True)
        for failure in failures:
            dataset_name = failure.get("dataset_name") or "<none>"
            print(f"  {dataset_name}: {failure.get('error')}", flush=True)
        raise SystemExit(1)

    return {
        "metrics": metrics,
        "overall_metrics": overall_metrics,
        "datasets": dataset_results,
        "summary_payload": summary_payload,
        "failures": failures,
    }


if __name__ == "__main__":
    main()
