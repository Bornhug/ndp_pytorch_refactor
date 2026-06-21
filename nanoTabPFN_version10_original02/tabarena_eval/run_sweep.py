from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation import (
    DEFAULT_MAX_FEATURES_EVAL,
    DEFAULT_NEW_INSTANCES_EVAL,
    DEFAULT_RANDOM_STATE,
    eval_model_cv,
    get_openml_datasets,
    load_checkpoint,
)


def _checkpoint_step(path: Path) -> int:
    name = path.stem
    if name.startswith("step-"):
        return int(name.split("-")[-1])
    if name.startswith("model_step_"):
        return int(name.split("_")[-1])
    tail = name.split("_")[-1]
    return int(tail) if tail.isdigit() else 0


def _sorted_checkpoints(run_dir: Path) -> List[Path]:
    ckpts = list(run_dir.glob("step-*.pt"))
    ckpts.extend(run_dir.glob("model_step_*.pt"))
    if not ckpts:
        ckpts = list(run_dir.glob("*.pt"))
    return sorted(ckpts, key=_checkpoint_step)


def _aggregate(values: List[float]) -> Dict[str, float | List[float]]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "repeats": values}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "repeats": [float(v) for v in values],
    }


def _init_metric_lists() -> Dict[str, List[float]]:
    return {"R2": [], "RMSE": [], "MAE": []}


def sweep_checkpoints(
    run_dir: Path,
    *,
    device: torch.device,
    embedding_size: int | None = None,
    num_attention_heads: int | None = None,
    mlp_hidden_size: int | None = None,
    num_layers: int | None = None,
    num_bars: int | None = None,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_features_eval: int = DEFAULT_MAX_FEATURES_EVAL,
    new_instances_eval: int = DEFAULT_NEW_INSTANCES_EVAL,
    output_json: Path | None = None,
) -> Dict[str, Dict]:
    ckpts = _sorted_checkpoints(run_dir)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {run_dir}")

    print("Loading regression benchmark datasets once for the full sweep...")
    datasets = get_openml_datasets(
        max_features_eval=max_features_eval,
        new_instances_eval=new_instances_eval,
        random_state=random_state,
        verbose=False,
    )
    if not datasets:
        raise RuntimeError("No datasets available after filtering.")

    results: Dict[str, Dict] = {}

    for ckpt_path in ckpts:
        step = _checkpoint_step(ckpt_path)
        print(f"\nEvaluating checkpoint {ckpt_path.name} (step={step})")

        model = load_checkpoint(
            ckpt_path,
            device,
            embedding_size=embedding_size,
            num_attention_heads=num_attention_heads,
            mlp_hidden_size=mlp_hidden_size,
            num_layers=num_layers,
            num_bars=num_bars,
        )

        metrics = eval_model_cv(
            model,
            datasets,
            device=device,
            n_splits=n_splits,
            random_state=random_state,
        )

        datasets_agg: Dict[str, Dict[str, Dict[str, float | List[float]]]] = {}
        for key, val in metrics.items():
            for metric_name in ["R2", "RMSE", "MAE"]:
                suffix = f"/{metric_name}"
                if key.endswith(suffix):
                    ds_name = key[: -len(suffix)]
                    ds_rec = datasets_agg.setdefault(ds_name, {})
                    ds_rec[metric_name] = _aggregate([float(val)])

        overall_agg = {
            metric_name: _aggregate([float(metrics[metric_name])])
            if metric_name in metrics
            else _aggregate([])
            for metric_name in ["R2", "RMSE", "MAE"]
        }

        results[str(ckpt_path)] = {
            "step": step,
            "overall": overall_agg,
            "datasets": datasets_agg,
        }

        print(
            "  Metrics: "
            f"R2={overall_agg['R2']['mean']:.6f}, "
            f"RMSE={overall_agg['RMSE']['mean']:.6f}, "
            f"MAE={overall_agg['MAE']['mean']:.6f}"
        )

        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with output_json.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved intermediate results to {output_json}")

    return results


def _collect_plot_data(results: Dict[str, Dict], metric: str):
    steps: List[int] = []
    overall_means: List[float] = []
    overall_stds: List[float] = []
    per_dataset_means: Dict[str, List[tuple[int, float]]] = {}

    for _, rec in results.items():
        step = rec.get("step")
        overall = rec.get("overall", {})
        metric_rec = overall.get(metric, {}) if isinstance(overall, dict) else {}
        mean_val = metric_rec.get("mean") if isinstance(metric_rec, dict) else None
        std_val = metric_rec.get("std") if isinstance(metric_rec, dict) else None

        if mean_val is None or (isinstance(mean_val, float) and np.isnan(mean_val)):
            continue

        step_int = int(step) if step is not None else 0
        steps.append(step_int)
        overall_means.append(float(mean_val))
        overall_stds.append(float(std_val) if std_val is not None else 0.0)

        datasets = rec.get("datasets", {})
        if isinstance(datasets, dict):
            for ds_name, ds_rec in datasets.items():
                metric_ds = ds_rec.get(metric, {}) if isinstance(ds_rec, dict) else {}
                ds_mean = metric_ds.get("mean") if isinstance(metric_ds, dict) else None
                if ds_mean is None or (isinstance(ds_mean, float) and np.isnan(ds_mean)):
                    continue
                per_dataset_means.setdefault(ds_name, []).append((step_int, float(ds_mean)))

    if not steps:
        return None

    order = np.argsort(steps)
    steps_sorted = [steps[i] for i in order]
    overall_sorted = [overall_means[i] for i in order]
    overall_std_sorted = [overall_stds[i] for i in order]

    per_dataset_sorted: Dict[str, List[tuple[int, float]]] = {}
    for ds_name, vals in per_dataset_means.items():
        if vals:
            per_dataset_sorted[ds_name] = sorted(vals, key=lambda x: x[0])

    return steps_sorted, overall_sorted, overall_std_sorted, per_dataset_sorted


def _split_plot_paths(plot_path: Path) -> tuple[Path, Path]:
    suffix = plot_path.suffix or ".png"
    stem = plot_path.stem if plot_path.suffix else plot_path.name
    mean_path = plot_path.parent / f"{stem}_mean{suffix}"
    datasets_path = plot_path.parent / f"{stem}_datasets{suffix}"
    return mean_path, datasets_path


def _plot_mean_curve(
    steps: List[int],
    means: List[float],
    stds: List[float],
    plot_path: Path,
    title: str,
    metric: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        steps,
        means,
        yerr=stds,
        fmt="-o",
        capsize=3,
        label=f"Overall ({metric})",
    )
    plt.xlabel("Training step")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved mean plot to {plot_path}")


def _plot_dataset_curves(
    per_dataset: Dict[str, List[tuple[int, float]]],
    plot_path: Path,
    title: str,
    metric: str,
) -> None:
    if not per_dataset:
        print("No per-dataset values to plot; skipping dataset plot.")
        return

    plt.figure(figsize=(8, 5))
    for ds_name, vals in per_dataset.items():
        if not vals:
            continue
        ds_steps = [v[0] for v in vals]
        ds_means = [v[1] for v in vals]
        plt.plot(ds_steps, ds_means, "-o", label=ds_name)

    plt.xlabel("Training step")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved dataset plot to {plot_path}")


def _plot_results(results: Dict[str, Dict], plot_path: Path, title: str, metric: str) -> None:
    plot_data = _collect_plot_data(results, metric=metric)
    if not plot_data:
        print(f"No valid values for metric '{metric}' to plot.")
        return

    steps_sorted, overall_sorted, overall_std_sorted, per_dataset_sorted = plot_data
    mean_path, datasets_path = _split_plot_paths(plot_path)
    _plot_mean_curve(steps_sorted, overall_sorted, overall_std_sorted, mean_path, title, metric)
    _plot_dataset_curves(per_dataset_sorted, datasets_path, title, metric)


def _load_results_json(json_path: Path) -> Dict[str, Dict]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep nanoTabPFN regression checkpoints with OpenML CV."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Directory containing step-*.pt or model_step_*.pt checkpoints.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--max-features-eval", type=int, default=DEFAULT_MAX_FEATURES_EVAL)
    parser.add_argument("--new-instances-eval", type=int, default=DEFAULT_NEW_INSTANCES_EVAL)
    parser.add_argument("--embedding-size", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--mlp-hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-bars", type=int, default=None)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save sweep results as JSON.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help=(
            "Optional base path for plots (PNG). When set, outputs *_mean.png and *_datasets.png. "
            "Use 'auto' for default."
        ),
    )
    parser.add_argument(
        "--plot-metric",
        type=str,
        default="R2",
        choices=["R2", "RMSE", "MAE"],
        help="Metric to plot.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip evaluation and plot from an existing JSON results file.",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoints_dir = Path(args.checkpoints_dir)
    output_json = (
        Path(args.output_json)
        if args.output_json
        else checkpoints_dir / "evaluation" / "sweep_results.json"
    )

    plot_path = None
    if args.plot_path:
        if args.plot_path.lower() == "auto":
            plot_path = checkpoints_dir / "evaluation" / "sweep_regression.png"
        else:
            plot_path = Path(args.plot_path)

    if args.plot_only:
        if output_json is None or not output_json.exists():
            raise FileNotFoundError(
                "plot-only requested but results JSON does not exist. "
                "Provide --output-json pointing to an existing file."
            )
        if plot_path is None:
            plot_path = checkpoints_dir / "evaluation" / "sweep_regression.png"
        results = _load_results_json(output_json)
        title = f"Regression CV ({args.n_splits} folds)"
        _plot_results(results, plot_path, title, metric=args.plot_metric)
        return

    results = sweep_checkpoints(
        checkpoints_dir,
        device=device,
        embedding_size=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        mlp_hidden_size=args.mlp_hidden_size,
        num_layers=args.num_layers,
        num_bars=args.num_bars,
        n_splits=args.n_splits,
        random_state=args.random_state,
        max_features_eval=args.max_features_eval,
        new_instances_eval=args.new_instances_eval,
        output_json=output_json,
    )

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output_json}")

    if plot_path is not None:
        title = f"Regression CV ({args.n_splits} folds)"
        _plot_results(results, plot_path, title, metric=args.plot_metric)


if __name__ == "__main__":
    main()
