from __future__ import annotations


"""
Plotting utility for sweep02_results.json produced by run_checkpoint_sweep02.py.

It renders:
  - Overall mean ROC AUC vs. training step (with std error bars).
  - Per-dataset mean ROC AUC vs. training step (one curve per dataset).
  - Two separate images: *_mean.png and *_datasets.png.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_plot_data(
    results: Dict[str, Dict],
) -> tuple[list[int], list[float], list[float], Dict[str, List[tuple[int, float]]]] | None:
    steps: List[int] = []
    overall_means: List[float] = []
    overall_stds: List[float] = []
    per_dataset_means: Dict[str, List[tuple[int, float]]] = {}

    for _, rec in results.items():
        step = rec.get("step")
        overall = rec.get("overall", {})
        mean_auc = overall.get("mean")
        std_auc = overall.get("std")
        if mean_auc is None or (isinstance(mean_auc, float) and np.isnan(mean_auc)):
            continue
        step_int = int(step) if step is not None else 0
        steps.append(step_int)
        overall_means.append(float(mean_auc))
        overall_stds.append(float(std_auc) if std_auc is not None else 0.0)

        for ds_name, ds_rec in rec.get("datasets", {}).items():
            ds_mean = ds_rec.get("mean")
            if ds_mean is None or (isinstance(ds_mean, float) and np.isnan(ds_mean)):
                continue
            per_dataset_means.setdefault(ds_name, []).append(
                (step_int, float(ds_mean))
            )

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
) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        steps,
        means,
        yerr=stds,
        fmt="-o",
        capsize=3,
        label="Overall (mean across datasets)",
    )
    plt.xlabel("Training step")
    plt.ylabel("ROC AUC")
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
) -> None:
    if not per_dataset:
        print("No per-dataset AUC values to plot; skipping dataset plot.")
        return
    plt.figure(figsize=(8, 5))
    for ds_name, vals in per_dataset.items():
        if not vals:
            continue
        ds_steps = [v[0] for v in vals]
        ds_means = [v[1] for v in vals]
        plt.plot(ds_steps, ds_means, "-o", label=ds_name)
    plt.xlabel("Training step")
    plt.ylabel("ROC AUC")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved dataset plot to {plot_path}")


def plot_results(results: Dict[str, Dict], output_path: Path) -> None:
    plot_data = _collect_plot_data(results)
    if not plot_data:
        print("No valid AUC values to plot.")
        return
    steps_sorted, overall_sorted, overall_std_sorted, per_dataset_sorted = plot_data
    mean_path, datasets_path = _split_plot_paths(output_path)
    title = "TabArena CV (5 folds x 5 repeats)"
    _plot_mean_curve(steps_sorted, overall_sorted, overall_std_sorted, mean_path, title)
    _plot_dataset_curves(per_dataset_sorted, datasets_path, title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sweep02_results.json")
    parser.add_argument(
        "--results-json",
        type=str,
        required=True,
        help="Path to sweep02_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=None,
        help=(
            "Base output PNG path. Two files are written: *_mean.png and *_datasets.png "
            "(default: alongside JSON as sweep02_plot.png)."
        ),
    )
    args = parser.parse_args()

    json_path = Path(args.results_json)
    output_path = Path(args.output) if args.output else json_path.with_name("sweep02_plot.png")

    results = load_results(json_path)
    plot_results(results, output_path)


if __name__ == "__main__":
    main()
