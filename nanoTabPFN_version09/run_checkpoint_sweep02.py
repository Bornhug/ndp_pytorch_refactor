from __future__ import annotations


"""
Checkpoint sweep using TabArena-style evaluation (evaluation02.py) with
5x5 runs (5-fold CV, 5 repetitions with different seeds).

For each checkpoint under a given directory (model_step_*.pt), this script:
  - loads the checkpoint
  - evaluates on TabArena tasks via evaluation02 (5-fold CV)
  - repeats the full CV 5 times with different seeds (affects data split + sampling)
  - reports the mean ROC AUC across the 5 repetitions (each repetition already averages folds)
  - saves a JSON log with per-checkpoint metrics
"""

import json
from pathlib import Path
from typing import Dict, List
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from evaluation02 import (
    get_openml_datasets,
    eval_model,
    _load_checkpoint,
    NanoTabPFNClassifier,
)


def _sorted_checkpoints(checkpoints_dir: Path) -> List[Path]:
    ckpts = sorted(
        checkpoints_dir.glob("model_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return ckpts


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sweep_checkpoints(
    checkpoints_dir: Path,
    *,
    device: torch.device,
    sampling_method: str = "ddim",
    num_sampling_steps: int = 5,
    n_repeats: int = 5,
    n_splits: int = 5,
    max_features_eval: int = 10,
    new_instances_eval: int = 200,
    target_classes_filter: int = 2,
    output_json: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all checkpoints in checkpoints_dir with evaluation02 protocol.
    Returns mapping: checkpoint_path -> metrics dict aggregated over repeats, including per-dataset means.
    """
    ckpts = _sorted_checkpoints(checkpoints_dir)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {checkpoints_dir}")

    results: Dict[str, Dict[str, float]] = {}

    for ckpt_path in ckpts:
        step = int(ckpt_path.stem.split("_")[-1])
        print(f"\nEvaluating checkpoint {ckpt_path.name} (step={step})")

        # Load once; reuse model/process per repetition
        model, process, config_dict = _load_checkpoint(ckpt_path, device)
        classifier = NanoTabPFNClassifier(
            model=model,
            process=process,
            device=device,
            num_classes=config_dict["model"]["num_outputs"],
            sampling_method=sampling_method,
            num_sampling_steps=num_sampling_steps,
            show_progress=False,
        )

        # Load datasets once (deterministic filtering/subsample is seed-dependent via train_test_split)
        # We reload per repetition after seeding to vary subsample/splits.
        repeat_overall: List[float] = []
        per_dataset_repeats: Dict[str, List[float]] = {}
        for rep in range(n_repeats):
            seed = 1000 + rep  # different initial uniform / randomness
            _set_global_seed(seed)
            print(f"  Repeat {rep + 1}/{n_repeats} with seed {seed}")

            datasets = get_openml_datasets(
                max_features_eval=max_features_eval,
                new_instances_eval=new_instances_eval,
                target_classes_filter=target_classes_filter,
                random_state=seed,
                verbose=True,
            )
            if not datasets:
                print("    No datasets after filtering; skipping this repeat.")
                continue

            metrics = eval_model(
                classifier,
                datasets,
                n_splits=n_splits,
                random_state=seed,
            )
            # Overall ROC AUC (mean across datasets)
            if "ROC AUC" in metrics:
                repeat_overall.append(metrics["ROC AUC"])
            # Per-dataset ROC AUC keys end with "/ROC AUC"
            for key, val in metrics.items():
                if key.endswith("/ROC AUC"):
                    per_dataset_repeats.setdefault(key, []).append(val)

        def _aggregate(vals: List[float]) -> Dict[str, float | List[float]]:
            if not vals:
                return {"mean": float("nan"), "std": float("nan"), "repeats": vals}
            return {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "repeats": [float(v) for v in vals],
            }

        datasets_agg: Dict[str, Dict[str, float | List[float]]] = {}
        for ds_name, vals in per_dataset_repeats.items():
            datasets_agg[ds_name] = _aggregate(vals)

        overall_agg = _aggregate(repeat_overall)

        results[str(ckpt_path)] = {
            "step": step,
            "overall": overall_agg,
            "datasets": datasets_agg,
        }

        print(
            f"  Mean ROC AUC over {len(repeat_overall)} repeats (overall): "
            f"{overall_agg['mean']:.6f} (std {overall_agg['std']:.6f})"
        )

        # Flush JSON after each checkpoint if requested
        if output_json is not None:
            out_path = Path(output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved intermediate results to {out_path}")

    return results


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
        steps.append(int(step) if step is not None else 0)
        overall_means.append(float(mean_auc))
        overall_stds.append(float(std_auc) if std_auc is not None else 0.0)

        for ds_name, ds_rec in rec.get("datasets", {}).items():
            ds_mean = ds_rec.get("mean")
            if ds_mean is None or (isinstance(ds_mean, float) and np.isnan(ds_mean)):
                continue
            per_dataset_means.setdefault(ds_name, []).append(
                (int(step) if step is not None else 0, float(ds_mean))
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint sweep with evaluation02 (TabArena CV, repeats)")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Directory containing model_step_*.pt checkpoints",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="ddim",
        choices=["direct", "ddpm", "ddim"],
        help="Sampling method for inference",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=5,
        help="Number of sampling steps for DDIM/DDPM (ignored for direct)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-repeats", type=int, default=20, help="Number of repetitions (different seeds)")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds per repetition")
    parser.add_argument("--max-features-eval", type=int, default=10)
    parser.add_argument("--new-instances-eval", type=int, default=200)
    parser.add_argument("--target-classes-filter", type=int, default=2)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save sweep results as JSON",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help=(
            "Optional base path for plots (PNG). When set, outputs *_mean.png and *_datasets.png. "
            "Defaults to <checkpoints-dir>/sweep02_roc_auc.png when set to 'auto'."
        ),
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoints_dir = Path(args.checkpoints_dir)

    results = sweep_checkpoints(
        checkpoints_dir,
        device=device,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        n_repeats=args.n_repeats,
        n_splits=args.n_splits,
        max_features_eval=args.max_features_eval,
        new_instances_eval=args.new_instances_eval,
        target_classes_filter=args.target_classes_filter,
        output_json=Path(args.output_json) if args.output_json else None,
    )

    # Final save (in case no checkpoints were processed or to ensure latest state)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {out_path}")

    # Optional plotting (mean curve and per-dataset curves)
    if args.plot_path:
        if args.plot_path.lower() == "auto":
            plot_path = checkpoints_dir / "sweep02_roc_auc.png"
        else:
            plot_path = Path(args.plot_path)
            plot_path.parent.mkdir(parents=True, exist_ok=True)

        plot_data = _collect_plot_data(results)
        if not plot_data:
            print("No valid AUC values to plot; skipping plot.")
        else:
            steps_sorted, overall_sorted, overall_std_sorted, per_dataset_sorted = plot_data
            mean_path, datasets_path = _split_plot_paths(plot_path)
            title = f"TabArena CV ({args.n_splits} folds x {args.n_repeats} repeats)"
            _plot_mean_curve(steps_sorted, overall_sorted, overall_std_sorted, mean_path, title)
            _plot_dataset_curves(per_dataset_sorted, datasets_path, title)


if __name__ == "__main__":
    main()
