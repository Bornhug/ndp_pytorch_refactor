from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TABARENA_EVAL_PATH = HERE / "tabarena_eval" / "evaluation.py"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bar_distribution import (
    DEFAULT_BORDERS_PATH,
    DEFAULT_NUM_BARS,
    border_asset_metadata,
    build_distribution,
)
from model import NanoTabPFNRegressor


def _load_tabarena_eval_module():
    spec = importlib.util.spec_from_file_location(
        "nanoTabPFN_v10_orig02_tabarena_eval", TABARENA_EVAL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load evaluation module from {TABARENA_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TABARENA_EVAL = _load_tabarena_eval_module()

TABPFN_REGRESSION_DATASETS = _TABARENA_EVAL.TABPFN_REGRESSION_DATASETS
get_openml_datasets = _TABARENA_EVAL.get_openml_datasets
eval_model_cv = _TABARENA_EVAL.eval_model_cv
load_checkpoint = _TABARENA_EVAL.load_checkpoint
quantile_levels_for_bins = _TABARENA_EVAL.quantile_levels_for_bins
sharpness_quantile_levels = _TABARENA_EVAL.sharpness_quantile_levels

DEFAULT_MAX_FEATURES_EVAL = 32
DEFAULT_NEW_INSTANCES_EVAL = 1000
DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 0
DEFAULT_NUM_BINS = 10
DEFAULT_SHARPNESS_COVERAGE = 0.90


def evaluate_regressor_cv(
    regressor: NanoTabPFNRegressor,
    *,
    max_features_eval: int = DEFAULT_MAX_FEATURES_EVAL,
    new_instances_eval: int = DEFAULT_NEW_INSTANCES_EVAL,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    num_bins: int = DEFAULT_NUM_BINS,
    sharpness_coverage: float = DEFAULT_SHARPNESS_COVERAGE,
    borders_path: str | None = None,
    verbose: bool = False,
    return_details: bool = False,
) -> Dict[str, Dict] | tuple[Dict[str, Dict], Dict[str, Any]]:
    """
    Evaluate a NanoTabPFNRegressor with one fixed KFold CV run across a fixed
    OpenML regression dataset subset.

    The row subsample for each dataset is drawn once using ``random_state`` so
    it can match TabICL_regression when the same seed is used there.
    """
    datasets = get_openml_datasets(
        max_features_eval=max_features_eval,
        new_instances_eval=new_instances_eval,
        random_state=random_state,
        verbose=verbose,
    )
    if not datasets:
        scores = {
            "overall": {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan")},
            "datasets": {},
        }
        if return_details:
            return scores, {"overall_metrics": {}, "datasets": {}}
        return scores

    eval_result = eval_model_cv(
        regressor.model,
        datasets,
        device=regressor.device,
        n_splits=n_splits,
        random_state=random_state,
        num_bins=num_bins,
        sharpness_coverage=sharpness_coverage,
        bar_distribution=build_distribution(
            borders_path,
            expected_num_bars=regressor.model.num_bars,
        ),
        return_details=return_details,
    )
    if return_details:
        metrics, details = eval_result
    else:
        metrics = eval_result
        details = None

    overall_scores = {
        "r2": float(metrics.get("R2", float("nan"))),
        "r2_se": float(metrics.get("R2_SE", float("nan"))),
        "r2_folds": [
            float(v) for v in metrics.get("R2_FOLD_VALUES", [])
        ],
        "rmse": float(metrics.get("RMSE", float("nan"))),
        "rmse_se": float(metrics.get("RMSE_SE", float("nan"))),
        "rmse_folds": [
            float(v) for v in metrics.get("RMSE_FOLD_VALUES", [])
        ],
        "mae": float(metrics.get("MAE", float("nan"))),
        "mae_se": float(metrics.get("MAE_SE", float("nan"))),
        "mae_folds": [
            float(v) for v in metrics.get("MAE_FOLD_VALUES", [])
        ],
        "qice": float(metrics.get("QICE", float("nan"))),
        "qice_se": float(metrics.get("QICE_SE", float("nan"))),
        "nmpiw": float(metrics.get("NMPIW", float("nan"))),
        "nmpiw_se": float(metrics.get("NMPIW_SE", float("nan"))),
    }
    per_dataset: Dict[str, Dict[str, float]] = {}
    for key, val in metrics.items():
        if key.endswith("/R2"):
            ds_name = key[: -len("/R2")]
            per_dataset.setdefault(ds_name, {})["r2"] = float(val)
        elif key.endswith("/RMSE"):
            ds_name = key[: -len("/RMSE")]
            per_dataset.setdefault(ds_name, {})["rmse"] = float(val)
        elif key.endswith("/MAE"):
            ds_name = key[: -len("/MAE")]
            per_dataset.setdefault(ds_name, {})["mae"] = float(val)
        elif key.endswith("/QICE"):
            ds_name = key[: -len("/QICE")]
            per_dataset.setdefault(ds_name, {})["qice"] = float(val)
        elif key.endswith("/MPIW"):
            ds_name = key[: -len("/MPIW")]
            per_dataset.setdefault(ds_name, {})["mpiw"] = float(val)
        elif key.endswith("/NMPIW"):
            ds_name = key[: -len("/NMPIW")]
            per_dataset.setdefault(ds_name, {})["nmpiw"] = float(val)
    scores = {"overall": overall_scores, "datasets": per_dataset}
    if return_details:
        return scores, details
    return scores


def _split_plot_paths(plot_path: Path) -> tuple[Path, Path]:
    suffix = plot_path.suffix or ".png"
    stem = plot_path.stem if plot_path.suffix else plot_path.name
    mean_path = plot_path.parent / f"{stem}_mean{suffix}"
    datasets_path = plot_path.parent / f"{stem}_datasets{suffix}"
    return mean_path, datasets_path


def save_eval_plots(
    eval_history,
    output_path: str | None = None,
    metric: str = "r2",
):
    if not eval_history:
        print("No eval history to plot.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required to plot eval history.")
        return

    times: list[float] = []
    overall_values: list[float] = []
    per_dataset_values: dict[str, list[tuple[float, float]]] = {}
    for t, scores in eval_history:
        overall = scores.get("overall") if isinstance(scores, dict) else None
        datasets_scores = scores.get("datasets") if isinstance(scores, dict) else None
        overall_scores = overall if overall is not None else scores
        value = (
            overall_scores.get(metric, float("nan"))
            if isinstance(overall_scores, dict)
            else float("nan")
        )
        times.append(t)
        overall_values.append(value)
        if isinstance(datasets_scores, dict):
            for name, ds_scores in datasets_scores.items():
                ds_value = ds_scores.get(metric, float("nan"))
                per_dataset_values.setdefault(name, []).append((t, ds_value))

    if not np.isfinite(overall_values).any():
        print(f"No finite values for metric '{metric}' to plot.")
        return

    if output_path is None:
        output_path = HERE / "train_eval_r2.png"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_path, datasets_path = _split_plot_paths(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(times, overall_values, "-o", label=f"nanoTabPFN (mean {metric})")
    plt.xlabel("Training time (seconds)")
    plt.ylabel(metric.upper())
    plt.title("nanoTabPFN regression evaluation during training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(mean_path, dpi=150)
    plt.close()
    print(f"Saved mean plot to {mean_path}")

    if per_dataset_values:
        plt.figure(figsize=(8, 5))
        for name, vals in per_dataset_values.items():
            if not vals:
                continue
            ds_times = [v[0] for v in vals]
            ds_values = [v[1] for v in vals]
            plt.plot(ds_times, ds_values, "-o", label=name)
        plt.xlabel("Training time (seconds)")
        plt.ylabel(metric.upper())
        plt.title(f"nanoTabPFN regression evaluation ({metric}) per dataset")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(datasets_path, dpi=150)
        plt.close()
        print(f"Saved dataset plot to {datasets_path}")
    else:
        print("No per-dataset values to plot.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a nanoTabPFN regression checkpoint on OpenML CV."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-features-eval", type=int, default=DEFAULT_MAX_FEATURES_EVAL)
    parser.add_argument("--new-instances-eval", type=int, default=DEFAULT_NEW_INSTANCES_EVAL)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--embedding-size", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--mlp-hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-bars", type=int, default=None)
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument("--sharpness-coverage", type=float, default=DEFAULT_SHARPNESS_COVERAGE)
    parser.add_argument("--borders-path", type=str, default=str(DEFAULT_BORDERS_PATH))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path for TabICL-style detailed evaluation JSON.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    model = load_checkpoint(
        checkpoint_path,
        device,
        embedding_size=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        mlp_hidden_size=args.mlp_hidden_size,
        num_layers=args.num_layers,
        num_bars=args.num_bars,
    )
    border_meta = border_asset_metadata(
        args.borders_path,
        expected_num_bars=int(model.num_bars),
    )
    regressor = NanoTabPFNRegressor(
        model,
        device,
        bar_distribution=build_distribution(
            args.borders_path,
            expected_num_bars=int(model.num_bars),
        ),
    )
    eval_result = evaluate_regressor_cv(
        regressor,
        max_features_eval=args.max_features_eval,
        new_instances_eval=args.new_instances_eval,
        n_splits=args.n_splits,
        random_state=args.random_state,
        num_bins=args.num_bins,
        sharpness_coverage=args.sharpness_coverage,
        borders_path=args.borders_path,
        verbose=args.verbose,
        return_details=args.output_json is not None,
    )
    if args.output_json is not None:
        scores, details = eval_result
        payload = {
            "config": {
                "checkpoint": str(checkpoint_path.resolve()),
                "device": args.device,
                "max_features_eval": int(args.max_features_eval),
                "new_instances_eval": int(args.new_instances_eval),
                "n_splits": int(args.n_splits),
                "random_state": int(args.random_state),
                "num_bins": int(args.num_bins),
                "sharpness_coverage": float(args.sharpness_coverage),
                "target_distribution": "FullSupportBarDistribution",
                **border_meta,
                "datasets_loaded": list(details.get("datasets", {}).keys()),
            },
            "overall_metrics": details.get("overall_metrics", {}),
            "overall_uncertainty": details.get("overall_uncertainty", {}),
            "datasets": details.get("datasets", {}),
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved evaluation details to {output_path}")
    else:
        scores = eval_result
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
