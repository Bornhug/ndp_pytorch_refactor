"""
TabPFN-paper regression benchmark evaluation for NDP regression.

Evaluates on the 29 regression datasets from Extended Data Table 4 of the
TabPFN paper (AutoML / OpenML-CTR23 benchmarks).

Protocol:
  - Fetch datasets by OpenML ID.
  - Preprocess: ordinal-encode categoricals, coerce numerics.
  - Optionally subsample large datasets to a fixed number of instances.
  - Repeated evaluation on fixed data splits by default (1 split x 20 repetitions).
  - Report R2, RMSE, MAE per dataset and mean across datasets.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

try:
    import openml
except Exception:
    openml = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_diffusion_processes.regressor import NDPRegressorWrapper
from tabicl_style.utils import amp_dtype_name, resolve_amp_settings

SKLEARN_DATA_HOME = Path(
    os.environ.get("SCIKIT_LEARN_DATA", str(ROOT / ".sklearn_data"))
).resolve()

DATASET_CACHE_DIR = Path(
    os.environ.get("TABARENA_CACHE", str(HERE / ".tabarena_cache"))
).resolve()

# ---------------------------------------------------------------------------
# TabPFN paper regression datasets (Extended Data Table 4)
# ---------------------------------------------------------------------------
TABPFN_REGRESSION_DATASETS: Dict[str, int] = {
    "abalone": 42726,
    "airfoil_self_noise": 44957,
    "auction_verification": 44958,
    "boston": 531,
    "cars": 44994,
    "colleges": 42727,
    "concrete_compressive_strength": 44959,
    "cpu_activity": 44978,
    "energy_efficiency": 44960,
    "geographical_origin_of_music": 44965,
    "grid_stability": 44973,
    "house_prices_nominal": 42563,
    "kin8nm": 44980,
    "Mercedes_Benz_Greener_Manufacturing": 42570,
    "MIP-2016-regression": 43071,
    "Moneyball": 41021,
    "pumadyn32nh": 44981,
    "QSAR_fish_toxicity": 44970,
    "quake": 550,
    "SAT11-HAND-runtime-regression": 41980,
    "sensory": 546,
    "socmob": 541,
    "space_ga": 507,
    "student_performance": 44967,
    "tecator": 505,
    "topo_2_1": 422,
    "us_crime": 42730,
    "yprop_4_1": 416,
}

DEFAULT_MAX_FEATURES_EVAL = 32
DEFAULT_NEW_INSTANCES_EVAL = 0
DEFAULT_MAX_ROWS_EVAL = 1000


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _get_cache_path(dataset_name: str) -> Path:
    """Return the on-disk pickle cache path for one OpenML dataset."""
    return DATASET_CACHE_DIR / f"reg_{dataset_name}.pkl"


def _load_from_cache(dataset_name: str, *, verbose: bool = False) -> Optional[Dict]:
    """Load preprocessed ``(X, y)`` data from cache if the cache is usable."""
    cache_path = _get_cache_path(dataset_name)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        if verbose:
            print(f"  Loaded from cache: {cache_path}", flush=True)
        return cached
    except Exception as e:
        if verbose:
            print(f"  Cache load failed for {dataset_name}: {e}", flush=True)
        return None


def _save_to_cache(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    verbose: bool = False,
) -> None:
    """Persist preprocessed dataset arrays so later evaluations skip OpenML fetch."""
    DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_cache_path(dataset_name)
    cached = {"X": X, "y": y, "dataset_name": dataset_name}
    try:
        with cache_path.open("wb") as f:
            pickle.dump(cached, f)
        if verbose:
            print(f"  Saved to cache: {cache_path}", flush=True)
    except Exception as e:
        if verbose:
            print(f"  Cache save failed for {dataset_name}: {e}", flush=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing pipeline used before NDP evaluation.

    Constant columns are dropped by excluding them from both masks. Columns that
    can be fully coerced to numeric are kept numeric; the remaining columns are
    ordinal-encoded with unknown categories mapped to NaN.
    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan = X[col].dropna().unique()
        if len(unique_non_nan) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_count = X[col].notna().sum()
        numeric_count = pd.to_numeric(X[col], errors="coerce").notna().sum()
        num_mask.append(non_nan_count == numeric_count)
        cat_mask.append(non_nan_count != numeric_count)

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(
            lambda x: pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x
        )),
        ("to_numeric", FunctionTransformer(
            lambda x: x.apply(pd.to_numeric, errors="coerce").to_numpy()
        )),
    ])
    cat_transformer = Pipeline([
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )),
    ])

    return ColumnTransformer(transformers=[
        ("num", num_transformer, num_mask),
        ("cat", cat_transformer, cat_mask),
    ])


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _fetch_openml_regression(
    dataset_name: str,
    data_id: int,
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch and preprocess one benchmark regression dataset from OpenML.

    The returned arrays are float32, rows with missing target values are
    removed, and feature preprocessing mirrors the TabPFN-style benchmark path.
    """
    data_home = str(SKLEARN_DATA_HOME)
    try:
        bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=data_home)
    except ValueError as e:
        if "md5 checksum" not in str(e).lower():
            raise
        cache_dir = SKLEARN_DATA_HOME / "openml"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        if verbose:
            print(f"  Cleared sklearn cache after md5 mismatch; retrying.", flush=True)
        bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=data_home)

    X_df = bunch.data
    y_series = bunch.target

    # Preprocess features
    X_np = X_df.to_numpy(copy=True)
    preprocessor = get_feature_preprocessor(X_np)
    X_np = preprocessor.fit_transform(X_np)

    # y: ensure float
    y_np = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=np.float64)

    # Drop rows with NaN in y
    valid = np.isfinite(y_np)
    X_np = np.asarray(X_np[valid], dtype=np.float32)
    y_np = np.asarray(y_np[valid], dtype=np.float32)

    return X_np, y_np


def get_regression_datasets(
    *,
    max_features_eval: int = DEFAULT_MAX_FEATURES_EVAL,
    new_instances_eval: int = DEFAULT_NEW_INSTANCES_EVAL,
    max_rows_eval: int = DEFAULT_MAX_ROWS_EVAL,
    random_state: int = 0,
    verbose: bool = True,
    use_cache: bool = True,
    dataset_names: list[str] | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load TabPFN regression benchmark datasets.

    Each dataset is loaded from cache when possible, otherwise fetched from
    OpenML and cached after preprocessing. Datasets that exceed the feature or
    row-count limits are skipped, and remaining datasets are optionally
    subsampled with a fixed RNG seed when ``new_instances_eval`` is positive.

    Returns:
        dict mapping dataset_name -> (X, y) where X is float32, y is float32.
    """
    if dataset_names is None:
        registry = TABPFN_REGRESSION_DATASETS
    else:
        registry = {
            name: TABPFN_REGRESSION_DATASETS[name]
            for name in dataset_names
            if name in TABPFN_REGRESSION_DATASETS
        }

    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    total = len(registry)
    cache_hits = 0
    cache_misses = 0

    for i, (name, data_id) in enumerate(registry.items(), start=1):
        if verbose:
            print(f"[{i}/{total}] Processing {name} (OpenML ID={data_id}) ...", flush=True)

        cached = None
        if use_cache:
            cached = _load_from_cache(name, verbose=verbose)

        if cached is not None:
            cache_hits += 1
            X = cached["X"]
            y = cached["y"]
        else:
            cache_misses += 1
            try:
                X, y = _fetch_openml_regression(name, data_id, verbose=verbose)
                if use_cache:
                    _save_to_cache(name, X, y, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"  FAILED to fetch {name}: {e}", flush=True)
                continue

        if X.shape[1] > max_features_eval:
            if verbose:
                print(
                    f"  Skipped: too many features ({X.shape[1]} > {max_features_eval})",
                    flush=True,
                )
            continue

        if max_rows_eval > 0 and len(y) > max_rows_eval:
            if verbose:
                print(
                    f"  Skipped: too many rows ({len(y)} > {max_rows_eval})",
                    flush=True,
                )
            continue

        # Subsample only when a positive row cap is requested.
        if new_instances_eval > 0 and new_instances_eval < len(y):
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(y), size=new_instances_eval, replace=False)
            X = X[idx]
            y = y[idx]

        # Drop any remaining NaN features (fill with 0)
        X = np.nan_to_num(X, nan=0.0)

        datasets[name] = (X, y)
        if verbose:
            print(
                f"  Added: {name} | X.shape={X.shape} | y.shape={y.shape} "
                f"(kept so far: {len(datasets)})",
                flush=True,
            )

    if verbose:
        print(f"\nCache stats: {cache_hits} hits, {cache_misses} misses", flush=True)

    return datasets


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute the regression metrics reported for each fold and dataset."""
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def _standard_error(values: list[float]) -> float:
    """Return the standard error over split-level metric values."""
    if not values:
        return float("nan")
    values_np = np.asarray(values, dtype=np.float64)
    return float(np.std(values_np) / np.sqrt(values_np.shape[0]))


def eval_model(
    model: NDPRegressorWrapper,
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    n_splits: int = 1,
    n_repeats: int = 20,
    random_state: int = 0,
    return_details: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, Any]]:
    """Evaluate with repeated CV on fixed splits.

    Repetitions are used as Monte Carlo samples on the same split(s): predictions
    are averaged first, then metrics are computed from the averaged prediction.
    If n_splits <= 1, use one fixed random holdout split.

    When ``return_details`` is true, also return fold-level indices,
    per-repetition predictions, averaged predictions, and metric details for
    JSON export.
    """
    metrics: Dict[str, float] = {}
    details: Dict[str, Any] | None = {"datasets": {}} if return_details else None

    items = list(datasets.items())
    if int(n_repeats) <= 0:
        raise ValueError(f"n_repeats must be positive, got {n_repeats}")
    if int(n_splits) <= 1:
        split_label = "split"
        split_count = 1
    else:
        split_label = "fold"
        split_count = int(n_splits)

    repeat_count = int(n_repeats)
    split_r2_values: list[list[float]] = [[] for _ in range(split_count)]
    split_rmse_values: list[list[float]] = [[] for _ in range(split_count)]
    split_mae_values: list[list[float]] = [[] for _ in range(split_count)]

    dataset_splits: Dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for dataset_name, (X, _y) in items:
        if int(n_splits) <= 1:
            splitter = ShuffleSplit(n_splits=1, test_size=0.5, random_state=int(random_state))
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
        dataset_splits[dataset_name] = list(splitter.split(X))

    iterator = items
    if tqdm is not None:
        iterator = tqdm(items, desc="Evaluating datasets", unit="dataset")

    for dataset_name, (X, y) in iterator:
        y_true_all = []
        y_pred_all = []
        rep_prediction_chunks = [[] for _ in range(repeat_count)]
        fold_logs = []

        fold_iterator = dataset_splits[dataset_name]
        if tqdm is not None:
            fold_iterator = tqdm(
                fold_iterator,
                desc=f"{dataset_name} {split_label}s",
                unit=split_label,
                leave=False,
                position=1,
            )

        for fold_idx, (train_idx, test_idx) in enumerate(fold_iterator, start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred_rep_np = model.predict_repeated(X_test, repeat_count)
            y_pred_mean = np.mean(y_pred_rep_np, axis=0)
            y_test_np = np.asarray(y_test, dtype=np.float32)

            y_true_all.append(y_test)
            y_pred_all.append(y_pred_mean)
            repeat_metrics = []
            for rep_idx in range(repeat_count):
                rep_prediction_chunks[rep_idx].append(y_pred_rep_np[rep_idx])
                rep_metrics = _compute_regression_metrics(
                    y_test_np,
                    np.asarray(y_pred_rep_np[rep_idx], dtype=np.float32),
                )
                repeat_metrics.append(rep_metrics)
            fold_metrics = _compute_regression_metrics(
                y_test_np,
                np.asarray(y_pred_mean, dtype=np.float32),
            )
            split_r2_values[fold_idx - 1].append(fold_metrics["R2"])
            split_rmse_values[fold_idx - 1].append(fold_metrics["RMSE"])
            split_mae_values[fold_idx - 1].append(fold_metrics["MAE"])
            fold_logs.append(
                {
                    "fold_index": int(fold_idx),
                    "test_indices": np.asarray(test_idx, dtype=np.int64).tolist(),
                    "y_true": y_test_np.tolist(),
                    "y_pred_repetitions": y_pred_rep_np.astype(np.float32).tolist(),
                    "repeat_metrics": repeat_metrics,
                    "y_pred_mean": np.asarray(y_pred_mean, dtype=np.float32).tolist(),
                    "metrics": fold_metrics,
                }
            )
        y_true_np = np.concatenate(y_true_all, axis=0).astype(np.float32)
        y_pred_np = np.concatenate(y_pred_all, axis=0).astype(np.float32)
        y_pred_rep_concat = [
            np.concatenate(chunks, axis=0).astype(np.float32).tolist()
            for chunks in rep_prediction_chunks
        ]

        dataset_metrics = _compute_regression_metrics(y_true_np, y_pred_np)

        metrics[f"{dataset_name}/R2"] = dataset_metrics["R2"]
        metrics[f"{dataset_name}/RMSE"] = dataset_metrics["RMSE"]
        metrics[f"{dataset_name}/MAE"] = dataset_metrics["MAE"]
        if details is not None:
            details["datasets"][dataset_name] = {
                "metrics": dataset_metrics,
                "y_true": y_true_np.tolist(),
                "y_pred_repetitions": y_pred_rep_concat,
                "y_pred_mean": y_pred_np.tolist(),
                "folds": fold_logs,
            }

    overall_r2_folds = [float(np.mean(vals)) for vals in split_r2_values if vals]
    overall_rmse_folds = [float(np.mean(vals)) for vals in split_rmse_values if vals]
    overall_mae_folds = [float(np.mean(vals)) for vals in split_mae_values if vals]

    if overall_r2_folds:
        metrics["R2"] = float(np.mean(overall_r2_folds))
        metrics["R2_SE"] = _standard_error(overall_r2_folds)
        metrics["R2_FOLD_VALUES"] = [float(v) for v in overall_r2_folds]
    if overall_rmse_folds:
        metrics["RMSE"] = float(np.mean(overall_rmse_folds))
        metrics["RMSE_SE"] = _standard_error(overall_rmse_folds)
        metrics["RMSE_FOLD_VALUES"] = [float(v) for v in overall_rmse_folds]
    if overall_mae_folds:
        metrics["MAE"] = float(np.mean(overall_mae_folds))
        metrics["MAE_SE"] = _standard_error(overall_mae_folds)
        metrics["MAE_FOLD_VALUES"] = [float(v) for v in overall_mae_folds]
    if details is not None:
        details["overall_metrics"] = {
            "R2": metrics.get("R2"),
            "R2_SE": metrics.get("R2_SE"),
            "R2_FOLD_VALUES": metrics.get("R2_FOLD_VALUES"),
            "RMSE": metrics.get("RMSE"),
            "RMSE_SE": metrics.get("RMSE_SE"),
            "RMSE_FOLD_VALUES": metrics.get("RMSE_FOLD_VALUES"),
            "MAE": metrics.get("MAE"),
            "MAE_SE": metrics.get("MAE_SE"),
            "MAE_FOLD_VALUES": metrics.get("MAE_FOLD_VALUES"),
        }
    if details is not None:
        return metrics, details
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    *,
    torch_compile: bool = False,
    torch_compile_mode: str | None = None,
):
    """Rebuild the training model/process pair and load checkpoint weights.

    Checkpoint configs are filtered against the current dataclasses so older
    checkpoints with removed fields can still be evaluated. EMA weights are
    preferred when present because they are the evaluation weights saved during
    training.
    """
    from tabicl_style.train import build_model_and_process, maybe_compile_model
    from tabicl_style.config import Config

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint["config"]

    # Reconstruct Config from dict. Filter nested keys so older checkpoints with
    # removed config fields can still be evaluated.
    default_config = Config()
    config = Config(
        **{
            k: type(getattr(default_config, k))(
                **{
                    item_key: item_value
                    for item_key, item_value in v.items()
                    if item_key
                    in {field.name for field in fields(type(getattr(default_config, k)))}
                }
            )
            if isinstance(v, dict)
            else v
            for k, v in config_dict.items()
            if hasattr(default_config, k)
        }
    )
    if torch_compile:
        config.training.torch_compile = True
    if torch_compile_mode is not None:
        config.training.torch_compile_mode = str(torch_compile_mode)

    model, process = build_model_and_process(config, device)

    # Prefer EMA weights if available
    if "ema_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    model = maybe_compile_model(model, config)
    return model, process, config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> Dict[str, float]:
    """Parse CLI options, run benchmark evaluation, print and optionally save metrics."""
    import argparse

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="TabPFN regression benchmark evaluation for NDP"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=500,
        help="Number of reverse diffusion steps",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
        help="Sampling method: ddpm (stochastic) or ddim (deterministic)",
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
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help="Enable CUDA autocast for model forwards during evaluation.",
    )
    amp_group.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        default=None,
        help="Disable evaluation autocast even if the checkpoint config used AMP.",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default=None,
        choices=["auto", "float16", "bfloat16", "float32"],
        help=(
            "Evaluation autocast dtype. Defaults to checkpoint config dtype; "
            "'auto' uses bfloat16 on supported CUDA GPUs, otherwise float16."
        ),
    )
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", type=str, default="reduce-overhead")
    parser.add_argument(
        "--max-features-eval",
        type=int,
        default=DEFAULT_MAX_FEATURES_EVAL,
    )
    parser.add_argument(
        "--max-rows-eval",
        type=int,
        default=DEFAULT_MAX_ROWS_EVAL,
        help="Skip datasets with more than this many rows; <=0 disables row filtering.",
    )
    parser.add_argument(
        "--new-instances-eval",
        type=int,
        default=DEFAULT_NEW_INSTANCES_EVAL,
        help="Maximum rows per dataset; <=0 uses all rows.",
    )
    parser.add_argument("--n-splits", type=int, default=1)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable dataset caching"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of dataset names (default: all 29)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "Optional path to save detailed evaluation results JSON "
            "(ground truth, repeated predictions, mean prediction, metrics)."
        ),
    )

    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)

    print(f"Loading checkpoint: {checkpoint_path}")
    model, process, config = _load_checkpoint(
        checkpoint_path,
        device,
        torch_compile=bool(args.torch_compile),
        torch_compile_mode=args.torch_compile_mode,
    )
    eval_amp_requested = (
        bool(getattr(config.training, "amp", False))
        if args.amp is None
        else bool(args.amp)
    )
    eval_amp_dtype_name = (
        str(getattr(config.training, "dtype", "float16"))
        if args.amp_dtype is None
        else str(args.amp_dtype)
    )
    eval_amp, eval_amp_dtype = resolve_amp_settings(
        eval_amp_requested,
        eval_amp_dtype_name,
        device=device,
    )
    if eval_amp:
        print(f"Evaluation AMP: enabled (dtype={amp_dtype_name(eval_amp_dtype)})")
    else:
        print("Evaluation AMP: not used")

    regressor = NDPRegressorWrapper(
        model=model,
        process=process,
        device=device,
        num_sampling_steps=args.num_sampling_steps,
        sampling_method=args.sampling_method,
        ddim_eta=args.ddim_eta,
        amp=eval_amp,
        amp_dtype=eval_amp_dtype,
    )

    dataset_names = None
    if args.datasets:
        dataset_names = [s.strip() for s in args.datasets.split(",")]

    print("Loading regression benchmark datasets...")
    datasets = get_regression_datasets(
        max_features_eval=args.max_features_eval,
        max_rows_eval=args.max_rows_eval,
        new_instances_eval=args.new_instances_eval,
        random_state=args.random_state,
        verbose=True,
        use_cache=not args.no_cache,
        dataset_names=dataset_names,
    )
    print(f"Loaded {len(datasets)} dataset(s).\n")

    eval_result = eval_model(
        regressor,
        datasets,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        return_details=args.output_json is not None,
    )
    if args.output_json is not None:
        metrics, details = eval_result
    else:
        metrics = eval_result
        details = None

    if not metrics:
        print("No datasets loaded; no metrics to report.")
        return {}

    if details is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "checkpoint": str(checkpoint_path.resolve()),
                "device": args.device,
                "num_sampling_steps": int(args.num_sampling_steps),
                "sampling_method": args.sampling_method,
                "ddim_eta": float(args.ddim_eta),
                "amp": bool(eval_amp),
                "amp_dtype": amp_dtype_name(eval_amp_dtype),
                "max_features_eval": int(args.max_features_eval),
                "max_rows_eval": int(args.max_rows_eval),
                "new_instances_eval": int(args.new_instances_eval),
                "n_splits": int(args.n_splits),
                "n_repeats": int(args.n_repeats),
                "random_state": int(args.random_state),
                "use_cache": not bool(args.no_cache),
                "datasets_requested": dataset_names,
                "datasets_loaded": list(datasets.keys()),
            },
            "overall_metrics": details["overall_metrics"],
            "datasets": details["datasets"],
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved evaluation details to {output_path}")

    # Print results
    print("\n" + "=" * 70)
    print(f"{'Dataset':<45} {'R2':>8} {'RMSE':>10} {'MAE':>10}")
    print("-" * 70)
    for name in TABPFN_REGRESSION_DATASETS:
        r2_key = f"{name}/R2"
        if r2_key not in metrics:
            continue
        print(
            f"{name:<45} "
            f"{metrics[f'{name}/R2']:>8.4f} "
            f"{metrics[f'{name}/RMSE']:>10.4f} "
            f"{metrics[f'{name}/MAE']:>10.4f}"
        )
    print("-" * 70)
    print(f"{'Mean R2':<45} {metrics.get('R2', float('nan')):>8.4f}")
    print(f"{'Mean RMSE':<45} {metrics.get('RMSE', float('nan')):>8.4f}")
    print(f"{'Mean MAE':<45} {metrics.get('MAE', float('nan')):>8.4f}")
    r2_se = metrics.get("R2_SE", float("nan"))
    rmse_se = metrics.get("RMSE_SE", float("nan"))
    mae_se = metrics.get("MAE_SE", float("nan"))
    print(
        f"{'Mean R2 +/- SE':<45} "
        f"{metrics.get('R2', float('nan')):>8.4f} +/- {r2_se:>8.4f}"
    )
    print(
        f"{'Mean RMSE +/- SE':<45} "
        f"{metrics.get('RMSE', float('nan')):>8.4f} +/- {rmse_se:>8.4f}"
    )
    print(
        f"{'Mean MAE +/- SE':<45} "
        f"{metrics.get('MAE', float('nan')):>8.4f} +/- {mae_se:>8.4f}"
    )
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    main()
