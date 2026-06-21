from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bar_distribution import DEFAULT_NUM_BARS, FullSupportBarDistribution, build_distribution
from model import NanoTabPFNModel, NanoTabPFNRegressor

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

DEFAULT_MAX_FEATURES_EVAL = 32
DEFAULT_NEW_INSTANCES_EVAL = 1000
DEFAULT_RANDOM_STATE = 0


# TabPFN-paper regression benchmark IDs (from TabICL_regression utilities).
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


def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """Drop constants, ordinal-encode categoricals, coerce numerics."""
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue

        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors="coerce").notna().sum()
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline(
        [
            (
                "to_pandas",
                FunctionTransformer(
                    lambda x: pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x
                ),
            ),
            (
                "to_numeric",
                FunctionTransformer(
                    lambda x: x.apply(pd.to_numeric, errors="coerce").to_numpy()
                ),
            ),
        ]
    )
    cat_transformer = Pipeline(
        [
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_mask),
            ("cat", cat_transformer, cat_mask),
        ]
    )


def get_openml_datasets(
    *,
    max_features_eval: int = DEFAULT_MAX_FEATURES_EVAL,
    new_instances_eval: int = DEFAULT_NEW_INSTANCES_EVAL,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load regression datasets and return dataset_name -> (X, y)."""
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    items = list(TABPFN_REGRESSION_DATASETS.items())

    for i, (name, data_id) in enumerate(items, start=1):
        if verbose:
            print(f"[{i}/{len(items)}] Fetching {name} (OpenML {data_id}) ...", flush=True)

        try:
            bunch = fetch_openml(data_id=data_id, as_frame=True)
        except Exception as exc:
            if verbose:
                print(f"  Skipped: failed to fetch ({exc})", flush=True)
            continue

        X_df = bunch.data
        y_series = bunch.target

        X_np = X_df.to_numpy(copy=True)
        preprocessor = get_feature_preprocessor(X_np)
        X_np = preprocessor.fit_transform(X_np)

        y_np = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(y_np)
        X_np = np.asarray(X_np[valid], dtype=np.float32)
        y_np = np.asarray(y_np[valid], dtype=np.float32)

        if X_np.shape[1] > max_features_eval:
            if verbose:
                print(
                    f"  Skipped: too many features ({X_np.shape[1]} > {max_features_eval})",
                    flush=True,
                )
            continue

        if len(y_np) < 10:
            if verbose:
                print("  Skipped: dataset too small", flush=True)
            continue

        if new_instances_eval < len(y_np):
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(y_np), size=new_instances_eval, replace=False)
            X_sub = X_np[idx]
            y_sub = y_np[idx]
        else:
            X_sub = X_np
            y_sub = y_np

        X_sub = np.nan_to_num(X_sub, nan=0.0)
        datasets[name] = (X_sub, y_sub)
        if verbose:
            print(
                f"  Added: {name} | X.shape={X_sub.shape} | y.shape={y_sub.shape}",
                flush=True,
            )

    return datasets


def _resolve_model_config(
    *,
    config: Optional[Dict],
    embedding_size: Optional[int],
    num_attention_heads: Optional[int],
    mlp_hidden_size: Optional[int],
    num_layers: Optional[int],
    num_bars: Optional[int],
) -> Dict[str, int]:
    cfg = {}
    if config:
        model_cfg = config.get("model", config)
        cfg = {
            "embedding_size": model_cfg.get("embedding_size"),
            "num_attention_heads": model_cfg.get("num_attention_heads"),
            "mlp_hidden_size": model_cfg.get("mlp_hidden_size"),
            "num_layers": model_cfg.get("num_layers"),
            "num_bars": model_cfg.get("num_bars"),
        }

    overrides = {
        "embedding_size": embedding_size,
        "num_attention_heads": num_attention_heads,
        "mlp_hidden_size": mlp_hidden_size,
        "num_layers": num_layers,
        "num_bars": num_bars,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val

    missing = [k for k, v in cfg.items() if v is None]
    if missing:
        raise ValueError(
            "Missing model config values: "
            + ", ".join(missing)
            + ". Provide CLI overrides or save config in the checkpoint."
        )

    out = {k: int(v) for k, v in cfg.items()}
    return out


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    *,
    embedding_size: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    mlp_hidden_size: Optional[int] = None,
    num_layers: Optional[int] = None,
    num_bars: Optional[int] = None,
) -> NanoTabPFNModel:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, NanoTabPFNModel):
        model = ckpt.to(device)
        model.eval()
        return model

    state_dict = None
    config = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            config = ckpt.get("config")
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            config = ckpt.get("config")
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        else:
            config = ckpt.get("config")
    if state_dict is None:
        raise ValueError(
            "Unsupported checkpoint format. Provide a state_dict, or a dict with "
            "'state_dict' and optional 'config'."
        )

    model_cfg = _resolve_model_config(
        config=config,
        embedding_size=embedding_size,
        num_attention_heads=num_attention_heads,
        mlp_hidden_size=mlp_hidden_size,
        num_layers=num_layers,
        num_bars=num_bars,
    )

    model = NanoTabPFNModel(
        embedding_size=model_cfg["embedding_size"],
        num_attention_heads=model_cfg["num_attention_heads"],
        mlp_hidden_size=model_cfg["mlp_hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_bars=model_cfg["num_bars"],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _standard_error(values: list[float]) -> float:
    if not values:
        return float("nan")
    values_np = np.asarray(values, dtype=np.float64)
    return float(np.std(values_np) / np.sqrt(values_np.shape[0]))


def _aggregate_metric_values(values: list[float]) -> Dict[str, Any]:
    values = [float(v) for v in values]
    if not values:
        return {"mean": float("nan"), "se": float("nan"), "values": []}
    return {
        "mean": float(np.mean(values)),
        "se": _standard_error(values),
        "values": values,
    }


def quantile_levels_for_bins(num_bins: int) -> np.ndarray:
    bins = int(num_bins)
    if bins <= 1:
        raise ValueError(f"num_bins must be greater than 1, got {num_bins}")
    return np.arange(1, bins, dtype=np.float64) / float(bins)


def sharpness_quantile_levels(coverage: float) -> np.ndarray:
    coverage_float = float(coverage)
    if not 0.0 < coverage_float < 1.0:
        raise ValueError(f"sharpness coverage must be in (0, 1), got {coverage}")
    tail_mass = (1.0 - coverage_float) / 2.0
    return np.asarray([tail_mass, 1.0 - tail_mass], dtype=np.float64)


def _as_1d_float_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def compute_qice_from_boundaries(
    y_true: Any,
    quantile_boundaries: Any,
    *,
    num_bins: int,
) -> Dict[str, Any]:
    y_true_np = _as_1d_float_array(y_true, name="y_true")
    boundaries = np.asarray(quantile_boundaries, dtype=np.float64)
    if boundaries.ndim == 1:
        boundaries = boundaries.reshape(1, -1)
    if boundaries.ndim != 2:
        raise ValueError(
            "quantile_boundaries must have shape [num_bins - 1, targets], "
            f"got {boundaries.shape}"
        )
    if boundaries.shape[1] != y_true_np.shape[0]:
        raise ValueError(
            "quantile_boundaries target count must match y_true length, "
            f"got {boundaries.shape[1]} and {y_true_np.shape[0]}"
        )
    bins = int(num_bins)
    if boundaries.shape[0] != bins - 1:
        raise ValueError(f"expected {bins - 1} boundaries, got {boundaries.shape[0]}")

    finite_mask = np.isfinite(y_true_np) & np.isfinite(boundaries).all(axis=0)
    excluded_count = int(y_true_np.shape[0] - np.count_nonzero(finite_mask))
    y_valid = y_true_np[finite_mask]
    boundaries_valid = boundaries[:, finite_mask]
    if y_valid.shape[0] == 0:
        raise ValueError("QICE requires at least one finite target and boundary column")

    bin_indices = np.empty(y_valid.shape[0], dtype=np.int64)
    for target_idx, target in enumerate(y_valid):
        bin_indices[target_idx] = int(
            np.searchsorted(boundaries_valid[:, target_idx], target, side="left")
        )
    bin_counts = np.bincount(bin_indices, minlength=bins).astype(np.int64)
    bin_proportions = bin_counts.astype(np.float64) / float(y_valid.shape[0])
    ideal_proportion = 1.0 / float(bins)
    qice = float(np.mean(np.abs(bin_proportions - ideal_proportion)))

    return {
        "QICE": qice,
        "bin_proportions": bin_proportions.tolist(),
        "bin_counts": bin_counts.tolist(),
        "bin_indices": bin_indices.astype(np.int64).tolist(),
        "target_count": int(y_valid.shape[0]),
        "quantile_levels": quantile_levels_for_bins(bins).tolist(),
        "ideal_proportion": ideal_proportion,
        "excluded_non_finite_count": excluded_count,
    }


def compute_sharpness_from_boundaries(
    y_true: Any,
    sharpness_boundaries: Any,
    *,
    coverage: float,
) -> Dict[str, Any]:
    y_true_np = _as_1d_float_array(y_true, name="y_true")
    boundaries = np.asarray(sharpness_boundaries, dtype=np.float64)
    if boundaries.shape[0] != 2:
        raise ValueError(
            "sharpness_boundaries must have shape [2, targets], "
            f"got {boundaries.shape}"
        )
    if boundaries.shape[1] != y_true_np.shape[0]:
        raise ValueError(
            "sharpness boundary target count must match y_true length, "
            f"got {boundaries.shape[1]} and {y_true_np.shape[0]}"
        )
    finite_mask = np.isfinite(y_true_np) & np.isfinite(boundaries).all(axis=0)
    excluded_count = int(y_true_np.shape[0] - np.count_nonzero(finite_mask))
    y_valid = y_true_np[finite_mask]
    boundaries_valid = boundaries[:, finite_mask]
    if y_valid.shape[0] == 0:
        raise ValueError("sharpness requires at least one finite target and interval")

    widths = boundaries_valid[1] - boundaries_valid[0]
    mpiw = float(np.mean(widths))
    target_std = float(np.std(y_valid))
    nmpiw = float(mpiw / target_std) if np.isfinite(target_std) and target_std > 0 else float("nan")
    levels = sharpness_quantile_levels(float(coverage))
    return {
        "MPIW": mpiw,
        "NMPIW": nmpiw,
        "target_std": target_std,
        "sharpness_coverage": float(coverage),
        "lower_quantile": float(levels[0]),
        "upper_quantile": float(levels[1]),
        "sharpness_target_count": int(y_valid.shape[0]),
        "sharpness_excluded_non_finite_count": excluded_count,
    }


def eval_model_cv(
    model: NanoTabPFNModel,
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    n_splits: int = 5,
    random_state: int = 0,
    num_bins: int = 10,
    sharpness_coverage: float = 0.90,
    bar_distribution: FullSupportBarDistribution | None = None,
    return_details: bool = False,
) -> Dict[str, Any] | tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate with KFold CV and optionally return TabICL-style details."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    bins = int(num_bins)
    quantile_levels = quantile_levels_for_bins(bins)
    sharpness_levels = sharpness_quantile_levels(float(sharpness_coverage))
    bar_distribution = (
        bar_distribution
        if bar_distribution is not None
        else build_distribution(expected_num_bars=model.num_bars)
    )

    metrics: Dict[str, Any] = {}
    details: Dict[str, Any] | None = {"datasets": {}} if return_details else None
    items = list(datasets.items())
    iterator = items
    if tqdm is not None:
        iterator = tqdm(items, desc="Evaluating datasets", unit="dataset")

    dataset_r2_values: list[float] = []
    dataset_rmse_values: list[float] = []
    dataset_mae_values: list[float] = []
    dataset_qice_values: list[float] = []
    dataset_mpiw_values: list[float] = []
    dataset_nmpiw_values: list[float] = []
    fold_r2_values: list[list[float]] = [[] for _ in range(int(n_splits))]
    fold_rmse_values: list[list[float]] = [[] for _ in range(int(n_splits))]
    fold_mae_values: list[list[float]] = [[] for _ in range(int(n_splits))]

    for dataset_name, (X, y) in iterator:
        if len(y) < n_splits:
            continue

        targets = []
        predictions = []
        boundary_chunks: list[np.ndarray] = []
        sharpness_boundary_chunks: list[np.ndarray] = []
        dataset_fold_r2_values: list[float] = []
        dataset_fold_rmse_values: list[float] = []
        dataset_fold_mae_values: list[float] = []
        dataset_fold_qice_values: list[float] = []
        dataset_fold_mpiw_values: list[float] = []
        dataset_fold_nmpiw_values: list[float] = []
        fold_logs: list[Dict[str, Any]] = []

        regressor = NanoTabPFNRegressor(
            model,
            device,
            bar_distribution=bar_distribution,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            regressor.fit(X_train, y_train)
            pred_info = regressor.predict_distribution(
                X_test,
                quantile_levels=quantile_levels,
                sharpness_levels=sharpness_levels,
            )
            y_pred = pred_info["y_pred_mean"]
            quantile_boundaries = pred_info["quantile_boundaries"]
            sharpness_boundaries = pred_info["sharpness_boundaries"]

            targets.append(y_test)
            predictions.append(y_pred)
            boundary_chunks.append(quantile_boundaries)
            sharpness_boundary_chunks.append(sharpness_boundaries)

            fold_metrics = {
                "R2": float(r2_score(y_test, y_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "MAE": float(mean_absolute_error(y_test, y_pred)),
            }
            fold_uncertainty = compute_qice_from_boundaries(
                y_test,
                quantile_boundaries,
                num_bins=bins,
            )
            fold_sharpness = compute_sharpness_from_boundaries(
                y_test,
                sharpness_boundaries,
                coverage=float(sharpness_coverage),
            )
            dataset_fold_r2_values.append(fold_metrics["R2"])
            dataset_fold_rmse_values.append(fold_metrics["RMSE"])
            dataset_fold_mae_values.append(fold_metrics["MAE"])
            dataset_fold_qice_values.append(fold_uncertainty["QICE"])
            dataset_fold_mpiw_values.append(fold_sharpness["MPIW"])
            dataset_fold_nmpiw_values.append(fold_sharpness["NMPIW"])
            fold_r2_values[fold_idx].append(fold_metrics["R2"])
            fold_rmse_values[fold_idx].append(fold_metrics["RMSE"])
            fold_mae_values[fold_idx].append(fold_metrics["MAE"])
            fold_logs.append(
                {
                    "fold_index": int(fold_idx),
                    "test_indices": np.asarray(test_idx, dtype=np.int64).tolist(),
                    "y_true": np.asarray(y_test, dtype=np.float32).tolist(),
                    "y_pred_mean": np.asarray(y_pred, dtype=np.float32).tolist(),
                    "quantile_levels": quantile_levels.tolist(),
                    "quantile_boundaries": quantile_boundaries.astype(np.float32).tolist(),
                    "sharpness_quantile_levels": sharpness_levels.tolist(),
                    "sharpness_boundaries": sharpness_boundaries.astype(np.float32).tolist(),
                    "bin_indices": fold_uncertainty["bin_indices"],
                    "bin_counts": fold_uncertainty["bin_counts"],
                    "bin_proportions": fold_uncertainty["bin_proportions"],
                    "metrics": fold_metrics,
                    "uncertainty": {
                        "QICE": fold_uncertainty["QICE"],
                        "target_count": fold_uncertainty["target_count"],
                        "excluded_non_finite_count": fold_uncertainty[
                            "excluded_non_finite_count"
                        ],
                    },
                    "sharpness": fold_sharpness,
                }
            )

        y_true = np.concatenate(targets, axis=0)
        y_pred = np.concatenate(predictions, axis=0)
        quantile_boundaries_np = np.concatenate(boundary_chunks, axis=1).astype(np.float32)
        sharpness_boundaries_np = np.concatenate(
            sharpness_boundary_chunks,
            axis=1,
        ).astype(np.float32)

        r2_summary = _aggregate_metric_values(dataset_fold_r2_values)
        rmse_summary = _aggregate_metric_values(dataset_fold_rmse_values)
        mae_summary = _aggregate_metric_values(dataset_fold_mae_values)
        dataset_metrics = {
            "R2": float(r2_score(y_true, y_pred)),
            "R2_SE": r2_summary["se"],
            "R2_FOLD_VALUES": r2_summary["values"],
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "RMSE_SE": rmse_summary["se"],
            "RMSE_FOLD_VALUES": rmse_summary["values"],
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "MAE_SE": mae_summary["se"],
            "MAE_FOLD_VALUES": mae_summary["values"],
        }
        dataset_uncertainty = compute_qice_from_boundaries(
            y_true,
            quantile_boundaries_np,
            num_bins=bins,
        )
        dataset_uncertainty["QICE_SE"] = _standard_error(dataset_fold_qice_values)
        dataset_uncertainty["QICE_FOLD_VALUES"] = [
            float(v) for v in dataset_fold_qice_values
        ]
        dataset_sharpness = compute_sharpness_from_boundaries(
            y_true,
            sharpness_boundaries_np,
            coverage=float(sharpness_coverage),
        )
        dataset_sharpness["MPIW_SE"] = _standard_error(dataset_fold_mpiw_values)
        dataset_sharpness["MPIW_FOLD_VALUES"] = [
            float(v) for v in dataset_fold_mpiw_values
        ]
        dataset_sharpness["NMPIW_SE"] = _standard_error(dataset_fold_nmpiw_values)
        dataset_sharpness["NMPIW_FOLD_VALUES"] = [
            float(v) for v in dataset_fold_nmpiw_values
        ]

        metrics[f"{dataset_name}/R2"] = dataset_metrics["R2"]
        metrics[f"{dataset_name}/RMSE"] = dataset_metrics["RMSE"]
        metrics[f"{dataset_name}/MAE"] = dataset_metrics["MAE"]
        metrics[f"{dataset_name}/QICE"] = dataset_uncertainty["QICE"]
        metrics[f"{dataset_name}/MPIW"] = dataset_sharpness["MPIW"]
        metrics[f"{dataset_name}/NMPIW"] = dataset_sharpness["NMPIW"]
        metrics[f"{dataset_name}/R2_SE"] = dataset_metrics["R2_SE"]
        metrics[f"{dataset_name}/RMSE_SE"] = dataset_metrics["RMSE_SE"]
        metrics[f"{dataset_name}/MAE_SE"] = dataset_metrics["MAE_SE"]

        dataset_r2_values.append(dataset_metrics["R2"])
        dataset_rmse_values.append(dataset_metrics["RMSE"])
        dataset_mae_values.append(dataset_metrics["MAE"])
        dataset_qice_values.append(dataset_uncertainty["QICE"])
        dataset_mpiw_values.append(dataset_sharpness["MPIW"])
        dataset_nmpiw_values.append(dataset_sharpness["NMPIW"])

        if details is not None:
            details["datasets"][dataset_name] = {
                "metrics": dataset_metrics,
                "uncertainty": dataset_uncertainty,
                "sharpness": dataset_sharpness,
                "y_true": np.asarray(y_true, dtype=np.float32).tolist(),
                "y_pred_mean": np.asarray(y_pred, dtype=np.float32).tolist(),
                "bin_indices": dataset_uncertainty["bin_indices"],
                "folds": fold_logs,
            }

    fold_r2_overall = [float(np.mean(vals)) for vals in fold_r2_values if vals]
    fold_rmse_overall = [float(np.mean(vals)) for vals in fold_rmse_values if vals]
    fold_mae_overall = [float(np.mean(vals)) for vals in fold_mae_values if vals]
    overall_r2 = _aggregate_metric_values(dataset_r2_values)
    overall_rmse = _aggregate_metric_values(dataset_rmse_values)
    overall_mae = _aggregate_metric_values(dataset_mae_values)
    overall_qice = _aggregate_metric_values(dataset_qice_values)
    overall_mpiw = _aggregate_metric_values(dataset_mpiw_values)
    overall_nmpiw = _aggregate_metric_values(
        [value for value in dataset_nmpiw_values if np.isfinite(value)]
    )

    if dataset_r2_values:
        metrics["R2"] = overall_r2["mean"]
        metrics["R2_SE"] = overall_r2["se"]
        metrics["R2_DATASET_VALUES"] = overall_r2["values"]
        metrics["R2_FOLD_VALUES"] = [float(v) for v in fold_r2_overall]
    if dataset_rmse_values:
        metrics["RMSE"] = overall_rmse["mean"]
        metrics["RMSE_SE"] = overall_rmse["se"]
        metrics["RMSE_DATASET_VALUES"] = overall_rmse["values"]
        metrics["RMSE_FOLD_VALUES"] = [float(v) for v in fold_rmse_overall]
    if dataset_mae_values:
        metrics["MAE"] = overall_mae["mean"]
        metrics["MAE_SE"] = overall_mae["se"]
        metrics["MAE_DATASET_VALUES"] = overall_mae["values"]
        metrics["MAE_FOLD_VALUES"] = [float(v) for v in fold_mae_overall]
    if dataset_qice_values:
        metrics["QICE"] = overall_qice["mean"]
        metrics["QICE_SE"] = overall_qice["se"]
    if dataset_nmpiw_values:
        metrics["NMPIW"] = overall_nmpiw["mean"]
        metrics["NMPIW_SE"] = overall_nmpiw["se"]

    if details is not None:
        details["overall_metrics"] = {
            "R2": metrics.get("R2"),
            "R2_SE": metrics.get("R2_SE"),
            "R2_DATASET_VALUES": metrics.get("R2_DATASET_VALUES"),
            "RMSE": metrics.get("RMSE"),
            "RMSE_SE": metrics.get("RMSE_SE"),
            "RMSE_DATASET_VALUES": metrics.get("RMSE_DATASET_VALUES"),
            "MAE": metrics.get("MAE"),
            "MAE_SE": metrics.get("MAE_SE"),
            "MAE_DATASET_VALUES": metrics.get("MAE_DATASET_VALUES"),
        }
        details["overall_uncertainty"] = {
            "QICE": overall_qice["mean"],
            "QICE_SE": overall_qice["se"],
            "QICE_DATASET_VALUES": overall_qice["values"],
            "NMPIW": overall_nmpiw["mean"],
            "NMPIW_SE": overall_nmpiw["se"],
            "NMPIW_DATASET_VALUES": overall_nmpiw["values"],
            "MPIW_DATASET_VALUES": overall_mpiw["values"],
            "dataset_count": len(dataset_qice_values),
            "num_bins": bins,
            "quantile_levels": quantile_levels.tolist(),
            "sharpness_coverage": float(sharpness_coverage),
            "sharpness_quantile_levels": sharpness_levels.tolist(),
            "ideal_proportion": 1.0 / float(bins),
        }
        return metrics, details
    return metrics
