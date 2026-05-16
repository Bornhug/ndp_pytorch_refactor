"""Shared helpers for fixed-task finetuning on TabICL_regression03."""

from __future__ import annotations

import copy
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import KFold, ShuffleSplit
from torch import nn

try:
    import wandb
except Exception:
    wandb = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_diffusion_processes.process import loss as diffusion_loss
from neural_diffusion_processes.types import Batch
from tabicl_style.config import Config
from tabicl_style.evaluation import (
    DEFAULT_MAX_FEATURES_EVAL,
    DEFAULT_NEW_INSTANCES_EVAL,
    NDPRegressorWrapper,
    TABPFN_REGRESSION_DATASETS,
    _compute_regression_metrics,
    _standard_error,
    _fetch_openml_regression,
    _load_from_cache,
    _save_to_cache,
    get_regression_datasets,
)
from tabicl_style.train import EMA, build_model_and_process, compute_grad_norm, compute_lr
from tabicl_style.utils import normalize_y, set_seed

OUTER_TEST_FRACTION = 0.2


@dataclass
class TaskSplit:
    dataset_name: str
    random_state: int
    max_features_eval: int
    new_instances_eval: int
    full_num_rows: int
    num_features: int
    outer_split_random_state: int
    inner_split_random_state: int
    subsample_indices: list[int]
    outer_context_indices: list[int]
    outer_target_indices: list[int]
    finetune_context_indices: list[int]
    finetune_target_indices: list[int]
    X_subsample: np.ndarray
    y_subsample: np.ndarray
    X_outer_context: np.ndarray
    y_outer_context: np.ndarray
    X_outer_target: np.ndarray
    y_outer_target: np.ndarray
    X_finetune_context: np.ndarray
    y_finetune_context: np.ndarray
    X_finetune_target: np.ndarray
    y_finetune_target: np.ndarray
    n_splits: int = 1
    fold_index: int = 1
    outer_splitter: str = "shuffle_split"


@dataclass
class TaskDatasetView:
    dataset_name: str
    random_state: int
    max_features_eval: int
    new_instances_eval: int
    full_num_rows: int
    num_features: int
    subsample_indices: np.ndarray
    X_subsample: np.ndarray
    y_subsample: np.ndarray


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def parse_dataset_names(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [name.strip() for name in str(raw).split(",") if name.strip()]
    return names or None


def _require_valid_dataset_names(dataset_names: list[str] | None) -> None:
    if dataset_names is None:
        return
    invalid = [name for name in dataset_names if name not in TABPFN_REGRESSION_DATASETS]
    if invalid:
        choices = ", ".join(sorted(TABPFN_REGRESSION_DATASETS))
        raise ValueError(
            f"Unknown dataset(s): {invalid}. Available dataset names: {choices}"
        )


def _load_full_dataset_arrays(
    dataset_name: str,
    *,
    use_cache: bool,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    _require_valid_dataset_names([dataset_name])
    data_id = TABPFN_REGRESSION_DATASETS[dataset_name]

    cached = _load_from_cache(dataset_name, verbose=verbose) if use_cache else None
    if cached is not None:
        X = cached["X"]
        y = cached["y"]
    else:
        X, y = _fetch_openml_regression(dataset_name, data_id, verbose=verbose)
        if use_cache:
            _save_to_cache(dataset_name, X, y, verbose=verbose)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y


def _make_subsample_indices(
    num_rows: int,
    *,
    new_instances_eval: int,
    random_state: int,
) -> np.ndarray:
    if new_instances_eval <= 0:
        raise ValueError(f"new_instances_eval must be positive, got {new_instances_eval}")
    if new_instances_eval < num_rows:
        rng = np.random.default_rng(int(random_state))
        return rng.choice(num_rows, size=int(new_instances_eval), replace=False)
    return np.arange(num_rows, dtype=np.int64)


def _split_with_test_count(
    num_rows: int,
    *,
    test_size: int,
    random_state: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if test_size <= 0 or test_size >= num_rows:
        raise ValueError(
            f"{split_name} split must leave both context and target non-empty; "
            f"got num_rows={num_rows}, test_size={test_size}"
        )
    splitter = ShuffleSplit(
        n_splits=1,
        test_size=int(test_size),
        random_state=int(random_state),
    )
    first_idx, second_idx = next(splitter.split(np.empty((num_rows, 1), dtype=np.float32)))
    return np.asarray(first_idx, dtype=np.int64), np.asarray(second_idx, dtype=np.int64)


def resolve_filtered_dataset_names(
    *,
    max_features_eval: int = DEFAULT_MAX_FEATURES_EVAL,
    new_instances_eval: int = DEFAULT_NEW_INSTANCES_EVAL,
    random_state: int = 0,
    use_cache: bool = True,
    verbose: bool = True,
    dataset_names: list[str] | None = None,
) -> list[str]:
    _require_valid_dataset_names(dataset_names)
    datasets = get_regression_datasets(
        max_features_eval=max_features_eval,
        new_instances_eval=new_instances_eval,
        random_state=random_state,
        verbose=verbose,
        use_cache=use_cache,
        dataset_names=dataset_names,
    )
    resolved = list(datasets.keys())
    if dataset_names is not None:
        missing = [name for name in dataset_names if name not in resolved]
        if missing:
            raise ValueError(
                "Requested dataset(s) were filtered out or unavailable under the "
                f"current evaluation settings: {missing}"
            )
    return resolved


def _load_task_dataset_view(
    dataset_name: str,
    *,
    max_features_eval: int = DEFAULT_MAX_FEATURES_EVAL,
    new_instances_eval: int = DEFAULT_NEW_INSTANCES_EVAL,
    random_state: int = 0,
    use_cache: bool = True,
    verbose: bool = True,
) -> TaskDatasetView:
    datasets = get_regression_datasets(
        max_features_eval=max_features_eval,
        new_instances_eval=new_instances_eval,
        random_state=random_state,
        verbose=verbose,
        use_cache=use_cache,
        dataset_names=[dataset_name],
    )
    if dataset_name not in datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' is not available under "
            f"max_features_eval={max_features_eval}"
        )

    X_eval, y_eval = datasets[dataset_name]
    X_full, y_full = _load_full_dataset_arrays(
        dataset_name,
        use_cache=use_cache,
        verbose=verbose,
    )
    if X_full.shape[1] > max_features_eval:
        raise ValueError(
            f"Dataset '{dataset_name}' has {X_full.shape[1]} features, which exceeds "
            f"max_features_eval={max_features_eval}"
        )

    subsample_indices = _make_subsample_indices(
        len(y_full),
        new_instances_eval=new_instances_eval,
        random_state=random_state,
    )
    X_subsample = np.nan_to_num(X_full[subsample_indices], nan=0.0).astype(np.float32)
    y_subsample = y_full[subsample_indices].astype(np.float32)

    if X_subsample.shape != X_eval.shape or y_subsample.shape != y_eval.shape:
        raise RuntimeError(
            f"Evaluation task reconstruction mismatch for dataset '{dataset_name}'."
        )
    if not np.allclose(X_subsample, X_eval) or not np.allclose(y_subsample, y_eval):
        raise RuntimeError(
            f"Evaluation task reconstruction drifted for dataset '{dataset_name}'."
        )

    return TaskDatasetView(
        dataset_name=dataset_name,
        random_state=int(random_state),
        max_features_eval=int(max_features_eval),
        new_instances_eval=int(new_instances_eval),
        full_num_rows=int(y_full.shape[0]),
        num_features=int(X_subsample.shape[1]),
        subsample_indices=subsample_indices.astype(np.int64),
        X_subsample=X_subsample,
        y_subsample=y_subsample,
    )


def _build_inner_finetune_indices(
    outer_context_idx: np.ndarray,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if outer_context_idx.ndim != 1 or outer_context_idx.size < 2:
        raise ValueError(
            "Outer context split must contain at least two rows to build the "
            f"inner finetune split; got {outer_context_idx.size} rows."
        )
    inner_target_size = max(1, int(outer_context_idx.shape[0] // 2))
    inner_context_local, inner_target_local = _split_with_test_count(
        outer_context_idx.shape[0],
        test_size=inner_target_size,
        random_state=random_state,
        split_name="inner",
    )
    return outer_context_idx[inner_context_local], outer_context_idx[inner_target_local]


def _build_task_split_from_indices(
    dataset_view: TaskDatasetView,
    *,
    outer_context_idx: np.ndarray,
    outer_target_idx: np.ndarray,
    outer_split_random_state: int,
    inner_split_random_state: int,
    n_splits: int,
    fold_index: int,
    outer_splitter: str,
) -> TaskSplit:
    finetune_context_idx, finetune_target_idx = _build_inner_finetune_indices(
        np.asarray(outer_context_idx, dtype=np.int64),
        random_state=int(inner_split_random_state),
    )

    if np.asarray(outer_target_idx, dtype=np.int64).size <= 0:
        raise ValueError(
            f"Outer split for dataset '{dataset_view.dataset_name}' is empty."
        )

    return TaskSplit(
        dataset_name=dataset_view.dataset_name,
        random_state=int(dataset_view.random_state),
        max_features_eval=int(dataset_view.max_features_eval),
        new_instances_eval=int(dataset_view.new_instances_eval),
        full_num_rows=int(dataset_view.full_num_rows),
        num_features=int(dataset_view.num_features),
        outer_split_random_state=int(outer_split_random_state),
        inner_split_random_state=int(inner_split_random_state),
        subsample_indices=dataset_view.subsample_indices.astype(np.int64).tolist(),
        outer_context_indices=np.asarray(outer_context_idx, dtype=np.int64).tolist(),
        outer_target_indices=np.asarray(outer_target_idx, dtype=np.int64).tolist(),
        finetune_context_indices=finetune_context_idx.tolist(),
        finetune_target_indices=finetune_target_idx.tolist(),
        X_subsample=dataset_view.X_subsample,
        y_subsample=dataset_view.y_subsample,
        X_outer_context=dataset_view.X_subsample[np.asarray(outer_context_idx, dtype=np.int64)],
        y_outer_context=dataset_view.y_subsample[np.asarray(outer_context_idx, dtype=np.int64)],
        X_outer_target=dataset_view.X_subsample[np.asarray(outer_target_idx, dtype=np.int64)],
        y_outer_target=dataset_view.y_subsample[np.asarray(outer_target_idx, dtype=np.int64)],
        X_finetune_context=dataset_view.X_subsample[finetune_context_idx],
        y_finetune_context=dataset_view.y_subsample[finetune_context_idx],
        X_finetune_target=dataset_view.X_subsample[finetune_target_idx],
        y_finetune_target=dataset_view.y_subsample[finetune_target_idx],
        n_splits=int(n_splits),
        fold_index=int(fold_index),
        outer_splitter=str(outer_splitter),
    )


def build_task_splits(
    dataset_name: str,
    *,
    max_features_eval: int = DEFAULT_MAX_FEATURES_EVAL,
    new_instances_eval: int = DEFAULT_NEW_INSTANCES_EVAL,
    random_state: int = 0,
    n_splits: int = 5,
    use_cache: bool = True,
    verbose: bool = True,
) -> list[TaskSplit]:
    if int(n_splits) <= 0:
        raise ValueError(f"n_splits must be positive, got {n_splits}")

    dataset_view = _load_task_dataset_view(
        dataset_name,
        max_features_eval=max_features_eval,
        new_instances_eval=new_instances_eval,
        random_state=random_state,
        use_cache=use_cache,
        verbose=verbose,
    )

    if int(n_splits) <= 1:
        if dataset_view.X_subsample.shape[0] % 10 != 0:
            raise ValueError(
                "Legacy single-split finetuning requires a row count divisible by 10 "
                "so the outer 80/20 and inner 50/50 splits remain exact; got "
                f"{dataset_view.X_subsample.shape[0]} rows."
            )
        outer_target_size = int(round(dataset_view.X_subsample.shape[0] * OUTER_TEST_FRACTION))
        outer_context_idx, outer_target_idx = _split_with_test_count(
            dataset_view.X_subsample.shape[0],
            test_size=outer_target_size,
            random_state=random_state,
            split_name="outer",
        )
        return [
            _build_task_split_from_indices(
                dataset_view,
                outer_context_idx=outer_context_idx,
                outer_target_idx=outer_target_idx,
                outer_split_random_state=int(random_state),
                inner_split_random_state=int(random_state + 1),
                n_splits=1,
                fold_index=1,
                outer_splitter="shuffle_split",
            )
        ]

    splitter = KFold(
        n_splits=int(n_splits),
        shuffle=True,
        random_state=int(random_state),
    )
    splits: list[TaskSplit] = []
    for fold_index, (outer_context_idx, outer_target_idx) in enumerate(
        splitter.split(dataset_view.X_subsample),
        start=1,
    ):
        splits.append(
            _build_task_split_from_indices(
                dataset_view,
                outer_context_idx=np.asarray(outer_context_idx, dtype=np.int64),
                outer_target_idx=np.asarray(outer_target_idx, dtype=np.int64),
                outer_split_random_state=int(random_state),
                inner_split_random_state=int(random_state) + int(fold_index),
                n_splits=int(n_splits),
                fold_index=int(fold_index),
                outer_splitter="kfold",
            )
        )
    return splits


def build_task_split(
    dataset_name: str,
    *,
    max_features_eval: int = DEFAULT_MAX_FEATURES_EVAL,
    new_instances_eval: int = DEFAULT_NEW_INSTANCES_EVAL,
    random_state: int = 0,
    n_splits: int = 1,
    fold_index: int = 1,
    use_cache: bool = True,
    verbose: bool = True,
) -> TaskSplit:
    splits = build_task_splits(
        dataset_name,
        max_features_eval=max_features_eval,
        new_instances_eval=new_instances_eval,
        random_state=random_state,
        n_splits=n_splits,
        use_cache=use_cache,
        verbose=verbose,
    )
    target_fold = int(fold_index)
    for task_split in splits:
        if int(task_split.fold_index) == target_fold:
            return task_split
    raise ValueError(
        f"Requested fold_index={target_fold} is unavailable for dataset '{dataset_name}'."
    )


def _map_to_original_indices(
    subsample_indices: list[int],
    local_indices: list[int],
) -> list[int]:
    subsample_np = np.asarray(subsample_indices, dtype=np.int64)
    local_np = np.asarray(local_indices, dtype=np.int64)
    return subsample_np[local_np].astype(np.int64).tolist()


def task_split_to_payload(task_split: TaskSplit) -> dict[str, Any]:
    return {
        "dataset_name": task_split.dataset_name,
        "random_state": task_split.random_state,
        "n_splits": int(task_split.n_splits),
        "fold_index": int(task_split.fold_index),
        "outer_splitter": str(task_split.outer_splitter),
        "max_features_eval": task_split.max_features_eval,
        "new_instances_eval": task_split.new_instances_eval,
        "full_num_rows": task_split.full_num_rows,
        "subsample_num_rows": int(task_split.y_subsample.shape[0]),
        "num_features": task_split.num_features,
        "outer_split_random_state": task_split.outer_split_random_state,
        "inner_split_random_state": task_split.inner_split_random_state,
        "subsample_indices": task_split.subsample_indices,
        "outer_context_indices": task_split.outer_context_indices,
        "outer_target_indices": task_split.outer_target_indices,
        "finetune_context_indices": task_split.finetune_context_indices,
        "finetune_target_indices": task_split.finetune_target_indices,
        "outer_context_original_indices": _map_to_original_indices(
            task_split.subsample_indices,
            task_split.outer_context_indices,
        ),
        "outer_target_original_indices": _map_to_original_indices(
            task_split.subsample_indices,
            task_split.outer_target_indices,
        ),
        "finetune_context_original_indices": _map_to_original_indices(
            task_split.subsample_indices,
            task_split.finetune_context_indices,
        ),
        "finetune_target_original_indices": _map_to_original_indices(
            task_split.subsample_indices,
            task_split.finetune_target_indices,
        ),
        "outer_context_size": int(task_split.y_outer_context.shape[0]),
        "outer_target_size": int(task_split.y_outer_target.shape[0]),
        "finetune_context_size": int(task_split.y_finetune_context.shape[0]),
        "finetune_target_size": int(task_split.y_finetune_target.shape[0]),
    }


def task_split_from_payload(
    payload: dict[str, Any],
    *,
    use_cache: bool = True,
    verbose: bool = True,
) -> TaskSplit:
    dataset_name = str(payload["dataset_name"])
    X_full, y_full = _load_full_dataset_arrays(
        dataset_name,
        use_cache=use_cache,
        verbose=verbose,
    )

    subsample_indices = np.asarray(payload["subsample_indices"], dtype=np.int64)
    outer_context_indices = np.asarray(payload["outer_context_indices"], dtype=np.int64)
    outer_target_indices = np.asarray(payload["outer_target_indices"], dtype=np.int64)
    finetune_context_indices = np.asarray(
        payload["finetune_context_indices"],
        dtype=np.int64,
    )
    finetune_target_indices = np.asarray(
        payload["finetune_target_indices"],
        dtype=np.int64,
    )

    if subsample_indices.ndim != 1 or subsample_indices.size == 0:
        raise ValueError(
            f"Invalid subsample_indices for dataset '{dataset_name}' in split payload."
        )
    if np.any(subsample_indices < 0) or np.any(subsample_indices >= len(y_full)):
        raise ValueError(
            f"split.json for dataset '{dataset_name}' contains out-of-range subsample indices."
        )

    X_subsample = np.nan_to_num(X_full[subsample_indices], nan=0.0).astype(np.float32)
    y_subsample = y_full[subsample_indices].astype(np.float32)

    for index_name, index_array in (
        ("outer_context_indices", outer_context_indices),
        ("outer_target_indices", outer_target_indices),
        ("finetune_context_indices", finetune_context_indices),
        ("finetune_target_indices", finetune_target_indices),
    ):
        if index_array.ndim != 1:
            raise ValueError(
                f"{index_name} for dataset '{dataset_name}' must be a flat index array."
            )
        if np.any(index_array < 0) or np.any(index_array >= X_subsample.shape[0]):
            raise ValueError(
                f"{index_name} for dataset '{dataset_name}' contains out-of-range values."
            )

    expected_rows = payload.get("subsample_num_rows")
    if expected_rows is not None and int(expected_rows) != int(X_subsample.shape[0]):
        raise ValueError(
            f"subsample_num_rows mismatch for dataset '{dataset_name}': "
            f"expected {expected_rows}, got {X_subsample.shape[0]}"
        )

    expected_features = payload.get("num_features")
    if expected_features is not None and int(expected_features) != int(X_subsample.shape[1]):
        raise ValueError(
            f"num_features mismatch for dataset '{dataset_name}': "
            f"expected {expected_features}, got {X_subsample.shape[1]}"
        )

    expected_full_rows = payload.get("full_num_rows")
    if expected_full_rows is not None and int(expected_full_rows) != int(y_full.shape[0]):
        raise ValueError(
            f"full_num_rows mismatch for dataset '{dataset_name}': "
            f"expected {expected_full_rows}, got {y_full.shape[0]}"
        )

    return TaskSplit(
        dataset_name=dataset_name,
        random_state=int(payload["random_state"]),
        max_features_eval=int(payload.get("max_features_eval", X_subsample.shape[1])),
        new_instances_eval=int(payload.get("new_instances_eval", X_subsample.shape[0])),
        full_num_rows=int(y_full.shape[0]),
        num_features=int(X_subsample.shape[1]),
        outer_split_random_state=int(
            payload.get("outer_split_random_state", payload["random_state"])
        ),
        inner_split_random_state=int(
            payload.get("inner_split_random_state", int(payload["random_state"]) + 1)
        ),
        subsample_indices=subsample_indices.tolist(),
        outer_context_indices=outer_context_indices.tolist(),
        outer_target_indices=outer_target_indices.tolist(),
        finetune_context_indices=finetune_context_indices.tolist(),
        finetune_target_indices=finetune_target_indices.tolist(),
        X_subsample=X_subsample,
        y_subsample=y_subsample,
        X_outer_context=X_subsample[outer_context_indices],
        y_outer_context=y_subsample[outer_context_indices],
        X_outer_target=X_subsample[outer_target_indices],
        y_outer_target=y_subsample[outer_target_indices],
        X_finetune_context=X_subsample[finetune_context_indices],
        y_finetune_context=y_subsample[finetune_context_indices],
        X_finetune_target=X_subsample[finetune_target_indices],
        y_finetune_target=y_subsample[finetune_target_indices],
        n_splits=int(payload.get("n_splits", 1)),
        fold_index=int(payload.get("fold_index", 1)),
        outer_splitter=str(payload.get("outer_splitter", "shuffle_split")),
    )


def load_task_split_json(
    split_path: str | Path,
    *,
    use_cache: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, Any], TaskSplit]:
    split_file = Path(split_path)
    with split_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    task_split = task_split_from_payload(
        payload,
        use_cache=use_cache,
        verbose=verbose,
    )
    return payload, task_split


def format_task_label(task_split: TaskSplit) -> str:
    if int(task_split.n_splits) > 1:
        return f"{task_split.dataset_name}-fold{int(task_split.fold_index)}"
    return str(task_split.dataset_name)


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for key in ("R2", "RMSE", "MAE"):
        values = [
            float(metric_dict[key])
            for metric_dict in metric_dicts
            if key in metric_dict
        ]
        if not values:
            continue
        aggregated[key] = float(np.mean(values))
        aggregated[f"{key}_SE"] = _standard_error(values)
        aggregated[f"{key}_FOLD_VALUES"] = [float(value) for value in values]
    return aggregated


def summarize_dataset_finetune_results(
    dataset_name: str,
    fold_results: list[dict[str, Any]],
    *,
    output_root: str | Path,
    n_splits: int | None = None,
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ordered_results = sorted(
        fold_results,
        key=lambda item: (
            int(item.get("fold_index", 1)),
            str(item.get("output_dir", "")),
        ),
    )
    inferred_n_splits = [
        int(item.get("n_splits", 1))
        for item in ordered_results
    ]
    inferred_n_splits.extend(
        int(item.get("n_splits", 1))
        for item in (failures or [])
        if "n_splits" in item
    )
    summary = {
        "dataset_name": str(dataset_name),
        "output_root": str(Path(output_root).resolve()),
        "n_splits": int(
            n_splits
            if n_splits is not None
            else max(inferred_n_splits, default=1)
        ),
        "completed": ordered_results,
        "failures": list(failures or []),
        "completed_fold_indices": [
            int(item.get("fold_index", 1))
            for item in ordered_results
        ],
    }
    if ordered_results:
        summary["base_metrics"] = aggregate_metric_dicts(
            [item["base_metrics"] for item in ordered_results if "base_metrics" in item]
        )
        summary["finetuned_metrics"] = aggregate_metric_dicts(
            [item["finetuned_metrics"] for item in ordered_results if "finetuned_metrics" in item]
        )
        summary["metric_deltas"] = aggregate_metric_dicts(
            [item["metric_deltas"] for item in ordered_results if "metric_deltas" in item]
        )
    else:
        summary["base_metrics"] = {}
        summary["finetuned_metrics"] = {}
        summary["metric_deltas"] = {}
    return summary


def _reconstruct_config(config_dict: dict[str, Any]) -> Config:
    default_config = Config()
    kwargs: dict[str, Any] = {}
    for key, value in config_dict.items():
        if not hasattr(default_config, key):
            continue
        current_value = getattr(default_config, key)
        if isinstance(value, dict):
            kwargs[key] = type(current_value)(**value)
        else:
            kwargs[key] = value
    return Config(**kwargs)


def build_finetune_config(
    base_config: Config,
    args: Any,
    *,
    dataset_name: str,
    task_label: str,
    output_dir: Path,
) -> Config:
    config = copy.deepcopy(base_config)
    tc = config.training
    oc = config.optimizer

    tc.device = str(args.device)
    tc.batch_size = 1
    tc.micro_batch_size = 1
    tc.num_epochs = 1
    tc.samples_per_epoch = int(args.finetune_steps)
    tc.loss_type = str(args.loss_type or tc.loss_type)
    tc.std_min = float(tc.std_min if args.std_min is None else args.std_min)
    tc.z_max = float(tc.z_max if args.z_max is None else args.z_max)
    tc.hard_z_threshold = float(
        tc.hard_z_threshold
        if args.hard_z_threshold is None
        else args.hard_z_threshold
    )
    tc.max_hard_z_frac = float(
        tc.max_hard_z_frac
        if args.max_hard_z_frac is None
        else args.max_hard_z_frac
    )
    tc.gradient_clipping = float(
        tc.gradient_clipping
        if args.gradient_clipping is None
        else args.gradient_clipping
    )
    tc.checkpoint_dir = str(output_dir)
    tc.checkpoint_path = None
    tc.save_every = int(args.save_every)
    tc.wandb_log = bool(args.wandb_log)
    tc.wandb_project = str(args.wandb_project)
    tc.wandb_name = (
        f"{args.wandb_name}-{task_label}"
        if args.wandb_name
        else f"finetune-{task_label}"
    )
    tc.wandb_id = (
        f"{args.wandb_id}-{task_label}"
        if args.wandb_id
        else None
    )
    tc.wandb_dir = str(args.wandb_dir) if args.wandb_dir else None
    tc.wandb_mode = str(args.wandb_mode)
    tc.eval_every = 0
    tc.num_workers = 0
    tc.micro_progress = False
    tc.prior_dir = None
    tc.tabicl_repo = None
    tc.np_seed = int(args.seed)
    tc.torch_seed = int(args.seed)

    oc.init_lr = float(oc.init_lr if args.init_lr is None else args.init_lr)
    oc.peak_lr = float(oc.peak_lr if args.peak_lr is None else args.peak_lr)
    oc.end_lr = float(oc.end_lr if args.end_lr is None else args.end_lr)
    oc.ema_rate = float(oc.ema_rate if args.ema_rate is None else args.ema_rate)
    oc.weight_decay = float(
        oc.weight_decay if args.weight_decay is None else args.weight_decay
    )

    return config


def load_finetune_state(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    args: Any,
    dataset_name: str,
    task_label: str,
    output_dir: Path,
) -> tuple[dict[str, Any], Config, torch.nn.Module, Any, EMA]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "config" not in checkpoint:
        raise ValueError(f"Checkpoint '{checkpoint_path}' does not contain a config.")

    base_config = _reconstruct_config(checkpoint["config"])
    config = build_finetune_config(
        base_config,
        args,
        dataset_name=dataset_name,
        task_label=task_label,
        output_dir=output_dir,
    )
    model, process = build_model_and_process(config, device)

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "ema_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    ema = EMA(model, decay=config.optimizer.ema_rate)
    if "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    else:
        ema.load_state_dict(model.state_dict())

    model.to(device)
    model.train()
    ema.shadow.to(device)
    ema.shadow.eval()
    return checkpoint, config, model, process, ema


def prepare_fixed_finetune_batch(
    task_split: TaskSplit,
    *,
    device: torch.device,
    config: Config,
) -> tuple[Batch, dict[str, Any]]:
    x_context = torch.from_numpy(task_split.X_finetune_context).to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    x_target = torch.from_numpy(task_split.X_finetune_target).to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    y_context = torch.from_numpy(task_split.y_finetune_context).to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    y_target = torch.from_numpy(task_split.y_finetune_target).to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    y_context_norm, y_target_norm, mean, std = normalize_y(y_context, y_target)
    raw_std = y_context.unsqueeze(-1).std(dim=1, keepdim=True, unbiased=False)

    tc = config.training
    tiny_std_task = raw_std.squeeze(-1).squeeze(-1) < float(tc.std_min)
    hard_outlier_frac = (
        (y_target_norm.abs() > float(tc.hard_z_threshold))
        .to(torch.float32)
        .mean(dim=(1, 2))
    )
    extreme_task = hard_outlier_frac > float(tc.max_hard_z_frac)
    finite_task = (
        torch.isfinite(y_context_norm).all(dim=(1, 2))
        & torch.isfinite(y_target_norm).all(dim=(1, 2))
    )
    clipped_target_values = int(
        (y_target_norm.abs() > float(tc.z_max)).sum().item()
    )
    valid_task = bool((~tiny_std_task & ~extreme_task & finite_task).item())
    if not valid_task:
        raise ValueError(
            "Fixed finetune task is invalid under the configured target filtering: "
            f"tiny_std={bool(tiny_std_task.item())}, "
            f"extreme={bool(extreme_task.item())}, "
            f"finite={bool(finite_task.item())}"
        )

    y_context_norm = y_context_norm.clamp(
        min=-float(tc.z_max),
        max=float(tc.z_max),
    )
    y_target_norm = y_target_norm.clamp(
        min=-float(tc.z_max),
        max=float(tc.z_max),
    )

    mask_context = torch.zeros(x_context.shape[:2], device=device, dtype=torch.float32)
    mask_target = torch.zeros(x_target.shape[:2], device=device, dtype=torch.float32)
    batch = Batch(
        x_target=x_target,
        y_target=y_target_norm,
        x_context=x_context,
        y_context=y_context_norm,
        mask_target=mask_target,
        mask_context=mask_context,
    )
    stats = {
        "seq_len": int(x_context.shape[1] + x_target.shape[1]),
        "train_size": int(x_context.shape[1]),
        "target_size": int(x_target.shape[1]),
        "context_mean": float(mean.squeeze().item()),
        "context_std": float(std.squeeze().item()),
        "raw_context_std": float(raw_std.squeeze().item()),
        "hard_outlier_frac": float(hard_outlier_frac.squeeze().item()),
        "clipped_target_values": clipped_target_values,
    }
    return batch, stats


def save_finetune_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    config: Config,
    step: int,
    task_payload: dict[str, Any],
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "curr_step": int(step),
        "config": asdict(config),
        "finetune_task": task_payload,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def evaluate_fixed_split(
    *,
    model: torch.nn.Module,
    process: Any,
    device: torch.device,
    task_split: TaskSplit,
    num_sampling_steps: int,
    sampling_method: str,
    random_state: int,
    ddim_eta: float = 0.0,
    n_repeats: int = 20,
) -> dict[str, Any]:
    eval_model = model.eval()
    wrapper = NDPRegressorWrapper(
        eval_model,
        process,
        device,
        num_sampling_steps=int(num_sampling_steps),
        sampling_method=str(sampling_method),
        ddim_eta=float(ddim_eta),
    )
    wrapper.fit(task_split.X_outer_context, task_split.y_outer_context)

    repeat_predictions: list[np.ndarray] = []
    for rep_idx in range(max(1, int(n_repeats))):
        seed = int(random_state) + rep_idx
        set_seed(seed, seed)
        y_pred = wrapper.predict(task_split.X_outer_target)
        repeat_predictions.append(np.asarray(y_pred, dtype=np.float32))

    y_pred_repetitions = np.stack(repeat_predictions, axis=0)
    y_pred_mean = np.mean(y_pred_repetitions, axis=0).astype(np.float32)
    y_true = np.asarray(task_split.y_outer_target, dtype=np.float32)
    metrics = _compute_regression_metrics(y_true, y_pred_mean)

    return {
        "metrics": metrics,
        "y_true": y_true.tolist(),
        "y_pred_repetitions": y_pred_repetitions.tolist(),
        "y_pred_mean": y_pred_mean.tolist(),
        "n_repeats": int(max(1, int(n_repeats))),
        "num_sampling_steps": int(num_sampling_steps),
        "sampling_method": str(sampling_method),
        "ddim_eta": float(ddim_eta),
    }


def _metric_deltas(
    base_metrics: dict[str, float],
    finetuned_metrics: dict[str, float],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in ("R2", "RMSE", "MAE"):
        if key in base_metrics and key in finetuned_metrics:
            deltas[f"{key}_delta"] = float(finetuned_metrics[key] - base_metrics[key])
    return deltas


def maybe_init_wandb_run(
    *,
    config: Config,
    task_payload: dict[str, Any],
) -> Any | None:
    tc = config.training
    if not tc.wandb_log or tc.wandb_mode == "disabled":
        return None
    if wandb is None:
        print("wandb is not available; skipping finetune logging.", flush=True)
        return None

    try:
        return wandb.init(
            dir=tc.wandb_dir,
            project=tc.wandb_project,
            name=tc.wandb_name,
            id=tc.wandb_id,
            config={
                "finetune_config": asdict(config),
                "task_split": task_payload,
            },
            resume="allow",
            mode=tc.wandb_mode,
        )
    except Exception as exc:
        print(f"wandb.init failed: {exc}", flush=True)
        return None


def add_shared_cli_args(parser, *, include_dataset: bool) -> None:
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the pretrained TabICL_regression03-compatible checkpoint.",
    )
    if include_dataset:
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="One evaluation dataset name to finetune.",
        )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--finetune-steps", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-features-eval", type=int, default=DEFAULT_MAX_FEATURES_EVAL)
    parser.add_argument("--new-instances-eval", type=int, default=DEFAULT_NEW_INSTANCES_EVAL)
    parser.add_argument("--num-sampling-steps", type=int, default=500)
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="DDIM eta; ignored unless --sampling-method ddim.",
    )
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--loss-type", type=str, default=None, choices=["l1", "l2"])
    parser.add_argument("--std-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--hard-z-threshold", type=float, default=None)
    parser.add_argument("--max-hard-z-frac", type=float, default=None)
    parser.add_argument("--gradient-clipping", type=float, default=None)
    parser.add_argument("--init-lr", type=float, default=None)
    parser.add_argument("--peak-lr", type=float, default=None)
    parser.add_argument("--end-lr", type=float, default=None)
    parser.add_argument("--ema-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb-log", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb_log", action="store_false")
    parser.add_argument("--wandb-project", type=str, default="TabICL-regression03")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="offline")
    parser.add_argument("--wandb-dir", type=str, default=None)


def run_task_split_finetune(
    args: Any,
    *,
    task_split: TaskSplit,
    output_dir: str | Path,
) -> dict[str, Any]:
    target_output_dir = Path(output_dir).resolve()
    target_output_dir.mkdir(parents=True, exist_ok=True)

    task_payload = task_split_to_payload(task_split)
    split_path = target_output_dir / "split.json"
    write_json(split_path, task_payload)

    task_label = format_task_label(task_split)
    device = torch.device(str(args.device))
    checkpoint, config, model, process, ema = load_finetune_state(
        args.checkpoint,
        device=device,
        args=args,
        dataset_name=task_split.dataset_name,
        task_label=task_label,
        output_dir=target_output_dir,
    )

    base_eval = evaluate_fixed_split(
        model=ema.shadow,
        process=process,
        device=device,
        task_split=task_split,
        num_sampling_steps=int(args.num_sampling_steps),
        sampling_method=str(args.sampling_method),
        ddim_eta=float(args.ddim_eta),
        random_state=int(task_split.random_state),
        n_repeats=int(args.n_repeats),
    )

    fixed_batch, batch_stats = prepare_fixed_finetune_batch(
        task_split,
        device=device,
        config=config,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.init_lr,
        weight_decay=config.optimizer.weight_decay,
    )

    set_seed(config.training.np_seed, config.training.torch_seed)
    amp_enabled = bool(config.training.amp and device.type == "cuda")
    if amp_enabled:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        scaler = None
        amp_ctx = nullcontext()

    diffusion_key = torch.Generator(device=device)
    diffusion_key.manual_seed(config.training.torch_seed)

    wandb_run = maybe_init_wandb_run(config=config, task_payload=task_payload)
    step_logs: list[dict[str, Any]] = []
    periodic_checkpoints: list[str] = []

    step_iterable = range(1, int(args.finetune_steps) + 1)
    if bool(args.verbose) and tqdm is not None:
        step_iterable = tqdm(
            step_iterable,
            total=int(args.finetune_steps),
            desc=f"finetune:{task_label}",
            leave=False,
        )

    for step in step_iterable:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        lr = compute_lr(config, int(step), total_steps=int(args.finetune_steps))
        for group in optimizer.param_groups:
            group["lr"] = lr

        with amp_ctx:
            loss = diffusion_loss(
                process,
                model,
                fixed_batch,
                diffusion_key,
                num_timesteps=config.diffusion.timesteps,
                loss_type=config.training.loss_type,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite finetune loss at step {step}")

        if amp_enabled:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        if config.training.gradient_clipping > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.gradient_clipping,
            )
        else:
            grad_norm = compute_grad_norm(list(model.parameters()))

        if amp_enabled:
            assert scaler is not None
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        ema.update(model)

        step_payload = {
            "step": int(step),
            "loss": float(loss.item()),
            "lr": float(lr),
            "grad_norm": float(grad_norm),
        }
        step_logs.append(step_payload)

        if tqdm is not None and hasattr(step_iterable, "set_postfix"):
            step_iterable.set_postfix(
                loss=f"{step_payload['loss']:.4f}",
                lr=f"{step_payload['lr']:.6f}",
            )

        if wandb_run is not None:
            try:
                wandb_run.log(
                    {
                        "step": int(step),
                        "train/loss": float(loss.item()),
                        "train/lr": float(lr),
                        "train/grad_norm": float(grad_norm),
                    },
                    step=int(step),
                )
            except Exception as exc:
                print(f"wandb.log failed: {exc}", flush=True)
                wandb_run = None

        if int(args.save_every) > 0 and int(step) % int(args.save_every) == 0:
            step_ckpt = save_finetune_checkpoint(
                target_output_dir / f"step-{step}.pt",
                model=model,
                ema=ema,
                optimizer=optimizer,
                config=config,
                step=int(step),
                task_payload=task_payload,
            )
            periodic_checkpoints.append(str(step_ckpt))

    last_checkpoint = save_finetune_checkpoint(
        target_output_dir / "last.pt",
        model=model,
        ema=ema,
        optimizer=optimizer,
        config=config,
        step=int(args.finetune_steps),
        task_payload=task_payload,
    )

    finetuned_eval = evaluate_fixed_split(
        model=ema.shadow,
        process=process,
        device=device,
        task_split=task_split,
        num_sampling_steps=int(args.num_sampling_steps),
        sampling_method=str(args.sampling_method),
        ddim_eta=float(args.ddim_eta),
        random_state=int(task_split.random_state),
        n_repeats=int(args.n_repeats),
    )

    if wandb_run is not None:
        summary_payload = {
            "eval/base_R2": float(base_eval["metrics"]["R2"]),
            "eval/base_RMSE": float(base_eval["metrics"]["RMSE"]),
            "eval/base_MAE": float(base_eval["metrics"]["MAE"]),
            "eval/finetuned_R2": float(finetuned_eval["metrics"]["R2"]),
            "eval/finetuned_RMSE": float(finetuned_eval["metrics"]["RMSE"]),
            "eval/finetuned_MAE": float(finetuned_eval["metrics"]["MAE"]),
        }
        summary_payload.update(
            {
                f"eval/{k}": v
                for k, v in _metric_deltas(
                    base_eval["metrics"],
                    finetuned_eval["metrics"],
                ).items()
            }
        )
        wandb_run.summary.update(summary_payload)
        wandb_run.finish()

    metric_deltas = _metric_deltas(
        base_eval["metrics"],
        finetuned_eval["metrics"],
    )
    metrics_payload = {
        "dataset_name": task_split.dataset_name,
        "task_label": task_label,
        "fold_index": int(task_split.fold_index),
        "n_splits": int(task_split.n_splits),
        "outer_splitter": str(task_split.outer_splitter),
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "last_checkpoint": str(last_checkpoint),
        "split_json": str(split_path),
        "task_split": task_payload,
        "batch_stats": batch_stats,
        "finetune_steps": int(args.finetune_steps),
        "training_history": step_logs,
        "periodic_checkpoints": periodic_checkpoints,
        "config": asdict(config),
        "base_evaluation": base_eval,
        "finetuned_evaluation": finetuned_eval,
        "metric_deltas": metric_deltas,
    }
    metrics_path = target_output_dir / "metrics.json"
    write_json(metrics_path, metrics_payload)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "dataset_name": task_split.dataset_name,
        "task_label": task_label,
        "fold_index": int(task_split.fold_index),
        "n_splits": int(task_split.n_splits),
        "outer_splitter": str(task_split.outer_splitter),
        "output_dir": str(target_output_dir),
        "split_json": str(split_path),
        "metrics_json": str(metrics_path),
        "last_checkpoint": str(last_checkpoint),
        "base_metrics": base_eval["metrics"],
        "finetuned_metrics": finetuned_eval["metrics"],
        "metric_deltas": metric_deltas,
        "finetune_steps": int(args.finetune_steps),
        "num_sampling_steps": int(args.num_sampling_steps),
        "sampling_method": str(args.sampling_method),
        "ddim_eta": float(args.ddim_eta),
        "base_checkpoint_step": int(checkpoint.get("curr_step", 0)),
    }


def run_dataset_finetune_suite(
    args: Any,
    *,
    dataset_name: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    target_dataset = str(dataset_name or args.dataset)
    target_output_dir = Path(output_dir or args.output_dir).resolve()
    target_output_dir.mkdir(parents=True, exist_ok=True)

    if int(args.finetune_steps) <= 0:
        raise ValueError(f"finetune_steps must be positive, got {args.finetune_steps}")

    use_cache = not bool(args.no_cache)
    verbose = bool(args.verbose)
    task_splits = build_task_splits(
        target_dataset,
        max_features_eval=int(args.max_features_eval),
        new_instances_eval=int(args.new_instances_eval),
        random_state=int(args.random_state),
        n_splits=int(args.n_splits),
        use_cache=use_cache,
        verbose=verbose,
    )

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    multi_fold = len(task_splits) > 1

    for task_split in task_splits:
        fold_output_dir = (
            target_output_dir / f"fold_{int(task_split.fold_index)}"
            if multi_fold
            else target_output_dir
        )
        if multi_fold:
            print(
                f"[fold {int(task_split.fold_index)}/{int(task_split.n_splits)}] "
                f"Finetuning {format_task_label(task_split)} -> {fold_output_dir}",
                flush=True,
            )
        try:
            completed.append(
                run_task_split_finetune(
                    args,
                    task_split=task_split,
                    output_dir=fold_output_dir,
                )
            )
        except Exception as exc:
            failure = {
                "dataset_name": target_dataset,
                "fold_index": int(task_split.fold_index),
                "n_splits": int(task_split.n_splits),
                "outer_splitter": str(task_split.outer_splitter),
                "output_dir": str(fold_output_dir),
                "error": str(exc),
            }
            failures.append(failure)
            if not multi_fold:
                raise

    summary = summarize_dataset_finetune_results(
        target_dataset,
        completed,
        output_root=target_output_dir,
        n_splits=int(args.n_splits),
        failures=failures,
    )
    if multi_fold or failures:
        summary_path = target_output_dir / "summary.json"
        payload = dict(summary)
        payload["summary_json"] = str(summary_path)
        write_json(summary_path, payload)
        summary = payload
    return summary


def run_single_task_finetune(
    args: Any,
    *,
    dataset_name: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    summary = run_dataset_finetune_suite(
        args,
        dataset_name=dataset_name,
        output_dir=output_dir,
    )
    if int(summary.get("n_splits", 1)) <= 1 and not summary.get("failures"):
        completed = summary.get("completed", [])
        if len(completed) == 1:
            return completed[0]
    return summary
