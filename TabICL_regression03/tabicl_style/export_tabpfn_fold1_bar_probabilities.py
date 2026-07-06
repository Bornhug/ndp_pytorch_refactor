"""Export fold-1 TabPFN bar probabilities for selected regression03 datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.model_selection import KFold, ShuffleSplit

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT_ROOT = ROOT.parent
NANOTABPFN_ROOT = PROJECT_ROOT / "nanoTabPFN_version10_original02"
NANOTABPFN_TABARENA = NANOTABPFN_ROOT / "tabarena_eval"
for path in (HERE, ROOT, NANOTABPFN_ROOT, NANOTABPFN_TABARENA):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tabicl_style.evaluation import (  # noqa: E402
    DEFAULT_MAX_FEATURES_EVAL,
    get_regression_datasets,
)
from tabicl_style.evaluation_tabpfn_uncertainty import (  # noqa: E402
    compute_qice_from_boundaries,
    make_tabpfn_model_factory,
    quantile_levels_for_bins,
    tabpfn_quantile_boundaries,
)
from model import NanoTabPFNRegressor  # noqa: E402
from bar_distribution import build_distribution  # noqa: E402
from evaluation import load_checkpoint  # noqa: E402


DEFAULT_DATASETS = "Moneyball,airfoil_self_noise,sensory,space_ga"
DEFAULT_NANOTABPFN_CHECKPOINT = (
    "nanoTabPFN_version10_original02/6M/model_step_0120000.pt"
)
PER_TARGET_FIGURE1_DIR = (
    PROJECT_ROOT
    / "TabICL_regression03"
    / "runs"
    / "00tables"
    / "per-target-results"
    / "figure1"
    / "figure1_accuracy_calibration"
)
BAR_EDGE_SEMANTICS = (
    "bar_probabilities[i] is the probability mass assigned to interval "
    "[bar_edges_raw[i], bar_edges_raw[i + 1]]. First and last intervals are "
    "TabPFN full-support tail buckets with finite implementation borders, "
    "not literal infinities."
)
BAR_PROBABILITY_COUNT = 5000
BAR_EDGE_COUNT = BAR_PROBABILITY_COUNT + 1
PredictionResult = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def parse_dataset_names(raw: str) -> list[str]:
    names = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one dataset must be provided")
    return names


def as_numpy_1d(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {arr.shape}")
    return arr


def logits_to_probabilities(logits: Any) -> np.ndarray:
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    probs = torch.softmax(logits.detach().to("cpu").to(torch.float32), dim=-1)
    arr = probs.numpy().astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"logits probabilities must be 2-D, got {arr.shape}")
    return arr


def validate_bar_edges_raw(value: Any, *, name: str) -> np.ndarray:
    edges = as_numpy_1d(value, name=name).astype(np.float32)
    if edges.shape[0] != BAR_EDGE_COUNT:
        raise ValueError(f"{name} expected {BAR_EDGE_COUNT} bar edges, got {edges.shape[0]}")
    if not np.all(np.isfinite(edges)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all(edges[1:] > edges[:-1]):
        raise ValueError(f"{name} must be strictly increasing")
    return edges


def ensure_output_path(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def selected_fold_indices(
    X: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
    fold_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if int(n_splits) <= 1:
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=0.5,
            random_state=int(random_state),
        )
    else:
        splitter = KFold(
            n_splits=int(n_splits),
            shuffle=True,
            random_state=int(random_state),
        )
    splits = list(splitter.split(X))
    index = int(fold_index) - 1
    if index < 0 or index >= len(splits):
        raise ValueError(f"fold_index {fold_index} is out of range for {len(splits)} split(s)")
    return splits[index]


def compute_target_entries(
    *,
    y_test: np.ndarray,
    y_pred_mean: np.ndarray,
    quantile_levels: np.ndarray,
    quantile_boundaries: np.ndarray,
    bar_probabilities: np.ndarray,
    fold_uncertainty: dict[str, Any],
    test_idx: np.ndarray,
) -> list[dict[str, Any]]:
    if bar_probabilities.shape[0] != len(y_test):
        raise ValueError(
            "bar probability row count must match targets, "
            f"got {bar_probabilities.shape[0]} and {len(y_test)}"
        )
    if bar_probabilities.shape[1] != BAR_PROBABILITY_COUNT:
        raise ValueError(
            f"expected {BAR_PROBABILITY_COUNT} bar probabilities, "
            f"got {bar_probabilities.shape[1]}"
        )
    if quantile_boundaries.shape != (len(quantile_levels), len(y_test)):
        raise ValueError(
            "quantile boundary shape mismatch: "
            f"{quantile_boundaries.shape} vs {(len(quantile_levels), len(y_test))}"
        )
    entries: list[dict[str, Any]] = []
    bin_indices = list(fold_uncertainty["bin_indices"])
    for target_pos, test_index in enumerate(np.asarray(test_idx, dtype=np.int64).tolist()):
        probs = np.asarray(bar_probabilities[target_pos], dtype=np.float32)
        prob_sum = float(np.sum(probs, dtype=np.float64))
        if not math.isfinite(prob_sum) or abs(prob_sum - 1.0) > 1e-5:
            raise ValueError(
                f"bar probabilities for target {target_pos} sum to {prob_sum}, expected 1"
            )
        entries.append(
            {
                "target_index_within_fold": int(target_pos),
                "test_index": int(test_index),
                "y_true": float(y_test[target_pos]),
                "y_pred_mean": float(y_pred_mean[target_pos]),
                "quantile_levels": quantile_levels.astype(float).tolist(),
                "quantile_boundaries": quantile_boundaries[:, target_pos]
                .astype(np.float32)
                .tolist(),
                "qice_bin_index": int(bin_indices[target_pos]),
                "bar_probabilities": probs.tolist(),
            }
        )
    return entries


def predict_nano(
    *,
    checkpoint_path: Path,
    device: torch.device,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    quantile_levels: np.ndarray,
) -> PredictionResult:
    model = load_checkpoint(checkpoint_path, device)
    regressor = NanoTabPFNRegressor(
        model,
        device,
        bar_distribution=build_distribution(expected_num_bars=model.num_bars),
    )
    regressor.fit(X_train, y_train)
    pred_info = regressor.predict_distribution(
        X_test,
        quantile_levels=quantile_levels,
        y_true=y_test,
    )
    y_pred_mean = as_numpy_1d(pred_info["y_pred_mean"], name="nanoTabPFN mean")
    quantile_boundaries = np.asarray(pred_info["quantile_boundaries"], dtype=np.float32)
    bar_probabilities = logits_to_probabilities(pred_info["logits"])
    normalized_edges = regressor.bar_distribution.borders.detach().to("cpu").numpy()
    bar_edges_raw = validate_bar_edges_raw(
        normalized_edges * float(regressor.y_std) + float(regressor.y_mean),
        name="nanoTabPFN raw bar edges",
    )
    return (
        y_pred_mean.astype(np.float32),
        quantile_boundaries,
        bar_probabilities,
        bar_edges_raw,
    )


def predict_official(
    *,
    factory: Callable[[], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray | None = None,
    quantile_levels: np.ndarray,
) -> PredictionResult:
    del y_test
    regressor = factory()
    regressor.fit(X_train, y_train)
    full_output = regressor.predict(X_test, output_type="full")
    if not isinstance(full_output, dict):
        raise ValueError("official TabPFN full output must be a dict")
    y_pred_mean = as_numpy_1d(full_output["mean"], name="official TabPFN mean")
    quantile_boundaries = tabpfn_quantile_boundaries(
        full_output,
        quantile_levels=quantile_levels,
    ).astype(np.float32)
    bar_probabilities = logits_to_probabilities(full_output["logits"])
    criterion = full_output.get("criterion")
    if criterion is None or not hasattr(criterion, "borders"):
        raise ValueError("official TabPFN full output criterion is missing borders")
    bar_edges_raw = validate_bar_edges_raw(
        getattr(criterion, "borders"),
        name="official TabPFN raw bar edges",
    )
    return (
        y_pred_mean.astype(np.float32),
        quantile_boundaries,
        bar_probabilities,
        bar_edges_raw,
    )


def existing_targets_for_method(method_name: str) -> dict[tuple[str, int], dict[str, Any]]:
    path = PER_TARGET_FIGURE1_DIR / f"{method_name}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset_result in payload.get("dataset_results", []):
        dataset = str(dataset_result["dataset"])
        for target in dataset_result.get("targets", []):
            # Existing figure1 per-target files are mixed: nanoTabPFN uses
            # zero-based fold labels, while other files use user-facing labels.
            # Test indices are unique across KFold folds, so they are the stable
            # alignment key for validating this fold-1 export.
            lookup[(dataset, int(target["test_index"]))] = target
    return lookup


def validate_against_existing(
    *,
    method_name: str,
    dataset_name: str,
    targets: list[dict[str, Any]],
    tolerance: float,
) -> None:
    existing = existing_targets_for_method(method_name)
    if not existing:
        print(f"Warning: no existing per-target file found for {method_name}; skipping match validation.")
        return
    for target in targets:
        key = (dataset_name, int(target["test_index"]))
        if key not in existing:
            raise ValueError(f"{method_name}/{dataset_name} missing existing target {key}")
        old = existing[key]
        if abs(float(old["y_true"]) - float(target["y_true"])) > tolerance:
            raise ValueError(f"{method_name}/{dataset_name}/{key} y_true mismatch")
        old_boundaries = np.asarray(old["quantile_boundaries"], dtype=np.float64)
        new_boundaries = np.asarray(target["quantile_boundaries"], dtype=np.float64)
        if old_boundaries.shape != new_boundaries.shape:
            raise ValueError(f"{method_name}/{dataset_name}/{key} boundary shape mismatch")
        max_diff = float(np.max(np.abs(old_boundaries - new_boundaries)))
        if max_diff > tolerance:
            raise ValueError(
                f"{method_name}/{dataset_name}/{key} boundary mismatch max_diff={max_diff}"
            )


def export_method(
    *,
    method_name: str,
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    fold_index: int,
    n_splits: int,
    random_state: int,
    quantile_levels: np.ndarray,
    predictor: Callable[..., PredictionResult],
    validation_tolerance: float,
) -> dict[str, Any]:
    dataset_results: list[dict[str, Any]] = []
    iterator = list(datasets.items())
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"Exporting {method_name}", unit="dataset")
    for dataset_name, (X, y) in iterator:
        train_idx, test_idx = selected_fold_indices(
            X,
            n_splits=n_splits,
            random_state=random_state,
            fold_index=fold_index,
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = np.asarray(y[train_idx], dtype=np.float32)
        y_test = np.asarray(y[test_idx], dtype=np.float32)
        y_pred_mean, quantile_boundaries, bar_probabilities, bar_edges_raw = predictor(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            quantile_levels=quantile_levels,
        )
        fold_uncertainty = compute_qice_from_boundaries(
            y_test,
            quantile_boundaries,
            num_bins=len(quantile_levels) + 1,
        )
        targets = compute_target_entries(
            y_test=y_test,
            y_pred_mean=y_pred_mean,
            quantile_levels=quantile_levels,
            quantile_boundaries=quantile_boundaries,
            bar_probabilities=bar_probabilities,
            fold_uncertainty=fold_uncertainty,
            test_idx=test_idx,
        )
        validate_against_existing(
            method_name=method_name,
            dataset_name=dataset_name,
            targets=targets,
            tolerance=float(validation_tolerance),
        )
        dataset_results.append(
            {
                "dataset": dataset_name,
                "fold_index": int(fold_index),
                "train_indices": np.asarray(train_idx, dtype=np.int64).tolist(),
                "test_indices": np.asarray(test_idx, dtype=np.int64).tolist(),
                "bar_edges_raw": bar_edges_raw.tolist(),
                "bar_edge_count": BAR_EDGE_COUNT,
                "bar_probability_count": BAR_PROBABILITY_COUNT,
                "bar_edge_semantics": BAR_EDGE_SEMANTICS,
                "targets": targets,
            }
        )
    return {
        "method": method_name,
        "dataset_results": dataset_results,
    }


def validate_payload(payload: dict[str, Any]) -> None:
    methods = payload.get("methods", {})
    if set(methods) != {"nanoTabPFN_6M", "TabPFN_v2_official", "TabPFN_v3_official"}:
        raise ValueError(f"unexpected methods: {sorted(methods)}")
    for method_name, method_payload in methods.items():
        dataset_results = method_payload.get("dataset_results", [])
        if len(dataset_results) != 4:
            raise ValueError(f"{method_name} expected 4 dataset results, got {len(dataset_results)}")
        for dataset_result in dataset_results:
            if int(dataset_result.get("fold_index")) != 1:
                raise ValueError(f"{method_name}/{dataset_result.get('dataset')} fold_index is not 1")
            bar_edges_raw = validate_bar_edges_raw(
                dataset_result.get("bar_edges_raw"),
                name=f"{method_name}/{dataset_result.get('dataset')} bar_edges_raw",
            )
            if int(dataset_result.get("bar_edge_count", -1)) != BAR_EDGE_COUNT:
                raise ValueError(
                    f"{method_name}/{dataset_result.get('dataset')} bar_edge_count is not "
                    f"{BAR_EDGE_COUNT}"
                )
            if int(dataset_result.get("bar_probability_count", -1)) != BAR_PROBABILITY_COUNT:
                raise ValueError(
                    f"{method_name}/{dataset_result.get('dataset')} "
                    f"bar_probability_count is not {BAR_PROBABILITY_COUNT}"
                )
            if dataset_result.get("bar_edge_semantics") != BAR_EDGE_SEMANTICS:
                raise ValueError(
                    f"{method_name}/{dataset_result.get('dataset')} has unexpected "
                    "bar_edge_semantics"
                )
            targets = dataset_result.get("targets", [])
            if len(targets) != 40:
                raise ValueError(
                    f"{method_name}/{dataset_result.get('dataset')} expected 40 targets, got {len(targets)}"
                )
            for target in targets:
                probs = target.get("bar_probabilities", [])
                if len(probs) != BAR_PROBABILITY_COUNT:
                    raise ValueError(
                        f"target does not have {BAR_PROBABILITY_COUNT} bar probabilities"
                    )
                if abs(float(sum(probs)) - 1.0) > 1e-5:
                    raise ValueError("target bar probabilities do not sum to 1")
            if len(bar_edges_raw) != len(targets[0]["bar_probabilities"]) + 1:
                raise ValueError(
                    f"{method_name}/{dataset_result.get('dataset')} edge/probability "
                    "count mismatch"
                )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fold-1 TabPFN 5000-bar probabilities for selected datasets."
    )
    parser.add_argument("--datasets", type=parse_dataset_names, default=parse_dataset_names(DEFAULT_DATASETS))
    parser.add_argument("--fold-index", type=int, default=1)
    parser.add_argument("--new-instances-eval", type=int, default=200)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-features-eval", type=int, default=DEFAULT_MAX_FEATURES_EVAL)
    parser.add_argument("--max-rows-eval", type=int, default=0)
    parser.add_argument(
        "--nanotabpfn-checkpoint",
        type=str,
        default=DEFAULT_NANOTABPFN_CHECKPOINT,
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="TabICL_regression03/runs/00tables/fold1_tabpfn_bar_probabilities_5000.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validation-tolerance", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    output_path = Path(args.output_json)
    ensure_output_path(output_path, overwrite=bool(args.overwrite))
    quantile_levels = quantile_levels_for_bins(int(args.num_bins))
    device = torch.device(str(args.device))

    print("Loading rows200 datasets...", flush=True)
    datasets = get_regression_datasets(
        max_features_eval=int(args.max_features_eval),
        max_rows_eval=int(args.max_rows_eval),
        new_instances_eval=int(args.new_instances_eval),
        random_state=int(args.random_state),
        verbose=True,
        use_cache=True,
        dataset_names=list(args.datasets),
    )
    missing = [name for name in args.datasets if name not in datasets]
    if missing:
        raise RuntimeError(f"missing requested dataset(s): {', '.join(missing)}")

    nano_checkpoint = Path(args.nanotabpfn_checkpoint)
    v2_factory = make_tabpfn_model_factory(
        device=str(args.device),
        model_path="auto",
        model_version="v2",
    )
    v3_factory = make_tabpfn_model_factory(
        device=str(args.device),
        model_path="auto",
        model_version="v3",
    )

    methods: dict[str, Any] = {}
    methods["nanoTabPFN_6M"] = export_method(
        method_name="nanoTabPFN_6M",
        datasets=datasets,
        fold_index=int(args.fold_index),
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        quantile_levels=quantile_levels,
        validation_tolerance=float(args.validation_tolerance),
        predictor=lambda **kwargs: predict_nano(
            checkpoint_path=nano_checkpoint,
            device=device,
            **kwargs,
        ),
    )
    methods["TabPFN_v2_official"] = export_method(
        method_name="TabPFN_v2_official",
        datasets=datasets,
        fold_index=int(args.fold_index),
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        quantile_levels=quantile_levels,
        validation_tolerance=float(args.validation_tolerance),
        predictor=lambda **kwargs: predict_official(factory=v2_factory, **kwargs),
    )
    methods["TabPFN_v3_official"] = export_method(
        method_name="TabPFN_v3_official",
        datasets=datasets,
        fold_index=int(args.fold_index),
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        quantile_levels=quantile_levels,
        validation_tolerance=float(args.validation_tolerance),
        predictor=lambda **kwargs: predict_official(factory=v3_factory, **kwargs),
    )

    payload = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": list(methods.keys()),
            "datasets": list(datasets.keys()),
            "fold_index": int(args.fold_index),
            "new_instances_eval": int(args.new_instances_eval),
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "num_bins": int(args.num_bins),
            "quantile_levels": quantile_levels.tolist(),
            "bar_edge_count": BAR_EDGE_COUNT,
            "bar_probability_count": BAR_PROBABILITY_COUNT,
            "bar_probability_source": "softmax(full_output.logits)",
            "bar_edge_semantics": BAR_EDGE_SEMANTICS,
            "nanotabpfn_checkpoint": str(nano_checkpoint),
        },
        "methods": methods,
    }
    validate_payload(payload)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, allow_nan=False)
        f.write("\n")
    with tmp_path.open("r", encoding="utf-8") as f:
        validate_payload(json.load(f))
    tmp_path.replace(output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {output_path} ({size_mb:.2f} MB)", flush=True)
    return payload


if __name__ == "__main__":
    main()
