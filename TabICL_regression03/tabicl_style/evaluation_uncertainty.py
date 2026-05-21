"""Compute uncertainty calibration metrics from saved evaluation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_dataset_names(raw: str | None) -> list[str] | None:
    """Parse a comma-separated dataset list while preserving order."""
    if raw is None:
        return None
    names = [part.strip() for part in str(raw).split(",") if part.strip()]
    return names or None


def default_output_path(input_json: Path) -> Path:
    """Return the default uncertainty JSON path next to the evaluation JSON."""
    return input_json.with_name(f"{input_json.stem}_uncertainty.json")


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _standard_error(values: list[float]) -> float:
    """Return the standard error over the provided metric values."""
    if not values:
        return float("nan")
    values_np = np.asarray(values, dtype=np.float64)
    return float(np.std(values_np) / np.sqrt(values_np.shape[0]))


def compute_qice(
    y_true: Any,
    y_pred_repetitions: Any,
    *,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Compute Quantile Interval Coverage Error for one dataset.

    ``y_pred_repetitions`` is expected to have shape ``[R, N]``: repeated
    predictive samples for each of the ``N`` true target values.
    """
    if int(num_bins) <= 1:
        raise ValueError(f"num_bins must be greater than 1, got {num_bins}")

    y_true_np = _as_float_array(y_true, name="y_true")
    y_pred_np = _as_float_array(y_pred_repetitions, name="y_pred_repetitions")

    if y_true_np.ndim != 1:
        raise ValueError(f"y_true must be one-dimensional, got shape {y_true_np.shape}")
    if y_pred_np.ndim != 2:
        raise ValueError(
            "y_pred_repetitions must be two-dimensional with shape [repeats, targets], "
            f"got shape {y_pred_np.shape}"
        )
    if y_pred_np.shape[1] != y_true_np.shape[0]:
        raise ValueError(
            "y_pred_repetitions target count must match y_true length, "
            f"got {y_pred_np.shape[1]} and {y_true_np.shape[0]}"
        )
    if y_pred_np.shape[0] <= 0:
        raise ValueError("y_pred_repetitions must contain at least one repetition")

    finite_mask = np.isfinite(y_true_np) & np.isfinite(y_pred_np).all(axis=0)
    excluded_count = int(y_true_np.shape[0] - np.count_nonzero(finite_mask))
    y_true_valid = y_true_np[finite_mask]
    y_pred_valid = y_pred_np[:, finite_mask]
    valid_count = int(y_true_valid.shape[0])
    if valid_count == 0:
        raise ValueError("QICE requires at least one finite target with finite predictions")

    bins = int(num_bins)
    quantile_levels = np.arange(1, bins, dtype=np.float64) / float(bins)
    boundaries = np.quantile(y_pred_valid, quantile_levels, axis=0)

    bin_indices = np.empty(valid_count, dtype=np.int64)
    for target_idx in range(valid_count):
        bin_indices[target_idx] = int(
            np.searchsorted(boundaries[:, target_idx], y_true_valid[target_idx], side="left")
        )

    bin_counts = np.bincount(bin_indices, minlength=bins).astype(np.int64)
    bin_proportions = bin_counts.astype(np.float64) / float(valid_count)
    ideal_proportion = 1.0 / float(bins)
    qice = float(np.mean(np.abs(bin_proportions - ideal_proportion)))

    return {
        "QICE": qice,
        "bin_proportions": bin_proportions.tolist(),
        "bin_counts": bin_counts.tolist(),
        "target_count": valid_count,
        "repeat_count": int(y_pred_np.shape[0]),
        "quantile_levels": quantile_levels.tolist(),
        "ideal_proportion": ideal_proportion,
        "excluded_non_finite_count": excluded_count,
    }


def compute_dataset_uncertainty(
    dataset_payload: dict[str, Any],
    *,
    dataset_name: str,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Compute dataset-level QICE plus fold-level SE when folds are present."""
    if "y_true" not in dataset_payload:
        raise ValueError(f"dataset {dataset_name} is missing required key: y_true")
    if "y_pred_repetitions" not in dataset_payload:
        raise ValueError(
            f"dataset {dataset_name} is missing required key: y_pred_repetitions"
        )

    result = compute_qice(
        dataset_payload["y_true"],
        dataset_payload["y_pred_repetitions"],
        num_bins=int(num_bins),
    )

    fold_results: list[dict[str, Any]] = []
    folds = dataset_payload.get("folds")
    if isinstance(folds, list) and folds:
        for position, fold_payload in enumerate(folds, start=1):
            if not isinstance(fold_payload, dict):
                raise ValueError(
                    f"dataset {dataset_name} fold {position} must be a JSON object"
                )
            if "y_true" not in fold_payload:
                raise ValueError(
                    f"dataset {dataset_name} fold {position} is missing required key: y_true"
                )
            if "y_pred_repetitions" not in fold_payload:
                raise ValueError(
                    f"dataset {dataset_name} fold {position} is missing required key: "
                    "y_pred_repetitions"
                )
            fold_qice = compute_qice(
                fold_payload["y_true"],
                fold_payload["y_pred_repetitions"],
                num_bins=int(num_bins),
            )
            fold_results.append(
                {
                    "fold_index": int(fold_payload.get("fold_index", position)),
                    **fold_qice,
                }
            )

    fold_qice_values = [float(fold_result["QICE"]) for fold_result in fold_results]
    if not fold_qice_values:
        fold_qice_values = [float(result["QICE"])]

    result["QICE_SE"] = _standard_error(fold_qice_values)
    result["QICE_FOLD_VALUES"] = fold_qice_values
    result["folds"] = fold_results
    return result


def load_evaluation_payload(input_json: Path) -> dict[str, Any]:
    """Load and validate the top-level evaluation JSON object."""
    with input_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("input JSON must contain a top-level object")
    if "datasets" not in payload:
        raise ValueError("input JSON is missing required top-level key: datasets")
    if not isinstance(payload["datasets"], dict):
        raise ValueError("input JSON key datasets must be an object")
    return payload


def compute_uncertainty_payload(
    evaluation_payload: dict[str, Any],
    *,
    input_json: Path,
    output_json: Path,
    num_bins: int = 10,
    dataset_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute QICE for selected datasets from an evaluation JSON payload."""
    if int(num_bins) <= 1:
        raise ValueError(f"num_bins must be greater than 1, got {num_bins}")

    datasets = evaluation_payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("input JSON key datasets must be an object")

    if dataset_names is None:
        selected_names = list(datasets.keys())
    else:
        missing = [name for name in dataset_names if name not in datasets]
        if missing:
            raise ValueError(f"requested dataset(s) not found: {', '.join(missing)}")
        selected_names = list(dataset_names)
    if not selected_names:
        raise ValueError("no datasets selected for uncertainty evaluation")

    dataset_results: dict[str, Any] = {}
    for dataset_name in selected_names:
        dataset_payload = datasets[dataset_name]
        if not isinstance(dataset_payload, dict):
            raise ValueError(f"dataset {dataset_name} must be a JSON object")
        dataset_results[dataset_name] = compute_dataset_uncertainty(
            dataset_payload,
            dataset_name=dataset_name,
            num_bins=int(num_bins),
        )

    qice_values = [float(result["QICE"]) for result in dataset_results.values()]
    overall_qice = float(np.mean(qice_values))
    overall_qice_se = _standard_error(qice_values)

    return {
        "config": {
            "input_json": str(input_json),
            "output_json": str(output_json),
            "num_bins": int(num_bins),
            "datasets_requested": dataset_names,
            "datasets_evaluated": selected_names,
            "source_evaluation_config": evaluation_payload.get("config"),
        },
        "overall_uncertainty": {
            "QICE": overall_qice,
            "QICE_SE": overall_qice_se,
            "QICE_DATASET_VALUES": qice_values,
            "dataset_count": len(dataset_results),
        },
        "datasets": dataset_results,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with the formatting used by other evaluation scripts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute QICE uncertainty calibration from evaluation.py JSON output."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to a detailed evaluation.py output JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write uncertainty JSON. Defaults next to --input-json.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of QICE quantile intervals.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset subset.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    input_json = Path(args.input_json)
    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else default_output_path(input_json)
    )
    dataset_names = parse_dataset_names(args.datasets)

    evaluation_payload = load_evaluation_payload(input_json)
    uncertainty_payload = compute_uncertainty_payload(
        evaluation_payload,
        input_json=input_json,
        output_json=output_json,
        num_bins=int(args.num_bins),
        dataset_names=dataset_names,
    )
    write_json(output_json, uncertainty_payload)

    print(f"Saved uncertainty results to {output_json}")
    print(
        f"Mean QICE over {uncertainty_payload['overall_uncertainty']['dataset_count']} "
        f"dataset(s): {uncertainty_payload['overall_uncertainty']['QICE']:.6f}"
    )
    return uncertainty_payload


if __name__ == "__main__":
    main()
