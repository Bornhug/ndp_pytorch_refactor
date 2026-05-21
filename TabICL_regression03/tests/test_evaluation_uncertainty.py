from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabicl_style import evaluation_uncertainty


def test_compute_qice_known_bin_assignments() -> None:
    y_true = [0.0, 1.5, 3.5, 4.5]
    y_pred_repetitions = [
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
    ]

    result = evaluation_uncertainty.compute_qice(
        y_true,
        y_pred_repetitions,
        num_bins=2,
    )

    assert result["bin_counts"] == [2, 2]
    assert result["bin_proportions"] == [0.5, 0.5]
    assert result["QICE"] == 0.0
    assert result["target_count"] == 4
    assert result["repeat_count"] == 3
    assert result["quantile_levels"] == [0.5]


def test_compute_qice_excludes_non_finite_targets_and_prediction_columns() -> None:
    result = evaluation_uncertainty.compute_qice(
        [0.0, np.nan, 2.0],
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, np.inf],
            [2.0, 3.0, 4.0],
        ],
        num_bins=2,
    )

    assert result["excluded_non_finite_count"] == 2
    assert result["target_count"] == 1
    assert sum(result["bin_counts"]) == 1


def test_main_reads_evaluation_json_and_writes_uncertainty_json(tmp_path) -> None:
    input_json = tmp_path / "eval.json"
    output_json = tmp_path / "uncertainty.json"
    payload = {
        "config": {"checkpoint": "step.pt", "n_repeats": 3},
        "overall_metrics": {"R2": 1.0},
        "datasets": {
            "dataset_a": {
                "metrics": {"R2": 1.0},
                "y_true": [0.0, 3.5, 2.5, 4.0],
                "y_pred_repetitions": [
                    [0.0, 1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 3.0, 4.0, 5.0],
                ],
                "folds": [
                    {
                        "fold_index": 1,
                        "y_true": [0.0, 3.5],
                        "y_pred_repetitions": [
                            [0.0, 2.0],
                            [1.0, 3.0],
                            [2.0, 4.0],
                        ],
                    },
                    {
                        "fold_index": 2,
                        "y_true": [2.5, 4.0],
                        "y_pred_repetitions": [
                            [2.0, 3.0],
                            [3.0, 4.0],
                            [4.0, 5.0],
                        ],
                    },
                ],
            },
            "dataset_b": {
                "metrics": {"R2": 0.0},
                "y_true": [10.0, 12.0],
                "y_pred_repetitions": [
                    [9.0, 11.0],
                    [10.0, 12.0],
                    [11.0, 13.0],
                ],
            },
        },
    }
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluation_uncertainty.main(
        [
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--num-bins",
            "2",
            "--datasets",
            "dataset_a",
        ]
    )

    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written == result
    assert written["config"]["input_json"] == str(input_json)
    assert written["config"]["output_json"] == str(output_json)
    assert written["config"]["num_bins"] == 2
    assert written["config"]["datasets_requested"] == ["dataset_a"]
    assert written["config"]["datasets_evaluated"] == ["dataset_a"]
    assert written["config"]["source_evaluation_config"] == payload["config"]
    assert written["overall_uncertainty"]["QICE"] == 0.25
    assert written["overall_uncertainty"]["QICE_SE"] == 0.0
    assert written["overall_uncertainty"]["QICE_DATASET_VALUES"] == [0.25]
    assert list(written["datasets"].keys()) == ["dataset_a"]
    assert written["datasets"]["dataset_a"]["QICE"] == 0.25
    assert written["datasets"]["dataset_a"]["QICE_FOLD_VALUES"] == [0.0, 0.5]
    assert written["datasets"]["dataset_a"]["QICE_SE"] == pytest.approx(
        np.std([0.0, 0.5]) / np.sqrt(2.0)
    )
    assert [fold["fold_index"] for fold in written["datasets"]["dataset_a"]["folds"]] == [1, 2]


def test_default_output_path_appends_uncertainty_suffix() -> None:
    path = evaluation_uncertainty.default_output_path(Path("run/step-1_eval.json"))
    assert path == Path("run/step-1_eval_uncertainty.json")


def test_missing_top_level_datasets_fails(tmp_path) -> None:
    input_json = tmp_path / "eval.json"
    input_json.write_text(json.dumps({"config": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="datasets"):
        evaluation_uncertainty.load_evaluation_payload(input_json)


@pytest.mark.parametrize(
    ("dataset_payload", "match"),
    [
        ({"y_pred_repetitions": [[1.0]]}, "y_true"),
        ({"y_true": [1.0]}, "y_pred_repetitions"),
        ({"y_true": [1.0], "y_pred_repetitions": [1.0]}, "two-dimensional"),
        (
            {"y_true": [1.0, 2.0], "y_pred_repetitions": [[1.0]]},
            "target count",
        ),
    ],
)
def test_dataset_schema_validation_failures(dataset_payload, match) -> None:
    payload = {"datasets": {"bad": dataset_payload}}

    with pytest.raises(ValueError, match=match):
        evaluation_uncertainty.compute_uncertainty_payload(
            payload,
            input_json=Path("eval.json"),
            output_json=Path("uncertainty.json"),
        )


def test_invalid_num_bins_fails() -> None:
    with pytest.raises(ValueError, match="num_bins"):
        evaluation_uncertainty.compute_qice([1.0], [[1.0]], num_bins=1)


def test_missing_requested_dataset_fails() -> None:
    payload = {
        "datasets": {
            "present": {
                "y_true": [1.0],
                "y_pred_repetitions": [[1.0]],
            }
        }
    }

    with pytest.raises(ValueError, match="not found"):
        evaluation_uncertainty.compute_uncertainty_payload(
            payload,
            input_json=Path("eval.json"),
            output_json=Path("uncertainty.json"),
            dataset_names=["missing"],
        )


def test_overall_qice_se_is_over_dataset_qice_values() -> None:
    payload = {
        "datasets": {
            "low_qice": {
                "y_true": [0.0, 3.5],
                "y_pred_repetitions": [
                    [0.0, 2.0],
                    [1.0, 3.0],
                    [2.0, 4.0],
                ],
            },
            "high_qice": {
                "y_true": [0.0, 1.0],
                "y_pred_repetitions": [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                ],
            },
        }
    }

    result = evaluation_uncertainty.compute_uncertainty_payload(
        payload,
        input_json=Path("eval.json"),
        output_json=Path("uncertainty.json"),
        num_bins=2,
    )

    dataset_values = result["overall_uncertainty"]["QICE_DATASET_VALUES"]
    assert dataset_values == [0.0, 0.5]
    assert result["overall_uncertainty"]["QICE"] == 0.25
    assert result["overall_uncertainty"]["QICE_SE"] == pytest.approx(
        np.std(dataset_values) / np.sqrt(2.0)
    )
    assert result["datasets"]["low_qice"]["QICE_SE"] == 0.0
    assert result["datasets"]["low_qice"]["QICE_FOLD_VALUES"] == [0.0]
