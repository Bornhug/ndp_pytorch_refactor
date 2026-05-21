from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabicl_style import evaluation


class _DeterministicRegressor:
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        return None

    def predict_repeated(self, X_test: np.ndarray, repeat_count: int) -> np.ndarray:
        prediction = np.asarray(X_test[:, 0], dtype=np.float32)
        return np.tile(prediction[None, :], (int(repeat_count), 1))


class _TwoFoldSplitter:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def split(self, X: np.ndarray):
        return [
            (np.asarray([2, 3]), np.asarray([0, 1])),
            (np.asarray([0, 1]), np.asarray([2, 3])),
        ]


def _patch_tiny_dataset(monkeypatch, *, rows: int = 8) -> None:
    X = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    y = np.arange(rows, dtype=np.float32)

    monkeypatch.setattr(evaluation, "TABPFN_REGRESSION_DATASETS", {"tiny": 1})
    monkeypatch.setattr(
        evaluation,
        "_fetch_openml_regression",
        lambda name, data_id, *, verbose=False: (X.copy(), y.copy()),
    )


def _load_row_count(
    monkeypatch,
    *,
    new_instances_eval: int,
    max_rows_eval: int = 0,
    rows: int = 8,
) -> int | None:
    _patch_tiny_dataset(monkeypatch, rows=rows)
    datasets = evaluation.get_regression_datasets(
        max_features_eval=32,
        max_rows_eval=max_rows_eval,
        new_instances_eval=new_instances_eval,
        random_state=0,
        verbose=False,
        use_cache=False,
        dataset_names=["tiny"],
    )
    if "tiny" not in datasets:
        return None
    return int(datasets["tiny"][0].shape[0])


def test_new_instances_eval_zero_keeps_all_rows(monkeypatch) -> None:
    assert _load_row_count(monkeypatch, new_instances_eval=0) == 8


def test_new_instances_eval_negative_keeps_all_rows(monkeypatch) -> None:
    assert _load_row_count(monkeypatch, new_instances_eval=-1) == 8


def test_new_instances_eval_positive_subsamples(monkeypatch) -> None:
    assert _load_row_count(monkeypatch, new_instances_eval=3) == 3


def test_new_instances_eval_larger_than_dataset_keeps_all_rows(monkeypatch) -> None:
    assert _load_row_count(monkeypatch, new_instances_eval=99) == 8


def test_max_rows_eval_keeps_dataset_at_limit(monkeypatch) -> None:
    assert (
        _load_row_count(
            monkeypatch,
            new_instances_eval=0,
            max_rows_eval=1000,
            rows=1000,
        )
        == 1000
    )


def test_max_rows_eval_skips_dataset_above_limit(monkeypatch) -> None:
    assert (
        _load_row_count(
            monkeypatch,
            new_instances_eval=0,
            max_rows_eval=1000,
            rows=1001,
        )
        is None
    )


def test_max_rows_eval_zero_disables_row_filter(monkeypatch) -> None:
    assert (
        _load_row_count(
            monkeypatch,
            new_instances_eval=0,
            max_rows_eval=0,
            rows=1001,
        )
        == 1001
    )


def test_max_rows_eval_happens_before_subsampling(monkeypatch) -> None:
    assert (
        _load_row_count(
            monkeypatch,
            new_instances_eval=200,
            max_rows_eval=1000,
            rows=1001,
        )
        is None
    )


def test_eval_model_se_uses_folds_within_dataset_and_datasets_overall(monkeypatch) -> None:
    def fake_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        base = float(np.mean(y_true))
        return {
            "R2": base,
            "RMSE": base + 10.0,
            "MAE": base + 20.0,
        }

    monkeypatch.setattr(evaluation, "KFold", _TwoFoldSplitter)
    monkeypatch.setattr(evaluation, "_compute_regression_metrics", fake_metrics)

    X = np.arange(4, dtype=np.float32).reshape(4, 1)
    datasets = {
        "dataset_a": (X, np.asarray([1.0, 1.0, 3.0, 3.0], dtype=np.float32)),
        "dataset_b": (X, np.asarray([5.0, 5.0, 9.0, 9.0], dtype=np.float32)),
    }

    metrics, details = evaluation.eval_model(
        _DeterministicRegressor(),
        datasets,
        n_splits=2,
        n_repeats=1,
        return_details=True,
    )

    dataset_a_metrics = details["datasets"]["dataset_a"]["metrics"]
    dataset_b_metrics = details["datasets"]["dataset_b"]["metrics"]

    assert dataset_a_metrics["R2_FOLD_VALUES"] == [1.0, 3.0]
    assert dataset_a_metrics["R2"] == 2.0
    assert np.isclose(dataset_a_metrics["R2_SE"], np.std([1.0, 3.0]) / np.sqrt(2.0))

    assert dataset_b_metrics["R2_FOLD_VALUES"] == [5.0, 9.0]
    assert dataset_b_metrics["R2"] == 7.0
    assert np.isclose(dataset_b_metrics["R2_SE"], np.std([5.0, 9.0]) / np.sqrt(2.0))

    assert metrics["R2"] == 4.5
    assert np.isclose(metrics["R2_SE"], np.std([2.0, 7.0]) / np.sqrt(2.0))
    assert metrics["R2_DATASET_VALUES"] == [2.0, 7.0]

    assert details["overall_metrics"]["RMSE"] == 14.5
    assert np.isclose(
        details["overall_metrics"]["RMSE_SE"],
        np.std([12.0, 17.0]) / np.sqrt(2.0),
    )
    assert details["overall_metrics"]["RMSE_DATASET_VALUES"] == [12.0, 17.0]
    assert details["overall_metrics"]["MAE"] == 24.5
    assert np.isclose(
        details["overall_metrics"]["MAE_SE"],
        np.std([22.0, 27.0]) / np.sqrt(2.0),
    )
