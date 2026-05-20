from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabicl_style import evaluation


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
