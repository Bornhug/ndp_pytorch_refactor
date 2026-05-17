from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import KFold


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from finetuning import common
from finetuning import evaluation as finetune_evaluation


def _build_full_dataset(num_rows: int = 20, num_features: int = 4) -> tuple[np.ndarray, np.ndarray]:
    X_full = np.arange(num_rows * num_features, dtype=np.float32).reshape(num_rows, num_features)
    y_full = np.arange(num_rows, dtype=np.float32)
    return X_full, y_full


def _write_task_dir(path: Path, *, dataset_name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "last.pt").write_bytes(b"checkpoint")
    with (path / "split.json").open("w", encoding="utf-8") as f:
        json.dump({"dataset_name": dataset_name}, f)


class FinetuningMultifoldTests(unittest.TestCase):
    def test_add_shared_cli_args_defaults_to_five_splits(self) -> None:
        parser = argparse.ArgumentParser()
        common.add_shared_cli_args(parser, include_dataset=True)

        args = parser.parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--dataset",
                "abalone",
                "--output-dir",
                "out",
            ]
        )

        self.assertEqual(args.n_splits, 5)

        args_legacy = parser.parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--dataset",
                "abalone",
                "--output-dir",
                "out",
                "--n-splits",
                "1",
            ]
        )
        self.assertEqual(args_legacy.n_splits, 1)

    def test_build_task_splits_matches_kfold_after_subsampling(self) -> None:
        X_full, y_full = _build_full_dataset(num_rows=20, num_features=4)
        new_instances_eval = 15
        random_state = 7
        subsample_indices = common._make_subsample_indices(
            len(y_full),
            new_instances_eval=new_instances_eval,
            random_state=random_state,
        )
        X_eval = np.nan_to_num(X_full[subsample_indices], nan=0.0).astype(np.float32)
        y_eval = y_full[subsample_indices].astype(np.float32)

        with patch.object(common, "get_regression_datasets", return_value={"toy": (X_eval, y_eval)}):
            with patch.object(common, "_load_full_dataset_arrays", return_value=(X_full, y_full)):
                task_splits = common.build_task_splits(
                    "toy",
                    max_features_eval=8,
                    new_instances_eval=new_instances_eval,
                    random_state=random_state,
                    n_splits=5,
                    use_cache=True,
                    verbose=False,
                )

        expected_splits = list(
            KFold(n_splits=5, shuffle=True, random_state=random_state).split(X_eval)
        )
        self.assertEqual(len(task_splits), 5)

        for fold_index, (task_split, (train_idx, test_idx)) in enumerate(
            zip(task_splits, expected_splits),
            start=1,
        ):
            self.assertEqual(task_split.fold_index, fold_index)
            self.assertEqual(task_split.n_splits, 5)
            self.assertEqual(task_split.outer_splitter, "kfold")
            self.assertEqual(task_split.outer_split_random_state, random_state)
            self.assertEqual(task_split.inner_split_random_state, random_state + fold_index)
            self.assertEqual(task_split.subsample_indices, subsample_indices.tolist())
            self.assertEqual(task_split.outer_context_indices, train_idx.tolist())
            self.assertEqual(task_split.outer_target_indices, test_idx.tolist())

    def test_task_split_payload_round_trip_supports_legacy_and_kfold(self) -> None:
        X_full, y_full = _build_full_dataset(num_rows=20, num_features=4)

        with patch.object(common, "get_regression_datasets", return_value={"toy": (X_full, y_full)}):
            with patch.object(common, "_load_full_dataset_arrays", return_value=(X_full, y_full)):
                legacy_split = common.build_task_split(
                    "toy",
                    max_features_eval=8,
                    new_instances_eval=20,
                    random_state=3,
                    n_splits=1,
                    use_cache=True,
                    verbose=False,
                )
                kfold_split = common.build_task_split(
                    "toy",
                    max_features_eval=8,
                    new_instances_eval=20,
                    random_state=3,
                    n_splits=5,
                    fold_index=2,
                    use_cache=True,
                    verbose=False,
                )

        legacy_payload = common.task_split_to_payload(legacy_split)
        self.assertEqual(legacy_payload["n_splits"], 1)
        self.assertEqual(legacy_payload["fold_index"], 1)
        self.assertEqual(legacy_payload["outer_splitter"], "shuffle_split")

        legacy_payload_compat = dict(legacy_payload)
        legacy_payload_compat.pop("n_splits")
        legacy_payload_compat.pop("fold_index")
        legacy_payload_compat.pop("outer_splitter")

        with patch.object(common, "_load_full_dataset_arrays", return_value=(X_full, y_full)):
            legacy_round_trip = common.task_split_from_payload(
                legacy_payload_compat,
                use_cache=True,
                verbose=False,
            )
            kfold_round_trip = common.task_split_from_payload(
                common.task_split_to_payload(kfold_split),
                use_cache=True,
                verbose=False,
            )

        self.assertEqual(legacy_round_trip.n_splits, 1)
        self.assertEqual(legacy_round_trip.fold_index, 1)
        self.assertEqual(legacy_round_trip.outer_splitter, "shuffle_split")
        self.assertEqual(kfold_round_trip.n_splits, 5)
        self.assertEqual(kfold_round_trip.fold_index, 2)
        self.assertEqual(kfold_round_trip.outer_splitter, "kfold")
        self.assertEqual(
            kfold_round_trip.outer_target_indices,
            kfold_split.outer_target_indices,
        )

    def test_discovery_supports_legacy_and_multifold_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            legacy_root = root / "legacy_single"
            _write_task_dir(legacy_root, dataset_name="legacy_ds")
            discovered_legacy = finetune_evaluation._scan_dataset_task_dirs(legacy_root)
            self.assertEqual(list(discovered_legacy.keys()), ["legacy_ds"])
            self.assertEqual(discovered_legacy["legacy_ds"], [legacy_root])

            single_multi_root = root / "single_multi"
            _write_task_dir(single_multi_root / "fold_1", dataset_name="single_multi_ds")
            _write_task_dir(single_multi_root / "fold_2", dataset_name="single_multi_ds")
            discovered_single_multi = finetune_evaluation._scan_dataset_task_dirs(single_multi_root)
            self.assertEqual(list(discovered_single_multi.keys()), ["single_multi_ds"])
            self.assertEqual(len(discovered_single_multi["single_multi_ds"]), 2)

            all_multi_root = root / "all_multi"
            _write_task_dir(all_multi_root / "abalone" / "fold_1", dataset_name="abalone")
            _write_task_dir(all_multi_root / "abalone" / "fold_2", dataset_name="abalone")
            _write_task_dir(all_multi_root / "boston" / "fold_1", dataset_name="boston")
            discovered_all_multi = finetune_evaluation._scan_dataset_task_dirs(all_multi_root)
            self.assertEqual(sorted(discovered_all_multi.keys()), ["abalone", "boston"])
            self.assertEqual(len(discovered_all_multi["abalone"]), 2)
            self.assertEqual(len(discovered_all_multi["boston"]), 1)

    def test_aggregate_dataset_and_overall_metrics_follow_fold_means(self) -> None:
        fold_results_a = [
            {
                "dataset_name": "dataset_a",
                "fold_index": 1,
                "n_splits": 2,
                "metrics": {"R2": 1.0, "RMSE": 2.0, "MAE": 3.0},
                "y_true": [1.0],
                "y_pred_repetitions": [[1.1], [1.2]],
                "y_pred_mean": [1.15],
                "y_true_normalized": [0.1],
                "y_pred_normalized": [0.2],
                "n_repeats": 2,
                "num_sampling_steps": 5,
                "sampling_method": "ddpm",
                "checkpoint_path": "a1.pt",
                "split_path": "a1.json",
            },
            {
                "dataset_name": "dataset_a",
                "fold_index": 2,
                "n_splits": 2,
                "metrics": {"R2": 3.0, "RMSE": 4.0, "MAE": 5.0},
                "y_true": [2.0],
                "y_pred_repetitions": [[2.1], [2.2]],
                "y_pred_mean": [2.15],
                "y_true_normalized": [0.3],
                "y_pred_normalized": [0.4],
                "n_repeats": 2,
                "num_sampling_steps": 5,
                "sampling_method": "ddpm",
                "checkpoint_path": "a2.pt",
                "split_path": "a2.json",
            },
        ]
        fold_results_b = [
            {
                "dataset_name": "dataset_b",
                "fold_index": 1,
                "n_splits": 2,
                "metrics": {"R2": 5.0, "RMSE": 6.0, "MAE": 7.0},
                "y_true": [3.0],
                "y_pred_repetitions": [[3.1], [3.2]],
                "y_pred_mean": [3.15],
                "y_true_normalized": [0.5],
                "y_pred_normalized": [0.6],
                "n_repeats": 2,
                "num_sampling_steps": 5,
                "sampling_method": "ddpm",
                "checkpoint_path": "b1.pt",
                "split_path": "b1.json",
            },
            {
                "dataset_name": "dataset_b",
                "fold_index": 2,
                "n_splits": 2,
                "metrics": {"R2": 7.0, "RMSE": 8.0, "MAE": 9.0},
                "y_true": [4.0],
                "y_pred_repetitions": [[4.1], [4.2]],
                "y_pred_mean": [4.15],
                "y_true_normalized": [0.7],
                "y_pred_normalized": [0.8],
                "n_repeats": 2,
                "num_sampling_steps": 5,
                "sampling_method": "ddpm",
                "checkpoint_path": "b2.pt",
                "split_path": "b2.json",
            },
        ]

        dataset_a = finetune_evaluation._aggregate_dataset_result(
            dataset_name="dataset_a",
            fold_results=fold_results_a,
        )
        dataset_b = finetune_evaluation._aggregate_dataset_result(
            dataset_name="dataset_b",
            fold_results=fold_results_b,
        )

        self.assertEqual(dataset_a["metrics"]["R2_FOLD_VALUES"], [1.0, 3.0])
        self.assertAlmostEqual(dataset_a["metrics"]["R2"], 2.0)
        self.assertEqual(dataset_a["completed_fold_indices"], [1, 2])
        self.assertEqual(dataset_a["y_true"], [1.0, 2.0])
        np.testing.assert_allclose(
            np.asarray(dataset_a["y_pred_repetitions"], dtype=np.float32),
            np.asarray([[1.1, 2.1], [1.2, 2.2]], dtype=np.float32),
        )

        overall_metrics = finetune_evaluation._aggregate_overall_metrics(
            {
                "dataset_a": dataset_a,
                "dataset_b": dataset_b,
            }
        )
        self.assertEqual(overall_metrics["R2_FOLD_VALUES"], [3.0, 5.0])
        self.assertAlmostEqual(overall_metrics["R2"], 4.0)
        self.assertEqual(overall_metrics["RMSE_FOLD_VALUES"], [4.0, 6.0])
        self.assertAlmostEqual(overall_metrics["MAE"], 6.0)

    def test_dataset_summary_preserves_requested_split_count_when_all_folds_fail(self) -> None:
        summary = common.summarize_dataset_finetune_results(
            "boston",
            [],
            output_root=Path(tempfile.gettempdir()) / "tabicl_summary_test",
            n_splits=5,
            failures=[
                {
                    "dataset_name": "boston",
                    "fold_index": 1,
                    "n_splits": 5,
                    "error": "checkpoint mismatch",
                }
            ],
        )

        self.assertEqual(summary["n_splits"], 5)
        self.assertEqual(summary["completed"], [])
        self.assertEqual(summary["completed_fold_indices"], [])
        self.assertEqual(summary["failures"][0]["fold_index"], 1)


if __name__ == "__main__":
    unittest.main()
