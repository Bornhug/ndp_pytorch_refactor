"""
TabArena-style evaluation for nanoTabPFN (version09).

This script mirrors the evaluation protocol used in the referenced notebook/script:
- Load TabArena OpenML tasks (task_ids list) and filter by dataset qualities.
- Subsample to a fixed number of instances (stratified).
- Apply a ColumnTransformer preprocessor (numeric coercion + ordinal encoding).
- Evaluate with 5-fold StratifiedKFold cross-validation.
- Report ROC-AUC per dataset and the mean ROC-AUC across datasets.
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch

import openml
from openml.tasks import TaskType

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer

# Add parent directory to path (match evaluation.py conventions)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SKLEARN_DATA_HOME = Path(
    os.environ.get("SCIKIT_LEARN_DATA", str(ROOT / ".sklearn_data"))
).resolve()

from evaluation import NanoTabPFNClassifier  # reuse sklearn-like wrapper

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _fetch_task_data_as_dataframe(task, dataset, *, verbose: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetch (X, y) for an OpenML task as pandas objects.

    Primary path matches the reference script: dataset.get_data(...).
    Fallback path uses sklearn.fetch_openml to avoid OpenML parquet/minio issues
    seen with some openml/minio version combinations.
    """
    try:
        X, y, _, _ = dataset.get_data(
            target=task.target_name, dataset_format="dataframe"
        )
        return X, y
    except TypeError as e:
        # Known failure mode: Minio.fget_object(...) unexpected keyword request_headers
        # Use sklearn's OpenML fetcher instead (downloads ARFF/CSV via sklearn).
        if verbose:
            print(
                f"  Falling back to sklearn.fetch_openml for dataset_id={task.dataset_id} "
                f"(openml/minio incompatibility: {e})",
                flush=True,
            )
        data_home = str(SKLEARN_DATA_HOME)
        try:
            bunch = fetch_openml(
                data_id=int(task.dataset_id),
                as_frame=True,
                data_home=data_home,
            )
        except ValueError as fetch_err:
            # Clean cached OpenML files on md5 mismatch, then retry once.
            if "md5 checksum" not in str(fetch_err).lower():
                raise
            cache_dir = SKLEARN_DATA_HOME / "openml"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            if verbose:
                print(
                    f"  Cleared sklearn OpenML cache at {cache_dir} after md5 mismatch; retrying.",
                    flush=True,
                )
            bunch = fetch_openml(
                data_id=int(task.dataset_id),
                as_frame=True,
                data_home=data_home,
            )
        frame = bunch.frame
        if frame is None:
            raise
        if task.target_name not in frame.columns:
            raise RuntimeError(
                f"Target column '{task.target_name}' not found in fetched frame for dataset_id={task.dataset_id}"
            ) from e
        y = frame[task.target_name]
        X = frame.drop(columns=[task.target_name])
        return X, y


def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    Fit a preprocessor that:
      - identifies numeric vs categorical columns (robust to numerics stored as strings)
      - drops constant columns (<=1 unique non-NaN value)
      - ordinal-encodes categorical columns
      - coerces numeric columns to numeric (NaNs allowed)
    """
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
    max_features_eval: int = 10,
    new_instances_eval: int = 200,
    target_classes_filter: int = 2,
    random_state: int = 0,
    verbose: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load TabArena OpenML tasks (v0.1 task list) and return a mapping:
      dataset_name -> (X, y)

    Matches the filtering and preprocessing logic from the reference script:
      - classification tasks only
      - NumberOfFeatures <= max_features_eval
      - NumberOfClasses <= target_classes_filter
      - PercentageOfInstancesWithMissingValues == 0
      - MinorityClassPercentage >= 2.5
      - subsample to new_instances_eval instances via stratified train_test_split
      - label-encode y
      - apply get_feature_preprocessor
    """
    task_ids = [
        363612,
        363613,
        363614,
        363615,
        363616,
        363618,
        363619,
        363620,
        363621,
        363623,
        363624,
        363625,
        363626,
        363627,
        363628,
        363629,
        363630,
        363631,
        363632,
        363671,
        363672,
        363673,
        363674,
        363675,
        363676,
        363677,
        363678,
        363679,
        363681,
        363682,
        363683,
        363684,
        363685,
        363686,
        363689,
        363691,
        363693,
        363694,
        363696,
        363697,
        363698,
        363699,
        363700,
        363702,
        363704,
        363705,
        363706,
        363707,
        363708,
        363711,
        363712,
    ]

    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    total = len(task_ids)
    for i, task_id in enumerate(task_ids, start=1):
        if verbose:
            print(f"[{i}/{total}] Fetching task_id={task_id} ...", flush=True)
        task = openml.tasks.get_task(task_id, download_splits=False)
        if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            if verbose:
                print("  Skipped: not a supervised classification task", flush=True)
            continue

        dataset = task.get_dataset(download_data=False)
        qualities = dataset.qualities

        if (
            int(qualities["NumberOfFeatures"]) > 100
            or int(qualities["NumberOfClasses"]) > target_classes_filter
            or float(qualities["PercentageOfInstancesWithMissingValues"]) > 0.0
            or float(qualities["MinorityClassPercentage"]) < 2.5
        ):
            if verbose:
                print(
                    "  Skipped by quality filter: "
                    f"features={int(qualities['NumberOfFeatures'])}, "
                    f"classes={int(qualities['NumberOfClasses'])}, "
                    f"missing%={float(qualities['PercentageOfInstancesWithMissingValues'])}, "
                    f"minority%={float(qualities['MinorityClassPercentage'])}",
                    flush=True,
                )
            continue

        if verbose:
            print(f"  Loading dataset='{dataset.name}' (dataset_id={task.dataset_id}) ...", flush=True)
        X, y = _fetch_task_data_as_dataframe(task, dataset, verbose=verbose)

        if new_instances_eval < len(y):
            _, X_sub, _, y_sub = train_test_split(
                X,
                y,
                test_size=new_instances_eval,
                stratify=y,
                random_state=random_state,
            )
        else:
            X_sub = X
            y_sub = y

        X_np = X_sub.to_numpy(copy=True)
        y_np = y_sub.to_numpy(copy=True)

        label_encoder = LabelEncoder()
        y_np = label_encoder.fit_transform(y_np)

        preprocessor = get_feature_preprocessor(X_np)
        X_np = preprocessor.fit_transform(X_np)

        datasets[str(dataset.name)] = (
            np.asarray(X_np, dtype=np.float32),
            np.asarray(y_np, dtype=np.int64),
        )
        if verbose:
            print(
                f"  Added: {dataset.name} | X.shape={datasets[str(dataset.name)][0].shape} | y.shape={datasets[str(dataset.name)][1].shape} "
                f"(kept so far: {len(datasets)})",
                flush=True,
            )

    return datasets


def eval_model(
    model: NanoTabPFNClassifier,
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    n_splits: int = 5,
    random_state: int = 0,
    log_csv_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a sklearn-like model with 5-fold StratifiedKFold and ROC-AUC.

    Returns metrics dict:
      - "{dataset_name}/ROC AUC": per-dataset ROC-AUC
      - "ROC AUC": mean ROC-AUC across datasets
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics: Dict[str, float] = {}
    items = list(datasets.items())
    iterator = items
    if tqdm is not None:
        iterator = tqdm(items, desc="Evaluating datasets", unit="dataset")
    for dataset_name, (X, y) in iterator:
        targets = []
        probabilities = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)

            # Optional logging: only for the first test sample of the first fold to keep CSV manageable.
            if log_csv_path is not None and fold_idx == 1:
                # Predict logits/proba for a single sample to generate debug CSV.
                _ = model.predict_logits(
                    X_test[:1],
                    desc=f"{dataset_name} fold {fold_idx}/{n_splits} (logged)",
                    log_json_path=log_csv_path,
                    log_label=y_test[:1],
                )
            y_proba = model.predict_proba(
                X_test,
                desc=f"{dataset_name} fold {fold_idx}/{n_splits}"
            )

            # Binary: use probability for positive class (matches reference script)
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba_eval = y_proba[:, 1]
            else:
                y_proba_eval = y_proba

            targets.append(y_test)
            probabilities.append(y_proba_eval)

        targets_np = np.concatenate(targets, axis=0)
        probabilities_np = np.concatenate(probabilities, axis=0)

        roc_auc = roc_auc_score(targets_np, probabilities_np, multi_class="ovr")
        metrics[f"{dataset_name}/ROC AUC"] = float(roc_auc)

    if metrics:
        metrics["ROC AUC"] = float(np.mean(list(metrics.values())))

    return metrics


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    """
    Local checkpoint loader mirroring nanoTabPFN_version09/evaluation.py.
    """
    from train import NanoTabPFNModel
    from ndp_discrete.process import D3PM, D3PMSchedule

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint["config"]

    use_absorbing = config_dict["diffusion"].get("use_absorbing", False)
    vocab_size = config_dict["model"]["num_outputs"] + (1 if use_absorbing else 0)

    model = NanoTabPFNModel(
        num_features=config_dict["model"]["num_features"],
        num_outputs=vocab_size,
        embedding_size=config_dict["model"]["embedding_size"],
        num_attention_heads=config_dict["model"]["num_attention_heads"],
        num_layers=config_dict["model"]["num_layers"],
    )
    state_dict = checkpoint["state_dict"]
    state_dict.pop("feature_mean", None)
    state_dict.pop("feature_std", None)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    trans_type = config_dict["diffusion"].get("transition_mat_type", "uniform")
    model_prediction = config_dict["diffusion"].get("model_prediction", "x_start")
    beta_type = "jsd" if use_absorbing else config_dict["diffusion"].get("schedule", "cosine")
    schedule = D3PMSchedule.make_uniform(
        T=config_dict["diffusion"]["timesteps"],
        vocab_size=vocab_size,
        beta_start=config_dict["diffusion"]["beta_start"],
        beta_end=config_dict["diffusion"]["beta_end"],
        beta_type=beta_type,
        transition_mat_type=trans_type,
        device=device,
        dtype=torch.float32,
    )
    process = D3PM(
        model,
        schedule,
        model_prediction=model_prediction,
        transition_mat_type=trans_type,
    )

    return model, process, config_dict


def main() -> Dict[str, float]:
    import argparse

    parser = argparse.ArgumentParser(description="TabArena-style evaluation (version09)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="ddim",
        choices=["direct", "ddpm", "ddim"],
        help="Inference method used inside NanoTabPFNClassifier",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=5,
        help="Number of steps for DDIM/DDPM (ignored for direct)",
    )
    parser.add_argument("--show-progress", action="store_true", help="Show per-fold progress bars during sampling")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument("--max-features-eval", type=int, default=10)
    parser.add_argument("--new-instances-eval", type=int, default=200)
    parser.add_argument("--target-classes-filter", type=int, default=2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--log-csv-path",
        type=str,
        default=None,
        help="Optional CSV path to log per-step diffusion details for the first test sample.",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    model, process, config_dict = _load_checkpoint(checkpoint_path, device)

    classifier = NanoTabPFNClassifier(
        model=model,
        process=process,
        device=device,
        num_classes=config_dict["model"]["num_outputs"],
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        show_progress=args.show_progress,
    )

    print("Loading OpenML TabArena tasks...")
    datasets = get_openml_datasets(
        max_features_eval=args.max_features_eval,
        new_instances_eval=args.new_instances_eval,
        target_classes_filter=args.target_classes_filter,
        random_state=args.random_state,
        verbose=True,
    )
    print(f"Loaded {len(datasets)} dataset(s).")

    metrics = eval_model(
        classifier,
        datasets,
        n_splits=args.n_splits,
        random_state=args.random_state,
        log_csv_path=args.log_csv_path,
    )

    if not metrics:
        print("No datasets passed filtering; no metrics to report.")
        return {}

    # Print per-dataset metrics and average
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.6f}")

    return metrics


if __name__ == "__main__":
    main()
