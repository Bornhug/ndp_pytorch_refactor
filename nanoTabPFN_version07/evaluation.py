"""
Evaluation script for nanoTabPFN diffusion-based classifier.
Provides sklearn-compatible interface and evaluation on benchmark datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import h5py
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from inference import ddpm_sample, ddim_sample
from ndp_discrete.process import D3PM


class NanoTabPFNClassifier:
    """
    Sklearn-compatible classifier wrapper for diffusion-based nanoTabPFN.
    
    Uses reverse diffusion sampling for inference.
    """
    
    def __init__(
        self,
        model,
        process: D3PM,
        device: torch.device,
        num_classes: int = 2,
        sampling_method: str = "ddim",
        num_sampling_steps: int = 50,
        show_progress: bool = False,
    ):
        """
        Args:
            model: Trained NanoTabPFNModel
            process: D3PM instance
            device: Device to run inference on
            num_classes: Number of output classes
            sampling_method: "ddpm" (slow, accurate) or "ddim" (fast)
            num_sampling_steps: Number of steps for DDIM (ignored for DDPM)
            show_progress: Whether to display tqdm progress bars during sampling
        """
        self.model = model.to(device)
        self.model.eval()
        self.process = process
        self.device = device
        self.num_classes = num_classes
        self.sampling_method = sampling_method
        self.num_sampling_steps = num_sampling_steps
        self.show_progress = show_progress
        
        # Will be set during fit()
        self.X_train = None
        self.y_train = None
        self.y_train_col = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> NanoTabPFNClassifier:
        """
        Store training data for use in prediction.
        
        Args:
            X_train: Training features [N_train, D]
            y_train: Training labels [N_train] (integer class labels)
        
        Returns:
            self
        """
        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train.astype(np.int64)
        self.y_train_col = self.y_train.reshape(-1, 1)
        
        return self
    
    
    def predict_logits(self, X_test: np.ndarray, *, desc: str | None = None) -> np.ndarray:
        """
        Predict raw label samples/logits using reverse diffusion sampling.
        
        Args:
            X_test: Test features [N_test, D]
        
        Returns:
            probabilities: Class probabilities [N_test, C]
        """
        if self.X_train is None or self.y_train_col is None:
            raise RuntimeError("Must call fit() before predict_logits()")
        
        # Convert to tensors
        x_train = torch.from_numpy(self.X_train).to(self.device)
        y_train = torch.from_numpy(self.y_train_col).to(self.device)
        x_test = torch.from_numpy(X_test.astype(np.float32)).to(self.device)
        
        # Run reverse diffusion sampling
        with torch.no_grad():
            progress_desc = desc or f"{self.sampling_method.upper()} sampling"
            if self.sampling_method == "ddpm":
                y_pred = ddpm_sample(
                    self.model,
                    self.process,
                    x_test,
                    x_train,
                    y_train,
                    self.num_classes,
                    self.device,
                    progress=self.show_progress,
                    progress_desc=progress_desc,
                )
            elif self.sampling_method == "ddim":
                y_pred = ddim_sample(
                    self.model,
                    self.process,
                    x_test,
                    x_train,
                    y_train,
                    self.num_classes,
                    self.device,
                    num_steps=self.num_sampling_steps,
                    progress=self.show_progress,
                    progress_desc=progress_desc,
                )
            else:
                raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        
        return y_pred.cpu().numpy()
    
    def logits_to_proba(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Convert raw diffusion outputs into class probabilities.
        
        Supports scalar label diffusion (default) and falls back to softmax
        when multiple output channels are present.
        """
        logits = torch.from_numpy(y_pred)
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)

        # Multi-channel logits (should not occur with scalar labels but handled for safety)
        if logits.shape[-1] > 1:
            return torch.softmax(logits, dim=-1).numpy()

        # Scalar path
        if self.num_classes == 2:
            pos = torch.sigmoid(logits.squeeze(-1))
            proba = torch.stack([1 - pos, pos], dim=-1)
            return proba.numpy()

        class_idx = logits.squeeze(-1).round().clamp(0, self.num_classes - 1).long()
        proba = torch.zeros(logits.shape[0], self.num_classes, dtype=torch.float32)
        proba.scatter_(1, class_idx.unsqueeze(-1), 1.0)
        return proba.numpy()
    
    def predict_proba(self, X_test: np.ndarray, *, desc: str | None = None) -> np.ndarray:
        """
        Predict class probabilities using reverse diffusion sampling.
        
        Args:
            X_test: Test features [N_test, D]
        
        Returns:
            probabilities: Class probabilities [N_test, C]
        """
        y_pred = self.predict_logits(X_test, desc=desc)
        return self.logits_to_proba(y_pred)

    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X_test: Test features [N_test, D]
        
        Returns:
            predictions: Class labels [N_test]
        """
        probabilities = self.predict_proba(X_test)
        return probabilities.argmax(axis=1)


def _load_breast_cancer_dataset(
    test_size: float = 0.5,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the sklearn breast cancer dataset."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0, stratify=y
    )
    return "breast_cancer", X_train, X_test, y_train, y_test


def _load_tabarena_diabetes_dataset(
    test_size: float = 0.5,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the TabArena "diabetes" task (Pima Indians) through OpenML.
    
    The dataset corresponds to OpenML data_id=37 and matches the task subset
    used in the official nanoTabPFN evaluation notebook.
    """
    diabetes = fetch_openml(data_id=37, as_frame=True)
    X_df = diabetes.data
    y_series = diabetes.target
    
    X = X_df.to_numpy(dtype=np.float32, copy=True)
    y_values = y_series.to_numpy(copy=True)
    if np.issubdtype(y_values.dtype, np.integer):
        y = y_values.astype(np.int64, copy=False)
    else:
        y = LabelEncoder().fit_transform(y_values.astype(str))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=0,
        stratify=y,
    )
    return "tabarena_diabetes", X_train, X_test, y_train, y_test


def _load_openml_dataset(
    *,
    dataset_name: str,
    data_id: int,
    test_size: float = 0.5,
    encode_categorical: bool = True,
    sample_size: int | None = None,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generic OpenML classification dataset loader."""
    dataset = fetch_openml(data_id=data_id, as_frame=True)
    X_df = dataset.data.copy()
    y_series = dataset.target.copy()
    
    if encode_categorical:
        for col in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                X_df.loc[:, col] = (
                    pd.Categorical(X_df[col]).codes.astype(np.float32)
                )
    X = X_df.to_numpy(dtype=np.float32, copy=True)
    
    y_values = y_series.to_numpy(copy=True)
    if np.issubdtype(y_values.dtype, np.integer):
        y = y_values.astype(np.int64, copy=False)
    else:
        y = LabelEncoder().fit_transform(y_values.astype(str))
    
    if sample_size is not None and sample_size < X.shape[0]:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X = X[idx]
        y = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=0,
        stratify=y,
    )
    return dataset_name, X_train, X_test, y_train, y_test


def _load_tabarena_blood_dataset(
    test_size: float = 0.5,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _load_openml_dataset(
        dataset_name="tabarena_blood_transfusion",
        data_id=1464,
        test_size=test_size,
        encode_categorical=True,
    )


def _load_tabarena_amazon_dataset(
    test_size: float = 0.5,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _load_openml_dataset(
        dataset_name="tabarena_amazon_employee_access",
        data_id=4135,
        test_size=test_size,
        encode_categorical=True,
        sample_size=1000,
    )


def load_benchmark_datasets(
    dataset_names: List[str] | None = None,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load benchmark datasets for evaluation.
    
    Args:
        dataset_names: Optional list of dataset identifiers to load. Supported
            values: "breast_cancer", "diabetes", "blood_transfusion",
            "amazon_employee_access", "all".
    
    Returns:
        List of (name, X_train, X_test, y_train, y_test) tuples
    """
    registry: Dict[
        str,
        Callable[[], Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ] = {
        "breast_cancer": _load_breast_cancer_dataset,
        "diabetes": _load_tabarena_diabetes_dataset,
        "blood_transfusion": _load_tabarena_blood_dataset,
        "amazon_employee_access": _load_tabarena_amazon_dataset,
    }
    
    if not dataset_names:
        dataset_names = ["breast_cancer"]
    
    if len(dataset_names) == 1 and dataset_names[0].lower() == "all":
        selected = list(registry.keys())
    else:
        selected = [name.lower() for name in dataset_names]
    
    datasets = []
    for key in selected:
        if key not in registry:
            raise ValueError(
                f"Unknown dataset '{key}'. Available options: "
                f"{', '.join(registry.keys())}"
            )
        loader = registry[key]
        datasets.append(loader())
    
    return datasets


def sample_prior_tasks(
    h5_path: Path,
    num_tasks: int = 5,
    context_ratio: float = 0.25,
    seed: int = 0,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Sample evaluation tasks directly from the training H5 prior dump.
    
    Args:
        h5_path: Path to the H5 file used for pretraining
        num_tasks: Number of tasks to sample for evaluation
        context_ratio: Fraction of each task's points to use as context/train
        seed: Random seed for reproducibility
    
    Returns:
        List of (name, X_train, X_test, y_train, y_test)
    """
    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    rng = np.random.default_rng(seed)
    datasets: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    
    with h5py.File(h5_path, "r") as f:
        X = f["X"]
        y = f["y"]
        num_datapoints = f["num_datapoints"]
        
        total_tasks = X.shape[0]
        if total_tasks == 0:
            raise RuntimeError("H5 file contains no tasks.")
        
        task_indices = rng.choice(
            total_tasks, size=min(num_tasks, total_tasks), replace=False
        )
        
        for task_idx in task_indices:
            n = int(num_datapoints[task_idx])
            if n < 2:
                continue
            
            xi = np.asarray(X[task_idx, :n, :], dtype=np.float32)
            yi = np.asarray(y[task_idx, :n], dtype=np.float32)
            
            if not np.isfinite(xi).all() or not np.isfinite(yi).all():
                continue
            
            # Binary labels stored as floats -> convert to ints
            if yi.ndim > 1:
                yi = yi.squeeze(-1)
            yi = yi.astype(np.int64)
            
            n_ctx = max(1, min(n - 1, int(round(context_ratio * n))))
            perm = rng.permutation(n)
            idx_ctx = perm[:n_ctx]
            idx_tgt = perm[n_ctx:]
            
            X_train = xi[idx_ctx].copy()
            y_train = yi[idx_ctx].copy()
            X_test = xi[idx_tgt].copy()
            y_test = yi[idx_tgt].copy()
            
            datasets.append(
                (
                    f"prior_task_{int(task_idx)}",
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                )
            )
    
    if not datasets:
        raise RuntimeError("Failed to sample any valid tasks from the H5 file.")
    
    return datasets


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        y_proba: Predicted probabilities [N, C]
    
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    
    # Balanced accuracy
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    
    # ROC-AUC (for binary classification)
    if y_proba.shape[1] == 2:
        # Use probability of positive class
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    else:
        # Multi-class: use one-vs-rest
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
    
    return metrics


def plot_roc_curve(
    dataset_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Plot and save ROC curve for binary classification datasets.
    """
    if y_proba.shape[1] != 2:
        print(
            f"Skipping ROC curve for dataset '{dataset_name}': "
            "only binary classification is supported."
        )
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    auc_value = roc_auc_score(y_true, y_proba[:, 1])
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is required to plot ROC curves. "
            "Install it or disable --plot-roc."
        )
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = dataset_name.replace(" ", "_")
    fig_path = output_dir / f"roc_{safe_name}.png"
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved ROC curve to {fig_path}")


def evaluate(
    classifier: NanoTabPFNClassifier,
    datasets: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    verbose: bool = True,
    plot_roc: bool = False,
    roc_dir: Path | None = None,
    export_excel_path: Path | None = None,
) -> Dict[str, float]:
    """
    Evaluate classifier on multiple datasets.
    
    Args:
        classifier: NanoTabPFNClassifier instance
        datasets: List of (name, X_train, X_test, y_train, y_test) tuples
        verbose: Whether to print per-dataset results
    
    Returns:
        Dictionary of averaged metrics
    """
    all_metrics = []
    writer = None
    if export_excel_path is not None:
        export_excel_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pd.ExcelWriter(export_excel_path)
    
    if verbose:
        print("\nEvaluation Results:")
        print("=" * 80)
    
    for name, X_train, X_test, y_train, y_test in datasets:
        # Fit and predict
        classifier.fit(X_train, y_train)
        desc = f"{name} [{classifier.sampling_method.upper()}]"
        context_count = len(X_train)
        target_count = len(X_test)
        y_logits = classifier.predict_logits(X_test, desc=desc)
        y_proba = classifier.logits_to_proba(y_logits)
        y_pred = y_proba.argmax(axis=1)

        if writer is not None:
            df = pd.DataFrame(
                {
                    "ground_truth_label": y_test,
                    "pred_logits": y_logits.tolist(),
                    "pred_proba": list(y_proba),
                    "pred_label": y_pred,
                }
            )
            df.to_excel(writer, sheet_name=name[:31], index=False)
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_proba)
        all_metrics.append(metrics)
        
        if plot_roc:
            if roc_dir is None:
                raise ValueError("roc_dir must be provided when plot_roc=True")
            plot_roc_curve(name, y_test, y_proba, roc_dir)
        
        if verbose:
            print(f"Dataset: {name:20s} | ", end="")
            print(f"ctx={context_count} tgt={target_count} | ", end="")
            print(f"ROC-AUC: {metrics['roc_auc']:.4f} | ", end="")
            print(f"Acc: {metrics['accuracy']:.4f} | ", end="")
            print(f"Bal-Acc: {metrics['balanced_accuracy']:.4f}")
    
    # Compute averages
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    if verbose:
        print("-" * 80)
        print(f"{'Average':20s}     | ", end="")
        print(f"ROC-AUC: {avg_metrics['roc_auc']:.4f} | ", end="")
        print(f"Acc: {avg_metrics['accuracy']:.4f} | ", end="")
        print(f"Bal-Acc: {avg_metrics['balanced_accuracy']:.4f}")
        print("=" * 80)
    
    if writer is not None:
        writer.close()

    return avg_metrics


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        (model, process, config_dict)
    """
    from train import NanoTabPFNModel
    from ndp_discrete.process import D3PM, D3PMSchedule
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config_dict = checkpoint["config"]
    
    # Reconstruct model
    model = NanoTabPFNModel(
        num_features=config_dict["model"]["num_features"],
        num_outputs=config_dict["model"]["num_outputs"],
        embedding_size=config_dict["model"]["embedding_size"],
        num_attention_heads=config_dict["model"]["num_attention_heads"],
        num_layers=config_dict["model"]["num_layers"],
    )
    state_dict = checkpoint["state_dict"]
    # Older checkpoints stored feature normalization buffers; drop them if present.
    state_dict.pop("feature_mean", None)
    state_dict.pop("feature_std", None)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Reconstruct diffusion process (uniform replacement)
    schedule = D3PMSchedule.make_uniform(
        T=config_dict["diffusion"]["timesteps"],
        vocab_size=config_dict["model"]["num_outputs"],
        beta_start=config_dict["diffusion"]["beta_start"],
        beta_end=config_dict["diffusion"]["beta_end"],
        device=device,
        dtype=torch.float32,
    )
    process = D3PM(model, schedule)
    
    return model, process, config_dict


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate nanoTabPFN classifier")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="ddim",
        choices=["ddpm", "ddim"],
        help="Sampling method for inference",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=500,
        help="Number of sampling steps (for DDIM)",
    )
    parser.add_argument(
        "--plot-roc",
        action="store_true",
        help="Plot and save ROC curves for supported datasets",
    )
    parser.add_argument(
        "--export-excel",
        type=str,
        default=None,
        help="Optional path to save per-dataset predictions (ground truth labels, raw outputs, and probabilities) as an Excel file.",
    )
    parser.add_argument(
        "--roc-dir",
        type=str,
        default=None,
        help="Directory to store ROC curve images (default: <checkpoint_dir>/roc_curves)",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display tqdm progress bars during sampling",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="?",
        default="breast_cancer",
        const="",
        help=(
            "Comma-separated list of evaluation datasets. "
            "Options: breast_cancer, diabetes, blood_transfusion, "
            "amazon_employee_access, all. Pass --datasets (with no value) "
            "to disable benchmarks."
        ),
    )
    parser.add_argument(
        "--prior-h5",
        type=str,
        default=None,
        help="Path to the training H5 file for meta-evaluation",
    )
    parser.add_argument(
        "--prior-num-tasks",
        type=int,
        default=5,
        help="Number of H5 tasks to sample when --prior-h5 is set",
    )
    parser.add_argument(
        "--prior-context-ratio",
        type=float,
        default=0.25,
        help="Context ratio to split H5 tasks into train/test sets",
    )
    parser.add_argument(
        "--prior-seed",
        type=int,
        default=0,
        help="Random seed for H5 task sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    
    roc_dir = None
    if args.plot_roc:
        roc_dir = Path(args.roc_dir) if args.roc_dir else checkpoint_path.parent / "roc_curves"
        roc_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    model, process, config_dict = load_checkpoint(checkpoint_path, device)
    
    print(f"Sampling method: {args.sampling_method}")
    if args.sampling_method == "ddim":
        print(f"Number of sampling steps: {args.num_sampling_steps}")
    
    # Create classifier
    classifier = NanoTabPFNClassifier(
        model=model,
        process=process,
        device=device,
        num_classes=config_dict["model"]["num_outputs"],
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        show_progress=args.show_progress,
    )
    
    # Load datasets
    print("\nLoading benchmark datasets...")
    requested_datasets = [
        name.strip()
        for name in (args.datasets or "").split(",")
        if name.strip()
    ]
    datasets: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    if requested_datasets or args.datasets:
        datasets.extend(load_benchmark_datasets(requested_datasets or None))
    
    if args.prior_h5:
        prior_sets = sample_prior_tasks(
            Path(args.prior_h5),
            num_tasks=args.prior_num_tasks,
            context_ratio=args.prior_context_ratio,
            seed=args.prior_seed,
        )
        datasets.extend(prior_sets)
    
    if not datasets:
        raise ValueError(
            "No datasets selected. Specify --datasets, --prior-h5, or both."
        )
    
    dataset_list_str = ", ".join(name for name, *_ in datasets)
    print(f"Loaded {len(datasets)} dataset(s): {dataset_list_str}")
    
    # Run evaluation
    export_excel_path = Path(args.export_excel) if args.export_excel else None
    if export_excel_path is not None:
        export_excel_path.parent.mkdir(parents=True, exist_ok=True)

    avg_metrics = evaluate(
        classifier,
        datasets,
        verbose=True,
        plot_roc=args.plot_roc,
        roc_dir=roc_dir,
        export_excel_path=export_excel_path,
    )
    
    return avg_metrics


if __name__ == "__main__":
    main()
