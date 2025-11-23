from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable, Tuple

import h5py
import numpy as np
import torch

from neural_diffusion_processes.types import Batch


def find_corrupted_tasks(h5_path: Path, max_tasks: int | None = None) -> set:
    """
    Scan the H5 file to identify tasks with non-finite values.
    
    Returns:
      set of task indices that contain NaN or Inf values
    """
    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"Could not find H5 file at {h5_path}")
    
    corrupted = set()
    
    with h5py.File(h5_path, "r") as f:
        X = f["X"]
        y = f["y"]
        num_datapoints = f["num_datapoints"]
        
        num_tasks = X.shape[0]
        if max_tasks is not None:
            num_tasks = min(num_tasks, int(max_tasks))
        
        for i in range(num_tasks):
            n = int(num_datapoints[i])
            if n <= 0:
                continue
            
            xi = np.asarray(X[i, :n, :], dtype=np.float64)
            yi = np.asarray(y[i, :n], dtype=np.float64)
            
            if not np.isfinite(xi).all() or not np.isfinite(yi).all():
                corrupted.add(i)
    
    return corrupted


def compute_feature_normalization(
    h5_path: Path, *, max_tasks: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature (z-score) normalization statistics over the prior dump.

    Returns:
      mean: [F]
      std:  [F]  (with small lower bound to avoid division by zero)
    """
    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"Could not find H5 file at {h5_path}")

    with h5py.File(h5_path, "r") as f:
        X = f["X"]  # [num_tasks, num_points, num_features]
        num_datapoints = f["num_datapoints"]  # [num_tasks]

        num_tasks = X.shape[0]
        if max_tasks is not None:
            num_tasks = min(num_tasks, int(max_tasks))

        sum_x: np.ndarray | None = None
        sum_x2: np.ndarray | None = None
        count: int = 0

        for i in range(num_tasks):
            n = int(num_datapoints[i])
            if n <= 0:
                continue
            # Read this task's datapoints; no need to keep them after accumulating.
            xi = np.asarray(X[i, :n, :], dtype=np.float64)  # [n, F]
            
            # Check for invalid values
            if not np.isfinite(xi).all():
                print(f"Warning: Task {i} contains non-finite values, skipping...")
                continue
            
            if sum_x is None:
                F = xi.shape[-1]
                sum_x = np.zeros(F, dtype=np.float64)
                sum_x2 = np.zeros(F, dtype=np.float64)

            # Use Welford's online algorithm for numerical stability
            # Instead of accumulating sum and sum of squares, use online mean/variance
            sum_x += xi.sum(axis=0)
            sum_x2 += np.square(xi).sum(axis=0)
            count += n
            
            # Check for overflow periodically
            if i % 10000 == 0 and i > 0:
                if not np.isfinite(sum_x).all() or not np.isfinite(sum_x2).all():
                    print(f"Warning: Overflow detected at task {i}")
                    print(f"  sum_x finite: {np.isfinite(sum_x).all()}")
                    print(f"  sum_x2 finite: {np.isfinite(sum_x2).all()}")
                    break

        if count == 0 or sum_x is None or sum_x2 is None:
            raise RuntimeError("No datapoints found when computing normalization statistics.")

        # Check for overflow before computing statistics
        if not np.isfinite(sum_x).all():
            print(f"ERROR: sum_x contains non-finite values!")
            print(f"  sum_x: {sum_x}")
            raise RuntimeError("sum_x contains non-finite values")
        
        if not np.isfinite(sum_x2).all():
            print(f"ERROR: sum_x2 contains non-finite values!")
            print(f"  sum_x2: {sum_x2}")
            raise RuntimeError("sum_x2 contains non-finite values")

        mean = sum_x / float(count)
        var = sum_x2 / float(count) - mean**2
        
        # Ensure variance is non-negative (numerical errors can make it slightly negative)
        var = np.maximum(var, 0.0)
        var = np.maximum(var, 1e-8)
        std = np.sqrt(var)
        
        # Final check
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            print(f"ERROR: Computed statistics are non-finite!")
            print(f"  mean: {mean}")
            print(f"  std: {std}")
            print(f"  count: {count}")
            raise RuntimeError("Computed normalization statistics are non-finite")

    return mean.astype(np.float32), std.astype(np.float32)


class PriorDumpDataLoader(Iterable[Batch]):
    """
    Lightweight data loader for the nanoTabPFN prior dump.

    For each step it samples `batch_size` tasks from the H5 file and builds
    context / target splits using the provided `single_eval_pos`.
    """

    def __init__(
        self,
        h5_path: Path,
        *,
        num_steps: int,
        batch_size: int,
        p_ctx: float = 0.25,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.num_steps = int(num_steps)
        self.batch_size = int(batch_size)
        self.p_ctx = float(p_ctx)
        self.device = device
        self.seed = int(seed)

    def __iter__(self) -> Generator[Batch, None, None]:
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"Could not find H5 file at {self.h5_path}")

        rng = np.random.RandomState(self.seed)

        with h5py.File(self.h5_path, "r") as f:
            X = f["X"]  # [num_tasks, num_points, num_features]
            y = f["y"]  # [num_tasks, num_points]
            num_datapoints = f["num_datapoints"]  # [num_tasks]

            num_tasks, n_points, num_features = X.shape

            # Use a fixed context ratio across tasks.
            n = int(num_datapoints[0])
            n_ctx = max(1, min(n - 1, int(round(self.p_ctx * n))))
            n_tgt = n - n_ctx

            for _ in range(self.num_steps):
                # Sample tasks with replacement, resampling if we hit corrupted tasks
                max_attempts = 100
                batch_valid = False
                
                for attempt in range(max_attempts):
                    task_indices = rng.randint(0, num_tasks, size=self.batch_size)

                    x_context = np.zeros(
                        (self.batch_size, n_ctx, num_features), dtype=np.float32
                    )
                    y_context = np.zeros((self.batch_size, n_ctx, 1), dtype=np.float32)

                    x_target = np.zeros(
                        (self.batch_size, n_tgt, num_features), dtype=np.float32
                    )
                    y_target = np.zeros((self.batch_size, n_tgt, 1), dtype=np.float32)

                    all_finite = True
                    for b, task_idx in enumerate(task_indices):
                        n_i = int(num_datapoints[task_idx])
                        xi = np.asarray(X[task_idx, :n_i, :], dtype=np.float32)  # [n_i, F]
                        yi = np.asarray(y[task_idx, :n_i], dtype=np.float32)     # [n_i]
                        
                        # Check for non-finite values and resample if found
                        if not np.isfinite(xi).all() or not np.isfinite(yi).all():
                            all_finite = False
                            break
                        
                        yi = yi.reshape(n_i, 1)  # [n_i, 1]

                        # Random context/target split (ignore single_eval_pos).
                        perm = rng.permutation(n_i)
                        idx_ctx = perm[:n_ctx]
                        idx_tgt = perm[n_ctx:n_ctx + n_tgt]

                        x_context[b, :, :] = xi[idx_ctx, :]
                        y_context[b, :, :] = yi[idx_ctx, :]

                        x_target[b, :, :] = xi[idx_tgt, :]
                        y_target[b, :, :] = yi[idx_tgt, :]
                    
                    if all_finite:
                        batch_valid = True
                        break
                
                if not batch_valid:
                    raise RuntimeError(
                        f"Could not sample a valid batch after {max_attempts} attempts. "
                        "Too many corrupted tasks in the dataset."
                    )

                # Convert to tensors (optionally directly on device).
                device = self.device if self.device is not None else torch.device("cpu")

                yield Batch(
                    x_target=torch.from_numpy(x_target).to(device),
                    y_target=torch.from_numpy(y_target).to(device),
                    x_context=torch.from_numpy(x_context).to(device),
                    y_context=torch.from_numpy(y_context).to(device),
                    mask_target=None,
                    mask_context=None,
                )
