from __future__ import annotations

import math
from pathlib import Path
from typing import Generator, Iterable, Tuple

import h5py
import numpy as np
import torch

from ndp_discrete.types import Batch


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
        skip_tasks: Iterable[int] | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.num_steps = int(num_steps)
        self.batch_size = int(batch_size)
        self.p_ctx = float(p_ctx)
        self.device = device
        self.seed = int(seed)
        self._skip_tasks = set(skip_tasks or [])

        self._pointer: int = 0
        self._rng = np.random.RandomState(self.seed)
        self._num_tasks: int | None = None
        self._num_features: int | None = None
        self._points_per_task: int | None = None
        self._tasks_emitted: int = 0
        self._dynamic_bad_tasks: set[int] = set()

    def _ensure_metadata(self) -> None:
        if self._num_tasks is not None:
            return
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"Could not find H5 file at {self.h5_path}")
        with h5py.File(self.h5_path, "r") as f:
            X = f["X"]
            num_datapoints = f["num_datapoints"]
            self._num_tasks = X.shape[0]
            self._num_features = X.shape[-1]
            self._points_per_task = int(num_datapoints[0])

    @property
    def num_tasks(self) -> int:
        self._ensure_metadata()
        return self._num_tasks or 0

    def _valid_task_count(self) -> int:
        return max(0, self.num_tasks - len(self._skip_tasks) - len(self._dynamic_bad_tasks))

    @property
    def steps_per_epoch(self) -> int:
        valid = self._valid_task_count()
        if valid == 0:
            return 0
        return math.ceil(valid / self.batch_size)

    def _should_skip(self, idx: int) -> bool:
        return idx in self._skip_tasks or idx in self._dynamic_bad_tasks

    def __iter__(self) -> Generator[Batch, None, None]:
        self._ensure_metadata()
        rng = self._rng

        num_tasks = self.num_tasks
        num_features = self._num_features or 0
        n_points = self._points_per_task or 0

        valid_total = self._valid_task_count()
        if num_tasks == 0 or valid_total == 0:
            raise RuntimeError("Prior H5 file does not contain any valid tasks.")

        # Use a fixed context ratio across tasks.
        n_ctx = max(1, min(n_points - 1, int(round(self.p_ctx * n_points))))
        n_tgt = n_points - n_ctx

        device = self.device if self.device is not None else torch.device("cpu")

        with h5py.File(self.h5_path, "r") as f:
            X = f["X"]  # [num_tasks, num_points, num_features]
            y = f["y"]  # [num_tasks, num_points]
            num_datapoints = f["num_datapoints"]  # [num_tasks]

            for _ in range(self.num_steps):
                emitted_so_far = self._tasks_emitted % valid_total
                valid_remaining = valid_total - emitted_so_far
                if valid_remaining == 0:
                    valid_remaining = valid_total
                current_batch_size = min(self.batch_size, valid_remaining)

                x_context = np.zeros(
                    (current_batch_size, n_ctx, num_features), dtype=np.float32
                )
                y_context = np.zeros((current_batch_size, n_ctx, 1), dtype=np.float32)

                x_target = np.zeros(
                    (current_batch_size, n_tgt, num_features), dtype=np.float32
                )
                y_target = np.zeros((current_batch_size, n_tgt, 1), dtype=np.float32)

                b = 0
                attempts = 0
                max_attempts = num_tasks * 2
                while b < current_batch_size:
                    if attempts >= max_attempts:
                        raise RuntimeError(
                            "Could not assemble a valid batch; too many corrupted tasks."
                        )
                    if self._pointer >= num_tasks:
                        print("Finished iteration over all stored datasets!")
                        self._pointer = 0
                    task_idx = self._pointer
                    self._pointer += 1
                    attempts += 1

                    if self._should_skip(task_idx):
                        continue

                    n_i = int(num_datapoints[task_idx])
                    xi = np.asarray(X[task_idx, :n_i, :], dtype=np.float32)  # [n_i, F]
                    yi = np.asarray(y[task_idx, :n_i], dtype=np.float32)     # [n_i]

                    if not np.isfinite(xi).all() or not np.isfinite(yi).all():
                        if task_idx not in self._dynamic_bad_tasks:
                            print(f"Skipping corrupted task {task_idx} (non-finite values).")
                            self._dynamic_bad_tasks.add(task_idx)
                            valid_total = self._valid_task_count()
                            if valid_total == 0:
                                raise RuntimeError(
                                    "All tasks are corrupted; cannot continue training."
                                )
                        continue

                    yi = yi.reshape(n_i, 1)  # [n_i, 1]

                    perm = rng.permutation(n_i)
                    idx_ctx = perm[:n_ctx]
                    idx_tgt = perm[n_ctx:n_ctx + n_tgt]

                    x_context[b, :, :] = xi[idx_ctx, :]
                    y_context[b, :, :] = yi[idx_ctx, :]

                    x_target[b, :, :] = xi[idx_tgt, :]
                    y_target[b, :, :] = yi[idx_tgt, :]

                    b += 1
                    self._tasks_emitted += 1

                yield Batch(
                    x_target=torch.from_numpy(x_target).to(device),
                    y_target=torch.from_numpy(y_target).to(device),
                    x_context=torch.from_numpy(x_context).to(device),
                    y_context=torch.from_numpy(y_context).to(device),
                    mask_target=None,
                    mask_context=None,
                )
