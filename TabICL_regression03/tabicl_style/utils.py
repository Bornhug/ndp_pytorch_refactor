"""Utility helpers for TabICL-regression NDP training/evaluation."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from neural_diffusion_processes.regressor import denormalize_y, normalize_y

RUN_DIR_RE = re.compile(r"run(\d+)$")
CHECKPOINT_RE = re.compile(r"step-(\d+)\.pt$")
BATCH_RE = re.compile(r"batch_(\d+)\.pt$")


def infer_lr_schedule_steps(config, total_steps: int) -> Tuple[int, int]:
    """Infer warmup and decay step counts from optimizer schedule fractions."""
    if total_steps <= 0:
        return 0, 0

    warmup_fraction = float(config.optimizer.warmup_fraction)
    decay_fraction = float(config.optimizer.decay_fraction)
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError("optimizer.warmup_fraction must be between 0.0 and 1.0.")
    if not 0.0 <= decay_fraction <= 1.0:
        raise ValueError("optimizer.decay_fraction must be between 0.0 and 1.0.")

    warmup_steps = int(round(total_steps * warmup_fraction))
    if warmup_fraction > 0.0:
        warmup_steps = max(1, warmup_steps)
    warmup_steps = min(warmup_steps, total_steps)

    remaining_steps = max(0, total_steps - warmup_steps)
    decay_steps = int(round(total_steps * decay_fraction))
    if decay_fraction > 0.0:
        decay_steps = max(1, decay_steps)
    decay_steps = min(decay_steps, remaining_steps)
    return warmup_steps, decay_steps


def compute_lr(config, step: int, *, total_steps: int) -> float:
    """Infer schedule steps and compute LR for a global train step."""
    if total_steps <= 0:
        return config.optimizer.end_lr

    warmup_steps, decay_steps = infer_lr_schedule_steps(config, total_steps)
    return compute_lr_from_schedule_steps(
        config,
        step,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )


def compute_lr_from_schedule_steps(
    config,
    step: int,
    *,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Compute LR using precomputed warmup and decay step counts."""
    init_lr = config.optimizer.init_lr
    peak_lr = config.optimizer.peak_lr
    end_lr = config.optimizer.end_lr

    if step <= warmup_steps:
        if warmup_steps == 0:
            return peak_lr
        alpha = step / float(warmup_steps)
        return init_lr + (peak_lr - init_lr) * alpha

    if decay_steps <= 0:
        return end_lr

    t = min(step - warmup_steps, decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t / float(decay_steps)))
    return end_lr + (peak_lr - end_lr) * cosine


def set_seed(np_seed: int, torch_seed: int) -> None:
    random.seed(torch_seed)
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def _device_type(device) -> str:
    """Return a normalized PyTorch device type from a string or torch.device."""
    if device is None:
        return ""
    if isinstance(device, torch.device):
        return device.type
    return torch.device(str(device)).type


def resolve_amp_dtype(dtype_name: str, device=None) -> torch.dtype:
    """Map config dtype names, including ``auto``, to PyTorch autocast dtypes."""
    name = str(dtype_name).lower()
    if name == "auto":
        if _device_type(device) == "cuda" and torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float16
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(
        "Unsupported AMP dtype "
        f"{dtype_name!r}; use auto, float16, bfloat16, or float32."
    )


def amp_dtype_name(dtype: torch.dtype) -> str:
    """Return a stable short dtype name for logs and JSON output."""
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype).replace("torch.", "")


def resolve_amp_settings(
    use_amp: bool,
    dtype_name: str,
    *,
    device=None,
) -> tuple[bool, torch.dtype]:
    """Resolve whether AMP should run and which autocast dtype to use."""
    amp_dtype = resolve_amp_dtype(dtype_name, device=device)
    amp_enabled = (
        bool(use_amp)
        and _device_type(device) == "cuda"
        and amp_dtype != torch.float32
    )
    return amp_enabled, amp_dtype


def split_context_target(
    X: torch.Tensor, y: torch.Tensor, train_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split (X, y) into context (train) and target (test) along the N dimension."""
    x_context = X[:, :train_size, :]
    y_context = y[:, :train_size]
    x_target = X[:, train_size:, :]
    y_target = y[:, train_size:]
    return x_context, y_context, x_target, y_target


def infer_next_checkpoint_dir(runs_root: str | Path) -> Path:
    """Return the next unused ``runXX`` directory below ``runs_root``.

    Existing folders named like ``run01`` or ``run12`` are scanned, and the next
    numeric suffix is selected. Non-matching directories are ignored.
    """
    root = Path(runs_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    used_indices = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = RUN_DIR_RE.fullmatch(path.name)
        if match:
            used_indices.append(int(match.group(1)))

    next_index = max(used_indices, default=0) + 1
    return root / f"run{next_index:02d}"


def infer_latest_run_dir(runs_root: str | Path) -> Path | None:
    """Return the highest-numbered existing ``runXX`` directory, if any."""
    root = Path(runs_root).expanduser()
    if not root.is_dir():
        return None

    numbered_runs = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = RUN_DIR_RE.fullmatch(path.name)
        if match:
            numbered_runs.append((int(match.group(1)), path))

    if not numbered_runs:
        return None
    return max(numbered_runs, key=lambda item: item[0])[1]


def checkpoint_step(path: str | Path) -> int:
    """Extract the numeric training step from a ``step-*.pt`` checkpoint."""
    match = CHECKPOINT_RE.fullmatch(Path(path).name)
    if not match:
        return -1
    return int(match.group(1))


def get_latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    """Find the highest-step checkpoint file in a checkpoint directory."""
    directory = Path(ckpt_dir)
    if not directory.is_dir():
        return None
    checkpoints = [
        path for path in directory.glob("step-*.pt") if checkpoint_step(path) >= 0
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_step)


def checkpoint_curr_step(path: str | Path) -> int:
    """Read ``curr_step`` from a checkpoint, falling back to the filename step."""
    fallback = checkpoint_step(path)
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        return int(checkpoint.get("curr_step", fallback))
    return fallback


def prior_batch_index(path: Path) -> int | None:
    """Return the numeric index from ``batch_*.pt`` or ``None`` if unmatched."""
    match = BATCH_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))


def prior_start_for_step(
    load_prior_start: int, curr_step: int, steps_per_epoch: int
) -> int:
    """Return the first prior batch index needed for the current training step."""
    if steps_per_epoch <= 0:
        return int(load_prior_start)
    return int(load_prior_start) + (int(curr_step) % int(steps_per_epoch))


def infer_steps_per_epoch(config) -> int:
    """Infer epoch length from pre-generated prior batch files.

    Training is pre-generated-prior-only: every ``batch_*.pt`` file in
    ``training.prior_dir`` at or after ``load_prior_start`` counts as one
    optimizer step.
    """
    prior_dir = config.training.prior_dir
    if prior_dir is None:
        raise ValueError(
            "training.prior_dir is required; live TabICL prior generation is "
            "not supported in TabICL_regression03 training."
        )

    p = Path(prior_dir).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(f"prior_dir not found: {p}")

    start = int(config.training.load_prior_start)
    count = 0
    for path in p.glob("batch_*.pt"):
        batch_index = prior_batch_index(path)
        if batch_index is not None and batch_index >= start:
            count += 1
    if count <= 0:
        raise FileNotFoundError(
            f"no pre-generated batch_*.pt files found in {p} "
            f"at or after index {start}"
        )
    return count
