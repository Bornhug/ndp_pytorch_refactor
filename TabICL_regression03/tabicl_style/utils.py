"""Utility helpers for TabICL-regression NDP training/evaluation."""

from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np
import torch

from neural_diffusion_processes.regressor import denormalize_y, normalize_y


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


def split_context_target(
    X: torch.Tensor, y: torch.Tensor, train_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split (X, y) into context (train) and target (test) along the N dimension."""
    x_context = X[:, :train_size, :]
    y_context = y[:, :train_size]
    x_target = X[:, train_size:, :]
    y_target = y[:, train_size:]
    return x_context, y_context, x_target, y_target
