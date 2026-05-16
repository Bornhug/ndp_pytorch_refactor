"""Utility helpers for TabICL-regression NDP training/evaluation."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch


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


def normalize_y(
    y_context: torch.Tensor,
    y_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-score normalize y using context statistics.

    Args:
        y_context: [B, N_C] or [B, N_C, 1]
        y_target:  [B, N_T] or [B, N_T, 1]

    Returns:
        y_context_norm, y_target_norm, mean, std  (all broadcastable)
    """
    # Work in at least 3-D for consistent broadcasting
    if y_context.ndim == 2:
        y_context = y_context.unsqueeze(-1)
    if y_target.ndim == 2:
        y_target = y_target.unsqueeze(-1)

    mean = y_context.mean(dim=1, keepdim=True)  # [B, 1, 1]
    std = y_context.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)  # [B, 1, 1]

    y_context_norm = (y_context - mean) / std
    y_target_norm = (y_target - mean) / std
    return y_context_norm, y_target_norm, mean, std


def denormalize_y(
    y_pred: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Reverse z-score normalization."""
    return y_pred * std + mean
