"""
Inference utilities for diffusion-based classification using the original NDP
RePaint sampler.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ndp_discrete.process import conditional_sample, D3PM


def _ensure_matrix(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if x.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [N, D], got shape {tuple(x.shape)}")
    return x


def _to_one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Ensure labels are one-hot encoded (float32).
    """
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    if y.size(-1) == num_classes:
        return y.to(dtype=torch.float32)
    y_idx = y.to(dtype=torch.long).squeeze(-1)
    return F.one_hot(y_idx, num_classes=num_classes).to(dtype=torch.float32)


def _conditional_sample(
    model,
    process: D3PM,
    x_tgt: torch.Tensor,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    *,
    device: torch.device,
    num_classes: int,
    num_sample_steps: Optional[int],
    progress: bool = False,
    progress_desc: str | None = None,
) -> torch.Tensor:
    x_tgt = _ensure_matrix(x_tgt.to(device=device, dtype=torch.float32), "x_tgt")
    x_context = _ensure_matrix(x_context.to(device=device, dtype=torch.float32), "x_context")
    y_context = _ensure_matrix(y_context.to(device=device), "y_context")
    y_context = _to_one_hot(y_context, num_classes)

    # Add batch dimension expected by the model/process.
    x_tgt = x_tgt.unsqueeze(0)
    x_context = x_context.unsqueeze(0)
    y_context = y_context.unsqueeze(0)

    return conditional_sample(
        model,
        process,
        x_target=x_tgt,
        x_context=x_context,
        y_context=y_context,
        mask_target=None,
        mask_context=None,
        num_sample_steps=num_sample_steps,
        progress=progress,
        progress_desc=progress_desc,
    )


def ddpm_sample(
    model,
    process: D3PM,
    x_tgt: torch.Tensor,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    num_classes: int,
    device: torch.device,
    *,
    progress: bool = False,
    progress_desc: str | None = None,
) -> torch.Tensor:
    """
    Run the original DDPM/RePaint sampler (full schedule).
    """
    samples = _conditional_sample(
        model,
        process,
        x_tgt,
        x_context,
        y_context,
        device=device,
        num_classes=num_classes,
        num_sample_steps=None,
        progress=progress,
        progress_desc=progress_desc,
    )
    return samples.squeeze(0)


def ddim_sample(
    model,
    process: D3PM,
    x_tgt: torch.Tensor,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    num_classes: int,
    device: torch.device,
    num_steps: int = 50,
    *,
    progress: bool = False,
    progress_desc: str | None = None,
) -> torch.Tensor:
    """
    Deterministic DDIM-style sampling via evenly spaced t-steps.
    """
    samples = _conditional_sample(
        model,
        process,
        x_tgt,
        x_context,
        y_context,
        device=device,
        num_classes=num_classes,
        num_sample_steps=num_steps,
        progress=progress,
        progress_desc=progress_desc,
    )
    return samples.squeeze(0)
