"""
Inference utilities for diffusion-based classification using the original NDP
RePaint sampler.
"""
from __future__ import annotations

from typing import Optional

import torch

from neural_diffusion_process_original.process import GaussianDiffusion


def _prepare_generator(device: torch.device) -> torch.Generator:
    """Create a per-call generator so sampling is stochastic but reproducible if desired."""
    gen = torch.Generator(device=device)
    seed = torch.randint(0, 2**31 - 1, (1,), device=torch.device("cpu")).item()
    gen.manual_seed(seed)
    return gen


def _ensure_matrix(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [N, D], got shape {tuple(x.shape)}")
    return x


def _conditional_sample(
    model,
    process: GaussianDiffusion,
    x_tgt: torch.Tensor,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    *,
    device: torch.device,
    num_classes: int,
    num_inner_steps: int,
    num_sample_steps: Optional[int],
    progress: bool = False,
    progress_desc: str | None = None,
) -> torch.Tensor:
    x_tgt = _ensure_matrix(x_tgt.to(device=device, dtype=torch.float32), "x_tgt")
    x_context = _ensure_matrix(x_context.to(device=device, dtype=torch.float32), "x_context")
    y_context = _ensure_matrix(y_context.to(device=device, dtype=torch.float32), "y_context")

    if y_context.shape[-1] != num_classes:
        raise ValueError(
            f"Context labels last dim ({y_context.shape[-1]}) != num_classes ({num_classes})"
        )

    generator = _prepare_generator(device)

    return process.conditional_sample(
        key=generator,
        x=x_tgt,
        mask=None,
        x_context=x_context,
        y_context=y_context,
        mask_context=None,
        model_fn=model,
        num_inner_steps=num_inner_steps,
        method="repaint",
        num_sample_steps=num_sample_steps,
        progress=progress,
        progress_desc=progress_desc,
    )


def ddpm_sample(
    model,
    process: GaussianDiffusion,
    x_tgt: torch.Tensor,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    num_classes: int,
    device: torch.device,
    *,
    progress: bool = False,
    progress_desc: str | None = None,
    num_inner_steps: int = 5,
) -> torch.Tensor:
    """
    Run the original DDPM/RePaint sampler (full schedule).
    """
    return _conditional_sample(
        model,
        process,
        x_tgt,
        x_context,
        y_context,
        device=device,
        num_classes=num_classes,
        num_inner_steps=num_inner_steps,
        num_sample_steps=None,
        progress=progress,
        progress_desc=progress_desc,
    )


def ddim_sample(
    model,
    process: GaussianDiffusion,
    x_tgt: torch.Tensor,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    num_classes: int,
    device: torch.device,
    num_steps: int = 50,
    *,
    progress: bool = False,
    progress_desc: str | None = None,
    num_inner_steps: int = 5,
) -> torch.Tensor:
    """
    Deterministic DDIM-style sampling via evenly spaced t-steps.
    """
    return _conditional_sample(
        model,
        process,
        x_tgt,
        x_context,
        y_context,
        device=device,
        num_classes=num_classes,
        num_inner_steps=num_inner_steps,
        num_sample_steps=num_steps,
        progress=progress,
        progress_desc=progress_desc,
    )
