"""
Inference utilities for diffusion-based classification.
Handles reverse diffusion sampling (DDPM and DDIM) for prediction.
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Tuple

from neural_diffusion_processes.process import GaussianDiffusion


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
) -> torch.Tensor:
    """
    Full DDPM sampling: run complete reverse diffusion process.
    
    Args:
        model: The trained NanoTabPFNModel
        process: GaussianDiffusion instance
        x_tgt: Test features [N_test, D]
        x_context: Train features [N_train, D]
        y_context: Train labels (one-hot) [N_train, C]
        num_classes: Number of classes
        device: Device to run on
    
    Returns:
        y_pred: Predicted labels (continuous) [N_test, C]
    """
    N_test = x_tgt.shape[0]
    
    # Initialize with random noise
    y_t = torch.randn(N_test, num_classes, device=device, dtype=torch.float32)
    
    # Add batch dimension if needed
    if x_tgt.ndim == 2:
        x_tgt = x_tgt.unsqueeze(0)  # [1, N_test, D]
    if x_context.ndim == 2:
        x_context = x_context.unsqueeze(0)  # [1, N_train, D]
    if y_context.ndim == 2:
        y_context = y_context.unsqueeze(0)  # [1, N_train, C]
    if y_t.ndim == 2:
        y_t = y_t.unsqueeze(0)  # [1, N_test, C]
    
    # Reverse diffusion loop: T-1 -> 0
    num_timesteps = len(process.betas)
    iterator = range(num_timesteps - 1, -1, -1)
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(
            iterator,
            desc=progress_desc or "DDPM sampling",
            total=num_timesteps,
        )
    
    for t_int in iterator:
        t_tensor = torch.tensor(t_int, device=device, dtype=torch.long)
        t_batch = t_tensor.unsqueeze(0)  # [1]
        
        # Predict noise
        with torch.no_grad():
            eps_hat = model(
                x_tgt=x_tgt,
                y_tgt=y_t,
                t=t_batch,
                mask_tgt=None,
                x_context=x_context,
                y_context=y_context,
                mask_context=None,
            )
        
        # Update y_t using DDPM backward step
        y_t = y_t.squeeze(0)  # [N_test, C]
        eps_hat = eps_hat.squeeze(0)  # [N_test, C]
        y_t = process.ddpm_backward_step(None, eps_hat, y_t, t_tensor)
        y_t = y_t.unsqueeze(0)  # [1, N_test, C]
    
    return y_t.squeeze(0)  # [N_test, C]


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
) -> torch.Tensor:
    """
    DDIM sampling: faster deterministic sampling with fewer steps.
    
    Args:
        model: The trained NanoTabPFNModel
        process: GaussianDiffusion instance
        x_tgt: Test features [N_test, D]
        x_context: Train features [N_train, D]
        y_context: Train labels (one-hot) [N_train, C]
        num_classes: Number of classes
        device: Device to run on
        num_steps: Number of sampling steps (< total timesteps)
    
    Returns:
        y_pred: Predicted labels (continuous) [N_test, C]
    """
    N_test = x_tgt.shape[0]
    total_timesteps = len(process.betas)
    
    # Create subsequence of timesteps for DDIM
    step_size = total_timesteps // num_steps
    timesteps = list(range(0, total_timesteps, step_size))
    timesteps = timesteps[:num_steps]
    timesteps.reverse()  # T-1, ..., 0
    
    # Initialize with random noise
    y_t = torch.randn(N_test, num_classes, device=device, dtype=torch.float32)
    
    # Add batch dimension if needed
    if x_tgt.ndim == 2:
        x_tgt = x_tgt.unsqueeze(0)
    if x_context.ndim == 2:
        x_context = x_context.unsqueeze(0)
    if y_context.ndim == 2:
        y_context = y_context.unsqueeze(0)
    if y_t.ndim == 2:
        y_t = y_t.unsqueeze(0)
    
    indices = range(len(timesteps))
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(
            indices,
            desc=progress_desc or "DDIM sampling",
            total=len(timesteps),
        )
    
    for i in indices:
        t_int = timesteps[i]
        t_tensor = torch.tensor(t_int, device=device, dtype=torch.long)
        t_batch = t_tensor.unsqueeze(0)
        
        # Predict noise
        with torch.no_grad():
            eps_hat = model(
                x_tgt=x_tgt,
                y_tgt=y_t,
                t=t_batch,
                mask_tgt=None,
                x_context=x_context,
                y_context=y_context,
                mask_context=None,
            )
        
        # DDIM update (deterministic)
        alpha_bar_t = process.alpha_bars[t_int]
        
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_bar_prev = process.alpha_bars[t_prev]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)
        
        # Predict x_0 from x_t and eps
        y_t_flat = y_t.squeeze(0)
        eps_hat_flat = eps_hat.squeeze(0)
        
        y_0_pred = (y_t_flat - torch.sqrt(1 - alpha_bar_t) * eps_hat_flat) / torch.sqrt(alpha_bar_t)
        
        # Compute y_{t-1}
        y_t = (
            torch.sqrt(alpha_bar_prev) * y_0_pred +
            torch.sqrt(1 - alpha_bar_prev) * eps_hat_flat
        )
        y_t = y_t.unsqueeze(0)
    
    return y_t.squeeze(0)  # [N_test, C]
