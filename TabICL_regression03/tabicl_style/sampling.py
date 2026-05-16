"""Conditional Gaussian reverse-diffusion sampling for regression NDP."""

from __future__ import annotations

from typing import Optional

import torch

from neural_diffusion_processes.process import GaussianDiffusion


def _build_reverse_schedule(
    num_timesteps: int,
    num_steps: int | None,
) -> list[int]:
    if num_steps is None or int(num_steps) <= 0 or int(num_steps) >= num_timesteps:
        return list(range(num_timesteps - 1, -1, -1))

    schedule = torch.linspace(num_timesteps - 1, 0, int(num_steps), dtype=torch.long)
    values = sorted({int(v) for v in schedule.tolist()}, reverse=True)
    if values[0] != num_timesteps - 1:
        values.insert(0, num_timesteps - 1)
    if values[-1] != 0:
        values.append(0)
    return values


@torch.no_grad()
def sample_predictions(
    process: GaussianDiffusion,
    model,
    *,
    x_target: torch.Tensor,
    x_context: torch.Tensor | None,
    y_context: torch.Tensor | None,
    mask_target: torch.Tensor | None = None,
    mask_context: torch.Tensor | None = None,
    num_steps: Optional[int] = None,
    sampling_method: str = "ddpm",
    ddim_eta: float = 0.0,
) -> torch.Tensor:
    """Run Gaussian reverse diffusion and return predicted y at t=0."""
    device = x_target.device
    dtype = x_target.dtype
    batch_size, num_target, _ = x_target.shape
    method = str(sampling_method).lower()
    if method not in {"ddpm", "ddim"}:
        raise ValueError(f"Unknown sampling_method: {sampling_method}")

    output_dim = y_context.shape[-1] if y_context is not None else 1
    y = torch.randn(
        batch_size,
        num_target,
        output_dim,
        device=device,
        dtype=dtype,
    )

    schedule = _build_reverse_schedule(len(process.betas), num_steps)
    for step_idx, t_int in enumerate(schedule):
        t_batch = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
        eps_hat = model(
            x_target=x_target,
            y_target=y,
            t=t_batch,
            mask_target=mask_target,
            x_context=x_context,
            y_context=y_context,
            mask_context=mask_context,
        )
        next_t = schedule[step_idx + 1] if step_idx + 1 < len(schedule) else -1
        y = process.reverse_step(
            None,
            eps_hat,
            y,
            t_int,
            method=method,
            next_t=next_t,
            eta=float(ddim_eta),
        )

    return y
