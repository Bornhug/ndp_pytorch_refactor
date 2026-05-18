"""Regression model wrappers for TabICL-style neural diffusion processes."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .model import BiDimensionalAttentionModel


def _ensure_y_column(y: torch.Tensor) -> torch.Tensor:
    """Return y as ``[B, N, 1]`` when callers pass ``[B, N]``."""
    if y.ndim == 2:
        return y.unsqueeze(-1)
    return y


def normalize_y(
    y_context: torch.Tensor,
    y_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-score normalize context and target y values using context statistics."""
    y_context = _ensure_y_column(y_context)
    y_target = _ensure_y_column(y_target)

    mean = y_context.mean(dim=1, keepdim=True)
    std = y_context.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)

    y_context_norm = (y_context - mean) / std
    y_target_norm = (y_target - mean) / std
    return y_context_norm, y_target_norm, mean, std


def denormalize_y(
    y_pred: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Reverse context-stat y normalization."""
    return y_pred * std + mean


class NDPRegressor(nn.Module):
    """Regression wrapper around ``BiDimensionalAttentionModel``.

    The wrapper normalizes target and context x-features per task, then delegates
    noised-y denoising to the core bidimensional attention model.
    """

    def __init__(
        self,
        *,
        embedding_size: int,
        num_attention_heads: int,
        num_layers: int,
        num_timesteps: int = 500,
    ) -> None:
        """Create a regression NDP for a fixed diffusion schedule length."""
        super().__init__()
        self.core = BiDimensionalAttentionModel(
            n_layers=num_layers,
            hidden_dim=embedding_size,
            num_heads=num_attention_heads,
            num_timesteps=num_timesteps,
            init_zero=True,
        )

    def _compute_x_stats(
        self,
        x_target: torch.Tensor,
        x_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-task feature normalization stats over context and target."""
        x = x_target if x_context is None else torch.cat([x_context, x_target], dim=1)
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6))
        return mean, std

    def _normalize_x(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Apply task-wise feature z-score normalization."""
        return (x - mean) / std

    def forward(
        self,
        x_target: torch.Tensor,
        y_target: torch.Tensor,
        t: torch.Tensor,
        x_context: torch.Tensor | None = None,
        y_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return predicted diffusion noise with shape ``[B, N_target, 1]``."""
        mean, std = self._compute_x_stats(x_target, x_context)
        x_target = self._normalize_x(x_target, mean, std)
        if x_context is not None:
            x_context = self._normalize_x(x_context, mean, std)

        return self.core(
            x_target=x_target,
            y_target=y_target,
            t=t.to(device=x_target.device, dtype=torch.long),
            x_context=x_context,
            y_context=y_context,
        )


class NDPRegressorWrapper:
    """Sklearn-compatible wrapper for NDP regression inference."""

    def __init__(
        self,
        model,
        process,
        device: torch.device,
        *,
        num_sampling_steps: int = 500,
        sampling_method: str = "ddpm",
        ddim_eta: float = 0.0,
    ) -> None:
        self.model = model
        self.model.eval()
        self.process = process
        self.device = device
        self.num_sampling_steps = int(num_sampling_steps)
        self.sampling_method = str(sampling_method).lower()
        self.ddim_eta = float(ddim_eta)

        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> NDPRegressorWrapper:
        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train.astype(np.float32)
        return self

    def predict_repeated(self, X_test: np.ndarray, num_repeats: int) -> np.ndarray:
        """Return repeated stochastic predictions with one batched sampler call."""
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Must call fit() before predict().")
        if int(num_repeats) <= 0:
            raise ValueError(f"num_repeats must be positive, got {num_repeats}.")

        repeat_count = int(num_repeats)

        x_context = torch.from_numpy(self.X_train).to(self.device).unsqueeze(0)
        y_context_raw = torch.from_numpy(self.y_train).to(self.device).unsqueeze(0)
        y_context, _, mean, std = normalize_y(
            y_context_raw,
            y_context_raw,
        )
        x_target = (
            torch.from_numpy(X_test.astype(np.float32)).to(self.device).unsqueeze(0)
        )
        if repeat_count > 1:
            x_context = x_context.expand(repeat_count, -1, -1)
            y_context = y_context.expand(repeat_count, -1, -1)
            x_target = x_target.expand(repeat_count, -1, -1)

        y_pred_norm = self.process.sample(
            None,
            x_target,
            model=self.model,
            x_context=x_context,
            y_context=y_context,
            output_dim=y_context.shape[-1],
            num_sample_steps=self.num_sampling_steps,
            method=self.sampling_method,
            eta=self.ddim_eta,
        )

        y_pred = denormalize_y(y_pred_norm, mean, std)
        return y_pred.squeeze(-1).detach().cpu().numpy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return one prediction vector for sklearn-style callers."""
        return self.predict_repeated(X_test, 1)[0]


__all__ = [
    "NDPRegressor",
    "NDPRegressorWrapper",
    "denormalize_y",
    "normalize_y",
]
