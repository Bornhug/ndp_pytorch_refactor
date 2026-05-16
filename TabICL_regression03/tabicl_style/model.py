"""NDP model wrapper for TabICL-regression training.

Wraps the BiDimensionalAttentionModel from neural_diffusion_processes/model.py
for continuous-valued regression.  The core model takes [x_feature, y_value] as
a 2-D input per feature slot and predicts 1-D noise.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from neural_diffusion_processes.model import BiDimensionalAttentionModel


class NDPRegressor(nn.Module):
    def __init__(
        self,
        *,
        embedding_size: int,
        num_attention_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.core = BiDimensionalAttentionModel(
            n_layers=num_layers,
            hidden_dim=embedding_size,
            num_heads=num_attention_heads,
            init_zero=True,
        )

    # ------------------------------------------------------------------
    # Feature normalization (z-score across context + target x-features)
    # ------------------------------------------------------------------

    def _stack_batch(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        if x.ndim == 2:
            return x.unsqueeze(0)
        return x

    def _compute_stats(
        self,
        x_target: torch.Tensor | None,
        x_context: torch.Tensor | None,
        mask_target: torch.Tensor | None,
        mask_context: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[None, None]:
        if x_target is None and x_context is None:
            return None, None

        def _sum_and_count(x, mask):
            if x is None:
                return None, None, None
            if mask is None:
                active = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
            else:
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0)
                if mask.ndim == 2 and mask.size(0) == 1 and x.size(0) > 1:
                    mask = mask.expand(x.size(0), -1)
                active = (1.0 - mask).to(dtype=x.dtype)
            active = active.unsqueeze(-1)  # [B, N, 1]
            sum_x = (x * active).sum(dim=1)
            sum_x2 = (x.pow(2) * active).sum(dim=1)
            count = active.sum(dim=1).clamp_min(1.0)
            return sum_x, sum_x2, count

        sum_tgt, sumsq_tgt, count_tgt = _sum_and_count(x_target, mask_target)
        sum_ctx, sumsq_ctx, count_ctx = _sum_and_count(x_context, mask_context)

        if sum_tgt is None:
            sum_x, sum_x2, count = sum_ctx, sumsq_ctx, count_ctx
        elif sum_ctx is None:
            sum_x, sum_x2, count = sum_tgt, sumsq_tgt, count_tgt
        else:
            sum_x = sum_tgt + sum_ctx
            sum_x2 = sumsq_tgt + sumsq_ctx
            count = count_tgt + count_ctx

        mean = sum_x / count
        var = sum_x2 / count - mean.pow(2)
        std = torch.sqrt(var.clamp_min(1e-6))
        return mean[:, None, :], std[:, None, :]

    def _normalize(self, x, mean, std):
        if x is None or mean is None or std is None:
            return x
        return (x - mean) / std

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_target: torch.Tensor,  # [B, N_T, D]
        y_target: torch.Tensor,  # [B, N_T, 1]  (noised y)
        t: torch.Tensor,  # [B]
        mask_target: torch.Tensor | None = None,
        x_context: torch.Tensor | None = None,  # [B, N_C, D]
        y_context: torch.Tensor | None = None,  # [B, N_C, 1]
        mask_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns predicted noise [B, N_T, 1]."""
        added_batch_dim = False
        if x_target.ndim == 2:
            x_target = x_target.unsqueeze(0)
            added_batch_dim = True
        if y_target.ndim == 2:
            y_target = y_target.unsqueeze(0)
        if t.ndim == 0:
            t = t.unsqueeze(0)

        x_context = self._stack_batch(x_context)
        y_context = self._stack_batch(y_context)

        # Normalize x-features
        mean, std = self._compute_stats(
            x_target, x_context, mask_target, mask_context
        )
        x_target = self._normalize(x_target, mean, std)
        x_context = (
            self._normalize(x_context, mean, std) if x_context is not None else None
        )

        out = self.core(
            x_tgt=x_target,
            y_tgt=y_target,
            t=t.to(dtype=x_target.dtype, device=x_target.device),
            mask_tgt=mask_target,
            x_context=x_context,
            y_context=y_context,
            mask_context=mask_context,
        )  # [B, N_T, 1]

        if added_batch_dim:
            out = out.squeeze(0)
        return out
