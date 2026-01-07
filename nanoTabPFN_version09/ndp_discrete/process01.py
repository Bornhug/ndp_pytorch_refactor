
"""
Discrete diffusion utilities for categorical labels (D3PM-style).

This replaces the Gaussian diffusion helper with a uniform-replacement
discrete process tailored to the NanoTabPFN setting:
  - labels are categorical; we diffuse over class tokens
  - the model takes (y_t, x, t, mask) and outputs logits for y_{t-1}
  - masks mark inactive points (1 = masked); they are excluded from loss
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Time embedding (sin/cos)
# -----------------------------
def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal embedding for integer timesteps.

    Args:
        t: Tensor [B] with integer timesteps in [1..T].
        dim: Embedding dimension.
    Returns:
        Tensor [B, dim].
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, device=device).float() / max(half - 1, 1)
    )
    tt = t.float().unsqueeze(1)  # [B,1]
    args = tt * freqs.unsqueeze(0)  # [B,half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# -----------------------------
# D3PM forward kernel
# -----------------------------
@dataclass
class D3PMSchedule:
    """
    Precomputed forward transition matrices Q_t and their cumulative products.
    """

    Q: torch.Tensor        # [T, V, V]
    Q_bar: torch.Tensor    # [T, V, V]

    @staticmethod
    def make_uniform(
        T: int,
        vocab_size: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.05,
        uniform_including_self: bool = True,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "D3PMSchedule":
        """
        Uniform replacement corruption:
          Q_t = (1-β_t) I + β_t U
        where U is uniform over all tokens (including self) or over "other tokens only".
        """
        device = torch.device(device)
        V = vocab_size
        betas = torch.linspace(beta_start, beta_end, T, device=device, dtype=dtype).clamp(0.0, 1.0)

        I = torch.eye(V, device=device, dtype=dtype)
        if uniform_including_self:
            U = torch.full((V, V), 1.0 / V, device=device, dtype=dtype)
        else:
            U = (torch.ones((V, V), device=device, dtype=dtype) - I) / (V - 1)

        Q = torch.empty((T, V, V), device=device, dtype=dtype)
        for t in range(T):
            beta = betas[t]
            Q[t] = (1.0 - beta) * I + beta * U

        Q_bar = torch.empty_like(Q)
        Q_bar[0] = Q[0]
        for t in range(1, T):
            Q_bar[t] = Q_bar[t - 1] @ Q[t]

        return D3PMSchedule(Q=Q, Q_bar=Q_bar)


# -----------------------------
# D3PM core
# -----------------------------
class D3PM(nn.Module):
    """
    Discrete diffusion with exact forward/posterior kernels and a categorical denoiser.
    The denoiser must accept (t, y_t_onehot, x, mask) and emit logits over classes.
    """

    def __init__(self, denoiser: nn.Module, schedule: D3PMSchedule) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.schedule = schedule

    @property
    def T(self) -> int:
        return self.schedule.Q.shape[0]

    @property
    def V(self) -> int:
        return self.schedule.Q.shape[1]

    def q_probs_xt_given_x0(self, y0_tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Return q(y_t | y_0) as probabilities.

        Args:
            y0_tokens: Tensor [B, L] of clean label tokens.
            t: Tensor [B] of timesteps in [1..T].
        Returns:
            Tensor [B, L, V] with categorical probabilities.
        """
        onehot = F.one_hot(y0_tokens, num_classes=self.V).to(self.schedule.Q.dtype)  # [B,L,V]
        Qbar_t = self.schedule.Q_bar[t - 1]  # [B,V,V] via advanced indexing
        probs = torch.einsum("blv,bvw->blw", onehot, Qbar_t)
        return probs.clamp_min(0.0)

    @torch.no_grad()
    def q_sample(
        self,
        y0_tokens: torch.Tensor,
        t: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample y_t ~ q(y_t | y_0).
        """
        probs = self.q_probs_xt_given_x0(y0_tokens, t)  # [B,L,V]
        B, L, V = probs.shape
        yt = torch.multinomial(probs.reshape(-1, V), 1, generator=generator).reshape(B, L)
        return yt

    def q_posterior_xtm1_given_xt_x0(
        self,
        y0_tokens: torch.Tensor,
        yt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute q(y_{t-1} | y_t, y_0) exactly as a categorical distribution over classes.
        """
        dtype = self.schedule.Q.dtype
        onehot0 = F.one_hot(y0_tokens, num_classes=self.V).to(dtype)  # [B,L,V]
        Qbar_prev = self.schedule.Q_bar[t - 2]  # [B,V,V]
        probs_prev = torch.einsum("blv,bvw->blw", onehot0, Qbar_prev)  # [B,L,V]

        Qt = self.schedule.Q[t - 1]  # [B,V,V]
        Qt_T = Qt.transpose(1, 2)
        trans = Qt_T.gather(dim=1, index=yt.unsqueeze(-1).expand(-1, -1, self.V))  # [B,L,V]

        unnorm = probs_prev * trans
        norm = unnorm.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return (unnorm / norm).clamp_min(0.0)

    def loss(
        self,
        y_tokens: torch.Tensor,
        x_feats: torch.Tensor,
        mask: torch.Tensor | None = None,
        t: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        D3PM objective with reconstruction at t=1 and KL for t>1.
        """
        device = y_tokens.device
        B, L = y_tokens.shape
        T = self.T

        if t is None:
            t = torch.randint(1, T + 1, (B,), device=device, generator=generator)

        if mask is None:
            mask = torch.zeros((B, L), device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.float32)
        active = 1.0 - mask  # 1 for valid points

        yt = self.q_sample(y_tokens, t, generator=generator)  # [B,L]
        y_onehot = F.one_hot(yt, num_classes=self.V).to(dtype=x_feats.dtype)  # [B,L,V]

        logits = self.denoiser(
            t.to(dtype=x_feats.dtype),
            y_onehot,
            x_feats,
            mask=mask,
        )  # [B,L,V]
        logp = F.log_softmax(logits, dim=-1)

        is_t1 = t == 1
        loss = torch.zeros((B,), device=device, dtype=logits.dtype)

        if is_t1.any():
            ce = F.nll_loss(
                logp[is_t1].reshape(-1, self.V),
                y_tokens[is_t1].reshape(-1),
                reduction="none",
            ).reshape(-1, L)
            ce = (ce * active[is_t1]).sum(dim=1) / active[is_t1].sum(dim=1).clamp_min(1.0)
            loss[is_t1] = ce

        if (~is_t1).any():
            idx = ~is_t1
            t2 = t[idx]
            y0_2 = y_tokens[idx]
            yt_2 = yt[idx]
            logp_2 = logp[idx]
            active_2 = active[idx]

            qpost = self.q_posterior_xtm1_given_xt_x0(y0_2, yt_2, t2)  # [B2,L,V]
            logq = (qpost.clamp_min(1e-12)).log()
            kl = (qpost * (logq - logp_2)).sum(dim=-1)  # [B2,L]
            kl = (kl * active_2).sum(dim=1) / active_2.sum(dim=1).clamp_min(1.0)
            loss[idx] = kl

        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        x_feats: torch.Tensor,
        mask: torch.Tensor | None = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Unconditional ancestral sampling for the target positions only.
        """
        if device is None:
            device = next(self.parameters()).device
        V = self.V
        T = self.T

        if mask is None:
            mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.float32)

        y_tokens = torch.randint(0, V, (batch_size, seq_len), device=device, generator=generator)

        for t_int in range(T, 0, -1):
            t = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
            y_onehot = F.one_hot(y_tokens, num_classes=V).to(dtype=x_feats.dtype)
            logits = self.denoiser(t.to(dtype=x_feats.dtype), y_onehot, x_feats, mask=mask)
            probs = torch.softmax(logits, dim=-1)
            y_tokens = torch.multinomial(probs.reshape(-1, V), 1, generator=generator).reshape(batch_size, seq_len)

        return y_tokens


# -----------------------------
# Conditional sampling with fixed context
# -----------------------------
@torch.no_grad()
def conditional_sample(
    model: nn.Module,
    process: D3PM,
    *,
    x_target: torch.Tensor,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    mask_target: torch.Tensor | None = None,
    mask_context: torch.Tensor | None = None,
    num_sample_steps: Optional[int] = None,
    num_inner_steps: int = 5,
    progress: bool = False,
    progress_desc: str | None = None,
) -> torch.Tensor:
    """
    Conditional sampling with RePaint-style inner loops that keep context labels clamped.
    """
    device = x_target.device
    dtype = x_target.dtype
    V = process.V
    T = process.T

    y_ctx_tokens = y_context.argmax(dim=-1)  # [B,M]
    B_tgt, N_tgt, _ = x_target.shape
    M = x_context.shape[1]

    if mask_target is None:
        mask_target = torch.zeros((B_tgt, N_tgt), device=device, dtype=torch.float32)
    if mask_context is None:
        mask_context = torch.zeros((B_tgt, M), device=device, dtype=torch.float32)

    y_tokens = torch.randint(0, V, (B_tgt, N_tgt), device=device)

    if num_sample_steps is None or num_sample_steps >= T:
        t_schedule = list(range(T, 0, -1))
    else:
        lin = torch.linspace(T, 1, int(num_sample_steps), dtype=torch.long)
        t_schedule = sorted({int(v) for v in lin.tolist()}, reverse=True)

    iterator = enumerate(t_schedule)
    if progress:
        from tqdm.auto import tqdm
        iterator = enumerate(tqdm(t_schedule, desc=progress_desc or "sampling"))

    for idx, t_int in iterator:
        t = torch.full((B_tgt,), t_int, device=device, dtype=torch.long)

        t_tensor = t.to(dtype=dtype)

        for _ in range(num_inner_steps):
            # (1) simulate context at time t
            if t_int > 1:
                y_ctx_t = process.q_sample(y_ctx_tokens, t, generator=None)
            else:
                y_ctx_t = y_ctx_tokens

            # (2) reverse step on context+target
            y_ctx_oh = F.one_hot(y_ctx_t, num_classes=V).to(dtype=dtype)
            y_tgt_oh = F.one_hot(y_tokens, num_classes=V).to(dtype=dtype)
            y_aug = torch.cat([y_ctx_oh, y_tgt_oh], dim=1)
            x_aug = torch.cat([x_context, x_target], dim=1)
            mask_aug = torch.cat([mask_context, mask_target], dim=1)

            logits = model(t_tensor, y_aug, x_aug, mask=mask_aug)
            probs = torch.softmax(logits, dim=-1)
            samples = torch.multinomial(probs.reshape(-1, V), 1).reshape_as(probs[..., 0])

            tgt_samples = samples[:, M:]
            y_tokens = torch.where(mask_target.bool(), y_tokens, tgt_samples)

            # (3) forward step toward next scheduled time (inject noise)
            # approximate using one forward transition of the next time in schedule
            if idx + 1 < len(t_schedule):
                next_t = t_schedule[idx + 1]
            else:
                next_t = max(t_int - 1, 1)
            if next_t >= 1:
                Qt_next = process.schedule.Q[next_t - 1]  # [V,V]
                probs_fwd = Qt_next[y_tokens]             # gather rows; shape [B,N,V]
                y_tokens = torch.multinomial(probs_fwd.reshape(-1, V), 1).reshape(B_tgt, N_tgt)

    return F.one_hot(y_tokens, num_classes=V).to(dtype=dtype)


# -----------------------------
# Module-level loss wrapper (compatibility)
# -----------------------------
def loss(
    process: D3PM,
    model: nn.Module,
    batch,
    key: torch.Generator,
    *,
    num_timesteps: int,
    loss_type: str | None = None,
) -> torch.Tensor:
    """
    Loss wrapper to mirror the previous API signature.
    Expects batch.y_* to be one-hot; converts to token indices internally.
    """
    _ = loss_type  # not used in discrete variant

    x_parts, y_parts, mask_parts = [], [], []

    if batch.x_context is not None and batch.y_context is not None:
        x_parts.append(batch.x_context)
        y_parts.append(batch.y_context)
        mask_parts.append(torch.zeros(batch.y_context.shape[:2], device=batch.y_context.device)
                          if batch.mask_context is None else batch.mask_context)

    x_parts.append(batch.x_target)
    y_parts.append(batch.y_target)
    mask_parts.append(torch.zeros(batch.y_target.shape[:2], device=batch.y_target.device)
                      if batch.mask_target is None else batch.mask_target)

    x_all = torch.cat(x_parts, dim=1)
    y_all = torch.cat(y_parts, dim=1)
    mask_all = torch.cat(mask_parts, dim=1) if mask_parts else None

    y_tokens = y_all.argmax(dim=-1)  # [B,N]
    return process.loss(
        y_tokens,
        x_all,
        mask=mask_all,
        t=None,
        generator=key,
    )


__all__ = [
    "sinusoidal_time_embedding",
    "D3PMSchedule",
    "D3PM",
    "loss",
    "conditional_sample",
]
