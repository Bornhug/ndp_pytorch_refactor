# ndp_diffusion_torch.py
# PyTorch port of the original JAX GaussianDiffusion utilities
# ============================================================

from __future__ import annotations
from typing import Tuple
import math
import torch
import torch.nn.functional as F

from .model import BiDimensionalAttentionModel


# ---------- helpers ---------------------------------------------------------
def _expand_to(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Make `a` broadcast along the trailing axes of `ref`
    so that shapes line up for element-wise arithmetic.
    """
    new_shape = a.shape + (1,) * (ref.ndim - a.ndim)
    return a.reshape(new_shape)


def cosine_schedule(beta_start: float,
                    beta_end: float,
                    timesteps: int,
                    s: float = 0.008) -> torch.Tensor:
    """
    DDPM-style cosine β-schedule (Nichol & Dhariwal, 2021).

    Returns: 1-D tensor of length `timesteps`
    """
    x  = torch.linspace(0, timesteps, timesteps + 1)
    f  = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = f / f[0]                       # cumulative ᾱ_t
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = betas.clamp(1e-4, 0.9999)           # numerical safety
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return betas * (beta_end - beta_start) + beta_start


# ---------- main diffusion engine ------------------------------------------
class GaussianDiffusion:
    """
    Implements q(x_t|x_0) and p_θ(x_{t-1}|x_t) for a fixed β-schedule.
    """

    def __init__(self, betas: torch.Tensor) -> None:
        """
        betas : shape [T]  — variance schedule (0 < β_t < 1)
        """
        self.device = betas.device
        self.dtype  = betas.dtype
        self.betas        = betas                                    # [T]
        self.alphas       = 1.0 - betas                              # [T]
        self.alpha_bars   = torch.cumprod(self.alphas, dim=0)        # [T]

    # ------------------------------------------------------------------ pt0
    def pt0(self,
            y0: torch.Tensor,        # clean targets [N, y_dim]
            t:  torch.Tensor         # scalar int64  []
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Closed-form p(y_t | y_0) moments.
        Returns (mean, var) each of shape [N, y_dim].
        """
        alpha_bars = self.alpha_bars[t].to(y0.device)  # ⬅️ ensure ᾱ_t is on the same device as y0
        m    = torch.sqrt(alpha_bars) * y0
        v    = (1.0 - alpha_bars)
        return m, v   # [N, y_dim]

    # ---------------------------------------------------------------- forward
    def forward(self,
                key: torch.Generator | None,  # kept for API compatibility
                y0: torch.Tensor,
                t:  torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample y_t and return the exact noise ε so that
            y_t = √ᾱ_t y0 + √(1-ᾱ_t) ε
        """
        m, v   = self.pt0(y0, t)
        # Use global RNG on the same device as y0 to avoid
        # device mismatches between generator and tensors.
        noise = torch.randn(y0.shape, dtype=y0.dtype, device=y0.device)
        # Clamp variance to prevent sqrt of negative or very small numbers
        v_clamped = v.clamp(min=1e-8)
        yt     = m + torch.sqrt(v_clamped) * noise
        return yt, noise # [N, y_dim]

    # -------------------------------------------------- single reverse step
    def ddpm_backward_step(self,
                           key: torch.Generator | None,
                           noise: torch.Tensor,  # ε̂_θ [N, y_dim]
                           yt:   torch.Tensor,   # [N, y_dim]
                           t:    torch.Tensor    # Scalar []
                           ) -> torch.Tensor:    # [N, y_dim]
        """
        Deterministic DDPM μ_t plus stochastic σ_t z  (Alg. 1 in DDPM paper).
        Implements the reverse diffusion step
            μ_t(y_t, ε̂_θ) = 1 / sqrt(α_t) * ( y_t - (β_t / sqrt(1 - ᾱ_t)) * ε̂_θ(y_t, t) )
            y_{t-1} = μ_t(y_t, ε̂_θ) + σ_t * z ,
        where in the "vanilla" DDPM parameterization we take
            σ_t^2 = β_t   ⇒   σ_t = sqrt(β_t),
        and
            α_t    = 1 - β_t
            ᾱ_t   = ∏_{s=1}^t α_s  (precomputed as self.alpha_bars).
        """
        β_t      = _expand_to(self.betas[t], yt) # [1, 1]
        α_t      = _expand_to(self.alphas[t], yt) # [1, 1]
        ᾱ_t     = _expand_to(self.alpha_bars[t], yt) # [1, 1]

        # alpha_bar at t-1 (handle t=0 specially)
        if t.item() > 0:
            ᾱ_prev = _expand_to(self.alpha_bars[t - 1], yt)
        else:
            ᾱ_prev = torch.ones_like(ᾱ_t)

        # posterior variance: \tildeβ_t = ((1 - ᾱ_{t-1}) / (1 - ᾱ_t)) * β_t
        tilde_β_t = (1.0 - ᾱ_prev) / (1.0 - ᾱ_t) * β_t

        z = torch.zeros_like(yt)
        if t.item() > 0:
            z = torch.randn(yt.shape, dtype=yt.dtype, device=yt.device)

        a = 1.0 / torch.sqrt(α_t)
        b = β_t / torch.sqrt(1.0 - ᾱ_t)
        mean = a * (yt - b * noise)
        yt_minus_one = mean + torch.sqrt(tilde_β_t) * z #TODO: variance might be wrong
        return yt_minus_one

    # TODO: check again, seems ddpm_backward_mean_var is never used
    # ---------------------------------------------- mean / var diagnostics
    def ddpm_backward_mean_var(self,
                               noise: torch.Tensor, # ε̂_θ [N, y_dim]
                               yt:    torch.Tensor, # [N, y_dim]
                               t:     torch.Tensor  # Scalar []
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        β_t  = _expand_to(self.betas[t], yt)
        α_t  = _expand_to(self.alphas[t], yt)
        ᾱ_t = _expand_to(self.alpha_bars[t], yt)
        mean = (yt - β_t / torch.sqrt(1 - ᾱ_t) * noise) / torch.sqrt(α_t)
        var  = torch.clamp(β_t * (t > 0), min=1e-3)
        return mean, var



    def sample(self,
               key: torch.Generator | None,
               x:   torch.Tensor,           # [B, x_dim]
               mask: torch.Tensor | None,
               *,
               x_context: torch.Tensor | None,
               y_context: torch.Tensor | None,
               mask_context: torch.Tensor | None,
               model: BiDimensionalAttentionModel,
               output_dim: int = 1
               ) -> torch.Tensor:
        """
        Draw *unconditional* sample y(x) from learned reverse process.
        Assumes x has shape [B, N, D] and returns y of shape [B, N, output_dim].
        """
        B, N, _ = x.shape
        y = torch.randn(B, N, output_dim, device=self.device, dtype=self.dtype, generator=key)  # initial y_T [B, N, T]

        # if mask is None:
        #     mask = torch.zeros(B, device=self.device, dtype=self.dtype) # [B]
        #
        # # TODO: zeros or ones?
        # if mask_context is None:
        #     mask_context = torch.ones (len(x_context),device=self.device, dtype=self.dtype)  # 1 ⇒ context

        for t_int in reversed(range(len(self.betas))):
            # Shared timestep across the batch; use a scalar tensor.
            t_tensor = torch.tensor(t_int, device=self.device, dtype=torch.long)
            noise_hat = model(
                x_tgt=x,
                y_tgt=y,
                t=t_tensor,
                mask_tgt=mask,
                x_context=x_context,
                y_context=y_context,
                mask_context=mask_context,
            )
            y = self.ddpm_backward_step(key, noise_hat, y, t_tensor)
        return y




def stratified_timesteps(
    batch_size: int,
    num_timesteps: int,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Stratified / low-discrepancy sampling of diffusion timesteps.

    Returns:
        t: Long tensor of shape [batch_size], with values in {0, ..., num_timesteps-1}.
    """
    if device is None:
        device = "cpu"

    T = float(num_timesteps)
    B = float(batch_size)

    # Width of each "bin" in continuous time
    step = T / B  # = num_timesteps / batch_size

    # Sample one offset in [0, step) for each batch element
    # shape: [B]
    # Use global RNG on the requested device. We ignore the optional
    # generator here to avoid device mismatches.
    t = torch.rand(batch_size, device=device) * step

    # Shift each sample into a different bin:
    # bin 0: [0*step, 1*step)
    # bin 1: [1*step, 2*step)
    # ...
    # bin B-1: [(B-1)*step, B*step) = [(B-1)*step, T)
    bins = step * torch.arange(batch_size, device=device)
    t = t + bins

    # Map to valid integer timesteps 0..T-1
    # (clamp just in case of tiny floating overshoot)
    t = t.clamp(0, T - 1 - 1e-6).long()
    return t # [B]


def loss(process: GaussianDiffusion,
         model: BiDimensionalAttentionModel,
         batch,
         key: torch.Generator,
         *,
         num_timesteps: int,
         loss_type: str = "l1") -> torch.Tensor:

    if loss_type == "l1":
        metric = lambda a, b: (a - b).abs()
    elif loss_type == "l2":
        metric = lambda a, b: (a - b) ** 2
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # B = batch size, N = # of target points
    B, N, y_dim = batch.y_target.shape

    device = batch.y_target.device
    t = stratified_timesteps(B, num_timesteps, device=device, generator=key)  # [B]

    # Expand for batched computation
    t = t.view(B, 1, 1)                              # [B,1,1]
    yt, noise_true = process.forward(key, batch.y_target, t)    # [B,N,1]


    mask_target = batch.mask_target if batch.mask_target is not None else torch.zeros(B, N, device=device)

    # Run model (BiDimensionalAttentionModel)
    noise_hat = model(
        x_tgt=batch.x_target,
        y_tgt=yt,
        t=t.to(batch.x_target.device).view(B),
        mask_tgt=mask_target.to(batch.x_target.device),
        x_context=batch.x_context,
        y_context=batch.y_context,
        mask_context=batch.mask_context,
    )

    # Per-point loss
    loss_per = metric(noise_true, noise_hat).sum(-1)  # [B,N]

    loss_per = loss_per * (1-mask_target)                    # [B,N]

    unmasked_count = (1 - mask_target).sum()
    
    # Prevent division by zero
    if unmasked_count == 0:
        return torch.tensor(0.0, device=device, dtype=loss_per.dtype)

    return loss_per.sum() / unmasked_count.clamp(min=1.0)

