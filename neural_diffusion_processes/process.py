# ndp_diffusion_torch.py
# PyTorch port of the original JAX GaussianDiffusion utilities
# ============================================================

from __future__ import annotations
from typing import Protocol, Tuple
import math
import torch
import torch.nn.functional as F

# ---------- protocol for the ε-network -------------------------------------
class EpsModel(Protocol):  # Bidimentional model
    def __call__(self,
                 t: torch.Tensor,        # scalar int64        []
                 y_target: torch.Tensor,       # noisy targets       [N, y_dim]
                 x_target: torch.Tensor,        # inputs              [N, x_dim]
                 mask: torch.Tensor,     # 1 if "missing"      [N]
                 *,
                 y_context:torch.Tensor,
                 x_context: torch.Tensor,
                 key: torch.Generator   # for dropout etc.
                 ) -> torch.Tensor:      # predicted noise     [N, y_dim]
        ...


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
                key: torch.Generator,  # a random seed class
                y0: torch.Tensor,
                t:  torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample y_t and return the exact noise ε so that
            y_t = √ᾱ_t y0 + √(1-ᾱ_t) ε
        """
        m, v   = self.pt0(y0, t)
        noise = torch.randn(y0.shape, dtype=y0.dtype, device=y0.device, generator=key)
        yt     = m + torch.sqrt(v) * noise
        return yt, noise # [N, y_dim]

    # -------------------------------------------------- single reverse step
    def ddpm_backward_step(self,
                           key: torch.Generator,
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
            z = torch.randn(yt.shape, dtype=yt.dtype, device=yt.device, generator=key)

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

    # ----------------------------------------------------------- full sample
    def sample(self,
               key: torch.Generator,
               x:   torch.Tensor,           # [B, x_dim]
               mask: torch.Tensor | None,
               *,
               x_context: torch.Tensor,
               y_context: torch.Tensor,
               model_fn: EpsModel,
               output_dim: int = 1
               ) -> torch.Tensor:
        """
        Draw *unconditional* sample y(x) from learned reverse process.
        """
        device = self.device
        B      = x.size(0)
        y = torch.randn(B, output_dim, device=device, dtype=self.dtype,
                        generator=key)             # initial y_T [B, T]

        if mask is None:
            mask = torch.zeros(B, device=device, dtype=self.dtype) # [B]

        for t in reversed(range(len(self.betas))):
            g1, g2 = torch.Generator(device), torch.Generator(device)
            g1.manual_seed(torch.randint(0, 2**63-1, (1,)).item())
            g2.manual_seed(torch.randint(0, 2**63-1, (1,)).item())

            eps_hat = model_fn(torch.tensor(t, device=device),
                               y, x, mask, key=g1)
            y = self.ddpm_backward_step(g2, eps_hat, y,
                                        torch.tensor(t, device=device))
        return y


    def sample(self,
               key: torch.Generator,
               x:   torch.Tensor,           # [B, x_dim]
               mask: torch.Tensor | None,
               *,
               x_context: torch.Tensor,
               y_context: torch.Tensor,
               model_fn: EpsModel,
               output_dim: int = 1
               ) -> torch.Tensor:
        """
        Draw *unconditional* sample y(x) from learned reverse process.
        """
        device = self.device
        B      = x.size(0)
        y = torch.randn(B, output_dim, device=device, dtype=self.dtype,
                        generator=key)             # initial y_T [B, T]

        if mask is None:
            mask = torch.zeros(B, device=device, dtype=self.dtype) # [B]

        for t in reversed(range(len(self.betas))):
            g1, g2 = torch.Generator(device), torch.Generator(device)
            g1.manual_seed(torch.randint(0, 2**63-1, (1,)).item())
            g2.manual_seed(torch.randint(0, 2**63-1, (1,)).item())

            eps_hat = model_fn(torch.tensor(t, device=device),
                               y, x, mask, key=g1)
            y = self.ddpm_backward_step(g2, eps_hat, y,
                                        torch.tensor(t, device=device))
        return y


   # ------------------------------------------------------- conditional DDPM
    @torch.no_grad()
    def conditional_sample(
        self,
        key: torch.Generator,
        x: torch.Tensor,                 # [N, x_dim]      – target inputs
        mask: torch.Tensor | None,
        *,
        x_context: torch.Tensor,         # [M, x_dim]
        y_context: torch.Tensor,         # [M, y_dim]
        mask_context: torch.Tensor | None,
        model_fn: EpsModel,
        num_inner_steps: int = 5,
    ) -> torch.Tensor:
        """
        Conditional generation.

        Methods:
          - 'repaint' (original): forward-diffuses context each step and uses RePaint inner loop.
          - 'unified' (ours): keeps context CLEAN, predicts eps for targets only, and updates targets only.
        """
        assert method in ("repaint", "unified"), "method must be 'repaint' or 'unified'"

        device, dtype = self.device, self.dtype
        B_tgt         = x.size(0)
        y_dim         = y_context.size(-1)

        # ---- default masks ------------------------------------------------
        if mask         is None: mask         = torch.zeros(B_tgt, device=device, dtype=dtype)  # 0 ⇒ target
        if mask_context is None: mask_context = torch.ones (len(x_context), device=device, dtype=dtype)  # 1 ⇒ context

        # ---- concatenate arrays for one network call ----------------------
        x_aug    = torch.cat([x_context, x], dim=0)          # [(M+N), x_dim]
        mask_aug = torch.cat([mask_context, mask], dim=0)    # [(M+N)]
        num_ctx  = len(x_context)

        # ---- initial noisy target at time T -------------------------------
        y_t = torch.randn(B_tgt, y_dim, device=device, dtype=dtype, generator=key)

        # ------------------ unified path: keep context CLEAN ----------------
        for t in range(len(self.betas) - 1, -1, -1):  # include t=0
            t_tensor = torch.tensor(t, device=device)
            # Net sees clean context + current noisy targets; predicts eps for targets only
            y_aug_in = torch.cat([y_context, y_t], dim=0)         # [(M+N), y_dim]
            g_model  = torch.Generator(device); g_model.manual_seed(torch.randint(0, 2**63-1, (1,)).item())
            g_rev    = torch.Generator(device);   g_rev.manual_seed(torch.randint(0, 2**63-1, (1,)).item())

            eps_hat_tgt = model_fn(t_tensor, y_aug_in, x_aug, mask_aug, key=g_model)  # [N, y_dim]
            y_t         = self.ddpm_backward_step(g_rev, eps_hat_tgt, y_t, t_tensor)  # update targets only
        return y_t

        # after the loop y_t is at time 0 → final sample
        return y_t


def loss(process: GaussianDiffusion,
         network: EpsModel,
         batch,
         key: torch.Generator,
         *,
         num_timesteps: int,
         loss_type: str = "l1") -> torch.Tensor:

    metric = (lambda a, b: (a - b).abs()) if loss_type == "l1" \
             else (lambda a, b: (a - b) ** 2)

    # B = batch size, N = # of target points
    B, N, y_dim = batch.y_target.shape

    device = batch.y_target.device
    t = torch.randint(0, num_timesteps, (B,), generator=key, device=device)  # [B]

    # Expand for batched computation
    t_ = t.view(B, 1, 1)                              # [B,1,1]
    ᾱ_t = process.alpha_bars[t_].to(device)          # [B,1,1]
    yt = torch.sqrt(ᾱ_t) * batch.y_target + \
         torch.sqrt(1. - ᾱ_t) * torch.randn_like(batch.y_target)  # [B,N,1]

    # Run model
    noise_hat = network(
        t.to(dtype=torch.float32),
        yt,
        batch.x_target,
        batch.mask_target if batch.mask_target is not None else torch.zeros(B, N, device=device),
        key=key
    )
    noise_true = (yt - torch.sqrt(ᾱ_t) * batch.y_target) / torch.sqrt(1. - ᾱ_t)

    # Per-point loss
    loss_per = metric(noise_true, noise_hat).sum(-1)  # [B,N]

    mask_target = batch.mask_target if batch.mask_target is not None else torch.zeros(B, N, device=device)
    mask = 1.0 - mask_target

    loss_per = loss_per * mask                        # [B,N]

    return loss_per.sum() / mask.sum()

