# samplers_ddim.py (refined parts only)
from __future__ import annotations
from dataclasses import dataclass
import torch
from typing import Callable, Optional

EpsModel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]

@dataclass
class DDIMSampler:
    process: any               # expects .alpha_bars [T]
    num_sample_steps: int = 50
    eta: float = 0.0           # 0 = deterministic DDIM, 1 ≈ DDPM variance

    def __post_init__(self):
        T  = int(self.process.alpha_bars.numel())
        ts = torch.linspace(T - 1, 0, steps=self.num_sample_steps)
        idx = ts.round().to(torch.long)                  # coarse, descending indices
        self.timesteps = torch.unique_consecutive(idx)   # remove consecutive dupes
        if self.timesteps[-1].item() != 0:
            self.timesteps = torch.cat(
                [self.timesteps, torch.tensor([0], device=idx.device, dtype=torch.long)]
            )

    # ---- General DDIM step (supports any eta) ------------------------------
    def _ddim_step(self, y, x, mask, t_i: int, t_j: int, model_fn: EpsModel, key: torch.Generator):
        """
        y  : current x_{t_i}
        return: x_{t_j} using DDIM Eq. (12) with σ_eta from Eq. (16)
        """
        device = y.device
        abars = self.process.alpha_bars.to(device)

        t_i_t = torch.tensor(t_i, device=device, dtype=torch.long)
        t_j_t = torch.tensor(t_j, device=device, dtype=torch.long)

        abar_i = abars[t_i_t]            # \bar{α}_{t_i}
        abar_j = abars[t_j_t]            # \bar{α}_{t_j}

        a_i  = torch.sqrt(abar_i)        # √\bar{α}_{t_i}
        ai1  = torch.sqrt(torch.clamp(1.0 - abar_i, min=0.0))
        a_j  = torch.sqrt(abar_j)        # √\bar{α}_{t_j}

        # ε̂(x_{t_i}, t_i)
        eps_hat = model_fn(t_i_t, y, x, mask, key=key)

        # x̂_0 from the forward relation
        x0_hat = (y - ai1 * eps_hat) / (a_i + 1e-12)

        # σ_eta from DDIM Eq. (16)
        if self.eta > 0:
            # sigma_eta = eta * sqrt((1 - abar_j)/(1 - abar_i)) * sqrt(1 - abar_i/abar_j)
            term1 = torch.clamp((1.0 - abar_j) / (1.0 - abar_i + 1e-12), min=0.0)
            term2 = torch.clamp(1.0 - (abar_i / (abar_j + 1e-12)), min=0.0)
            sigma_eta = self.eta * torch.sqrt(term1 * term2)
        else:
            sigma_eta = torch.tensor(0.0, device=device, dtype=y.dtype)

        # coefficient for ε̂ at t_j
        coeff_eps = torch.sqrt(torch.clamp(1.0 - abar_j - sigma_eta**2, min=0.0))

        # optional stochastic kick
        if float(sigma_eta) > 0.0:
            z = torch.randn(y.shape, dtype=y.dtype, device=device, generator=key)
        else:
            z = None

        # DDIM Eq. (12): x_{t_j} = √\bar{α}_{t_j} x̂_0 + √(1-\bar{α}_{t_j}-σ^2) ε̂ + σ z
        y_next = a_j * x0_hat + coeff_eps * eps_hat + (sigma_eta * z if z is not None else 0.0)
        return y_next

    # ---- Unconditional -----------------------------------------------------
    @torch.no_grad()
    def sample(self, key: torch.Generator, x: torch.Tensor, model_fn: EpsModel,
                      mask: Optional[torch.Tensor] = None, y_dim: int = 1):
        device = x.device
        N = x.size(0)
        y = torch.randn((N, y_dim), dtype=x.dtype, device=device, generator=key)  # x_{t_max} ~ N(0,I)
        if mask is None:
            mask = torch.zeros(N, device=device, dtype=x.dtype)

        for i in range(len(self.timesteps) - 1):
            t_i = int(self.timesteps[i].item())
            t_j = int(self.timesteps[i + 1].item())
            y   = self._ddim_step(y, x, mask, t_i, t_j, model_fn, key)
        return y
