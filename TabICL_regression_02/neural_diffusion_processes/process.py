# ndp_diffusion_torch.py
# PyTorch port of the original JAX GaussianDiffusion utilities
# ============================================================

from __future__ import annotations

import math
from typing import Protocol, Tuple

import torch


def _expand_to(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Make `a` broadcast along the trailing axes of `ref`
    so that shapes line up for element-wise arithmetic.
    """
    return a.reshape(a.shape + (1,) * (ref.ndim - a.ndim))


def _as_int(value: int | torch.Tensor) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _randn_like(
    ref: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    try:
        return torch.randn_like(ref, generator=generator)
    except TypeError:
        return torch.randn(
            ref.shape,
            dtype=ref.dtype,
            device=ref.device,
            generator=generator,
        )


def _build_reverse_schedule(
    num_timesteps: int,
    num_sample_steps: int | None,
    *,
    include_zero: bool,
) -> list[int]:
    if (
        num_sample_steps is None
        or int(num_sample_steps) <= 0
        or int(num_sample_steps) >= num_timesteps
    ):
        if include_zero:
            return list(range(num_timesteps - 1, -1, -1))
        return list(range(num_timesteps - 1, 0, -1))

    lin = torch.linspace(0, num_timesteps - 1, int(num_sample_steps), dtype=torch.long)
    schedule = sorted({int(v) for v in lin.tolist()}, reverse=True)
    if not schedule or schedule[0] != num_timesteps - 1:
        schedule = [num_timesteps - 1] + [t for t in schedule if t != num_timesteps - 1]

    if include_zero:
        if 0 not in schedule:
            schedule.append(0)
    else:
        schedule = [t for t in schedule if t != 0]

    return schedule


def cosine_schedule(
    beta_start: float,
    beta_end: float,
    timesteps: int,
    s: float = 0.008,
) -> torch.Tensor:
    """
    DDPM-style cosine schedule (Nichol & Dhariwal, 2021).

    Returns: 1-D tensor of length `timesteps`
    """
    x = torch.linspace(0, timesteps, timesteps + 1)
    f = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = f / f[0]
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = betas.clamp(1e-4, 0.9999)
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return betas * (beta_end - beta_start) + beta_start


class EpsModel(Protocol):
    def __call__(
        self,
        t: torch.Tensor,
        yt: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        key: torch.Generator | None,
    ) -> torch.Tensor:
        ...


class GaussianDiffusion:
    """
    Implements q(x_t|x_0) and p(x_{t-1}|x_t) for a fixed beta schedule.
    """

    def __init__(self, betas: torch.Tensor) -> None:
        self.device = betas.device
        self.dtype = betas.dtype
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _normalize_sampling_method(self, method: str) -> str:
        method = str(method).lower()
        if method not in {"repaint", "ddpm", "ddim"}:
            raise ValueError(f"Unknown sampling method: {method}")
        return method

    def pt0(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Closed-form p(y_t | y_0) moments.
        Returns (mean, var) each of shape [N, y_dim].
        """
        alpha_bar_t = self.alpha_bars[t].to(y0.device)
        mean = torch.sqrt(alpha_bar_t) * y0
        var = 1.0 - alpha_bar_t
        return mean, var

    def forward(
        self,
        key: torch.Generator | None,
        y0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample y_t and return the exact noise eps so that
            y_t = sqrt(alpha_bar_t) y0 + sqrt(1-alpha_bar_t) eps
        """
        del key
        mean, var = self.pt0(y0, t)
        noise = torch.randn(y0.shape, dtype=y0.dtype, device=y0.device)
        yt = mean + torch.sqrt(var) * noise
        return yt, noise

    def predict_x0(
        self,
        noise: torch.Tensor,
        yt: torch.Tensor,
        t: int | torch.Tensor,
    ) -> torch.Tensor:
        t_int = _as_int(t)
        alpha_bar_t = _expand_to(self.alpha_bars[t_int].to(yt.device), yt)
        return (
            yt - torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8)) * noise
        ) / torch.sqrt(alpha_bar_t.clamp(min=1e-8))

    def ddpm_backward_step(
        self,
        key: torch.Generator | None,
        noise: torch.Tensor,
        yt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deterministic DDPM mean plus stochastic beta_t noise.
        """
        beta_t = _expand_to(self.betas[t], yt)
        alpha_t = _expand_to(self.alphas[t], yt)
        alpha_bar_t = _expand_to(self.alpha_bars[t], yt)

        z = torch.zeros_like(yt)
        if t.item() > 0:
            z = torch.randn(yt.shape, dtype=yt.dtype, device=yt.device, generator=key)

        a = 1.0 / torch.sqrt(alpha_t)
        b = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = a * (yt - b * noise)
        return mean + torch.sqrt(beta_t) * z

    def ddim_backward_step(
        self,
        key: torch.Generator | None,
        noise: torch.Tensor,
        yt: torch.Tensor,
        t: int | torch.Tensor,
        *,
        next_t: int | torch.Tensor | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        if float(eta) < 0.0:
            raise ValueError(f"DDIM eta must be non-negative, got {eta}")

        t_int = _as_int(t)
        next_t_int = t_int - 1 if next_t is None else _as_int(next_t)

        x0_hat = self.predict_x0(noise, yt, t_int)
        if next_t_int < 0:
            return x0_hat

        alpha_bar_t = _expand_to(self.alpha_bars[t_int].to(yt.device), yt)
        alpha_bar_next = _expand_to(self.alpha_bars[next_t_int].to(yt.device), yt)

        eta_value = float(eta)
        if eta_value == 0.0:
            sigma = torch.zeros_like(alpha_bar_t)
        else:
            sigma = eta_value * torch.sqrt(
                ((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t).clamp(min=1e-8)).clamp(min=0.0)
                * (1.0 - alpha_bar_t / alpha_bar_next.clamp(min=1e-8)).clamp(min=0.0)
            )

        direction = torch.sqrt((1.0 - alpha_bar_next - sigma.pow(2)).clamp(min=0.0)) * noise
        y_next = torch.sqrt(alpha_bar_next.clamp(min=1e-8)) * x0_hat + direction
        if eta_value > 0.0:
            y_next = y_next + sigma * _randn_like(yt, generator=key)
        return y_next

    def reverse_step(
        self,
        key: torch.Generator | None,
        noise: torch.Tensor,
        yt: torch.Tensor,
        t: int | torch.Tensor,
        *,
        method: str = "ddpm",
        next_t: int | torch.Tensor | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        method = self._normalize_sampling_method(method)
        if method in {"repaint", "ddpm"}:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(int(t), device=yt.device)
            return self.ddpm_backward_step(key, noise, yt, t)
        return self.ddim_backward_step(key, noise, yt, t, next_t=next_t, eta=eta)

    def ddpm_backward_mean_var(
        self,
        noise: torch.Tensor,
        yt: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        beta_t = _expand_to(self.betas[t], yt)
        alpha_t = _expand_to(self.alphas[t], yt)
        alpha_bar_t = _expand_to(self.alpha_bars[t], yt)
        mean = (yt - beta_t / torch.sqrt(1 - alpha_bar_t) * noise) / torch.sqrt(alpha_t)
        var = torch.clamp(beta_t * (t > 0), min=1e-3)
        return mean, var

    def sample(
        self,
        key: torch.Generator | None,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        model_fn: EpsModel,
        output_dim: int = 1,
        num_sample_steps: int | None = None,
        method: str = "ddpm",
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Draw an unconditional sample y(x) from the learned reverse process.
        """
        device = self.device
        batch_size = x.size(0)
        y = torch.randn(
            batch_size,
            output_dim,
            device=device,
            dtype=self.dtype,
            generator=key,
        )

        if mask is None:
            mask = torch.zeros(batch_size, device=device, dtype=self.dtype)

        method = self._normalize_sampling_method(method)
        t_schedule = _build_reverse_schedule(
            len(self.betas),
            num_sample_steps,
            include_zero=True,
        )

        for step_idx, t in enumerate(t_schedule):
            g_model = torch.Generator(device)
            g_rev = torch.Generator(device)
            g_model.manual_seed(torch.randint(0, 2**63 - 1, (1,)).item())
            g_rev.manual_seed(torch.randint(0, 2**63 - 1, (1,)).item())

            eps_hat = model_fn(torch.tensor(t, device=device), y, x, mask, key=g_model)
            next_t = t_schedule[step_idx + 1] if step_idx + 1 < len(t_schedule) else -1
            y = self.reverse_step(
                g_rev,
                eps_hat,
                y,
                torch.tensor(t, device=device),
                method=method,
                next_t=next_t,
                eta=eta,
            )
        return y

    @torch.no_grad()
    def conditional_sample(
        self,
        key: torch.Generator | None,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        mask_context: torch.Tensor | None,
        model_fn: EpsModel,
        num_inner_steps: int = 5,
        method: str = "repaint",
        num_sample_steps: int | None = None,
        eta: float = 0.0,
        progress: bool = False,
        progress_desc: str | None = None,
    ) -> torch.Tensor:
        """
        Conditional sampling.

        - ``repaint`` / ``ddpm`` keep the original RePaint-style loop.
        - ``ddim`` uses a direct sparse reverse path to ``t=0`` without
          inner repaint steps or extra target re-noising, so low-step DDIM
          matches the intended coarse deterministic schedule.
        """
        method = self._normalize_sampling_method(method)
        device, dtype = self.device, self.dtype
        num_target = x.size(0)
        y_dim = y_context.size(-1)

        if mask is None:
            mask = torch.zeros(num_target, device=device, dtype=dtype)
        if mask_context is None:
            mask_context = torch.zeros(len(x_context), device=device, dtype=dtype)

        x_aug = torch.cat([x_context, x], dim=0)
        mask_aug = torch.cat([mask_context, mask], dim=0)
        num_context = len(x_context)

        y_t = torch.randn(num_target, y_dim, device=device, dtype=dtype, generator=key)
        if method == "ddim":
            t_schedule = _build_reverse_schedule(
                len(self.betas),
                num_sample_steps,
                include_zero=True,
            )
        else:
            t_schedule = _build_reverse_schedule(
                len(self.betas),
                num_sample_steps,
                include_zero=False,
            )

        bar = None
        if progress:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(t_schedule), desc=progress_desc or "sampling")

        for idx, t in enumerate(t_schedule):
            t_tensor = torch.tensor(t, device=device)

            if method == "ddim":
                # Deterministic sparse conditional DDIM:
                # use the current context-noised labels at timestep t, then jump
                # directly to the next sparse timestep from the schedule.
                g_fwd = torch.Generator(device)
                g_fwd.manual_seed(torch.randint(0, 2**63 - 1, (1,)).item())
                y_ctx_t, _ = self.forward(g_fwd, y_context, t_tensor)

                y_aug = torch.cat([y_ctx_t, y_t], dim=0)
                g_model = torch.Generator(device)
                g_rev = torch.Generator(device)
                g_model.manual_seed(torch.randint(0, 2**63 - 1, (1,)).item())
                g_rev.manual_seed(torch.randint(0, 2**63 - 1, (1,)).item())

                eps_hat = model_fn(t_tensor, y_aug, x_aug, mask_aug, key=g_model)
                next_t = t_schedule[idx + 1] if idx + 1 < len(t_schedule) else -1
                y_prev = self.reverse_step(
                    g_rev,
                    eps_hat,
                    y_aug,
                    t_tensor,
                    method=method,
                    next_t=next_t,
                    eta=eta,
                )
                y_t = y_prev[num_context:]
                if bar is not None:
                    bar.update(1)
                continue

            for _ in range(int(num_inner_steps)):
                g_fwd = torch.Generator(device)
                g_fwd.manual_seed(torch.randint(0, 2**63 - 1, (1,)).item())
                y_ctx_t, _ = self.forward(g_fwd, y_context, t_tensor)

                y_aug = torch.cat([y_ctx_t, y_t], dim=0)

                g_model = torch.Generator(device)
                g_rev = torch.Generator(device)
                g_model.manual_seed(torch.randint(0, 2**63 - 1, (1,)).item())
                g_rev.manual_seed(torch.randint(0, 2**63 - 1, (1,)).item())

                eps_hat = model_fn(t_tensor, y_aug, x_aug, mask_aug, key=g_model)
                y_prev = self.reverse_step(
                    g_rev,
                    eps_hat,
                    y_aug,
                    t_tensor,
                    method=method,
                    eta=eta,
                )
                y_t = y_prev[num_context:]

                if idx + 1 < len(t_schedule):
                    t_next = t_schedule[idx + 1]
                else:
                    t_next = max(t - 1, 0)
                beta_next = _expand_to(self.betas[t_next], y_t)
                y_t = (
                    torch.sqrt(1.0 - beta_next) * y_t
                    + torch.sqrt(beta_next) * _randn_like(y_t, generator=g_fwd)
                )

            g_fwd = torch.Generator(device)
            g_fwd.manual_seed(torch.randint(0, 2**63 - 1, (1,)).item())
            y_ctx_t, _ = self.forward(g_fwd, y_context, t_tensor)

            y_aug = torch.cat([y_ctx_t, y_t], dim=0)
            g_model = torch.Generator(device)
            g_rev = torch.Generator(device)
            g_model.manual_seed(torch.randint(0, 2**63 - 1, (1,)).item())
            g_rev.manual_seed(torch.randint(0, 2**63 - 1, (1,)).item())

            eps_hat = model_fn(t_tensor, y_aug, x_aug, mask_aug, key=g_model)
            y_prev = self.reverse_step(
                g_rev,
                eps_hat,
                y_aug,
                t_tensor,
                method=method,
                eta=eta,
            )
            y_t = y_prev[num_context:]
            if bar is not None:
                bar.update(1)

        if bar is not None:
            bar.close()

        return y_t


def loss(
    process: GaussianDiffusion,
    network: EpsModel,
    batch,
    key: torch.Generator,
    *,
    num_timesteps: int,
    loss_type: str = "l1",
) -> torch.Tensor:
    metric = (lambda a, b: (a - b).abs()) if loss_type == "l1" else (lambda a, b: (a - b) ** 2)

    batch_size, num_points, _y_dim = batch.y_target.shape
    device = batch.y_target.device
    t = torch.randint(0, num_timesteps, (batch_size,), generator=key, device=device)

    t_ = t.view(batch_size, 1, 1)
    alpha_bar_t = process.alpha_bars[t_].to(device)
    yt = torch.sqrt(alpha_bar_t) * batch.y_target + torch.sqrt(1.0 - alpha_bar_t) * torch.randn_like(
        batch.y_target
    )

    noise_hat = network(
        t.to(dtype=torch.float32),
        yt,
        batch.x_target,
        batch.mask_target if batch.mask_target is not None else torch.zeros(batch_size, num_points, device=device),
        key=key,
    )
    noise_true = (yt - torch.sqrt(alpha_bar_t) * batch.y_target) / torch.sqrt(1.0 - alpha_bar_t)

    loss_per = metric(noise_true, noise_hat).sum(-1)

    mask_target = (
        batch.mask_target if batch.mask_target is not None else torch.zeros(batch_size, num_points, device=device)
    )
    mask = 1.0 - mask_target

    loss_per = loss_per * mask
    return loss_per.sum() / mask.sum()
