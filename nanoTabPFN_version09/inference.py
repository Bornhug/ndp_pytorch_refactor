"""
Inference utilities for conditional NDP without RePaint sampling.
"""

from __future__ import annotations


from typing import Optional

import torch
import torch.nn.functional as F

from ndp_discrete.process import D3PM


def _ensure_matrix(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if x.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [N, D], got shape {tuple(x.shape)}")
    return x


def _to_one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Ensure labels are one-hot encoded (float32).
    """
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    if y.size(-1) == num_classes:
        return y.to(dtype=torch.float32)
    y_idx = y.to(dtype=torch.long).squeeze(-1)
    return F.one_hot(y_idx, num_classes=num_classes).to(dtype=torch.float32)


def _jump_posterior_from_x0(
    p_x0: torch.Tensor,  # [B,N,K] probs over x0
    x_t: torch.Tensor,  # [B,N] tokens at time t
    q_onestep: torch.Tensor,  # [T,K,K]
    q_mats: torch.Tensor,  # [T,K,K] cumulative forward
    t_int: int,
    next_t_int: int,
) -> torch.Tensor:
    """
    Compute p(x_{next_t} | x_t, x0_pred) for a jump of size k = t_int - next_t_int.
    Uses multi-step forward kernel and cumulative q_mats to incorporate p_theta(x0|x_t).
    """
    if next_t_int < 0:
        next_t_int = 0
    gap = t_int - next_t_int
    if gap <= 0:
        raise ValueError(f"Invalid jump: t={t_int}, next_t={next_t_int}")

    B, N, K = p_x0.shape
    device = p_x0.device

    # Forward jump kernel q(x_t | x_{next_t})
    q_jump = torch.eye(K, device=device)
    for s in range(next_t_int + 1, t_int + 1):
        q_jump = q_jump @ q_onestep[s]
    # Forward cumulative from x0 to x_{next_t}
    q_tk_given_x0 = q_mats[next_t_int].to(device=device)  # [K,K] rows=x0, cols=x_{t-k}

    p_post = torch.zeros_like(p_x0)
    for b in range(B):
        for n in range(N):
            # Prior over x_{t-k}: sum_{x0} p_theta(x0|xt) * q(x_{t-k}|x0)
            p_prior = torch.matmul(p_x0[b, n], q_tk_given_x0)  # [K]
            # Likelihood term q(x_t | x_{t-k})
            likelihood = q_jump[:, x_t[b, n]]  # [K]
            post = p_prior * likelihood
            post = post / (post.sum() + 1e-8)
            p_post[b, n] = post
    return p_post  # [B,N,K]


@torch.no_grad()
def direct_logits(
    model,
    process: D3PM,
    x_tgt: torch.Tensor,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    num_classes: int,
    device: torch.device,
    *,
    generator: Optional[torch.Generator] = None,
    t_value: Optional[int] = None,
) -> torch.Tensor:
    """
    Single-step forward pass to obtain target logits conditioned on context.

    Args:
        model: trained NanoTabPFNModel
        process: D3PM instance (provides vocab size / schedule length)
        x_tgt: [N_t, F]
        x_context: [N_c, F]
        y_context: [N_c] or one-hot
        num_classes: number of real classes (without absorbing)
        device: torch device
        t_value: optional timestep to condition on (defaults to T-1 for maximum noise)
    """
    x_tgt = _ensure_matrix(x_tgt.to(device=device, dtype=torch.float32), "x_tgt")
    x_context = _ensure_matrix(
        x_context.to(device=device, dtype=torch.float32), "x_context"
    )
    y_context = _ensure_matrix(y_context.to(device=device), "y_context")

    target_vocab = process.V
    y_context = _to_one_hot(y_context, num_classes)
    if target_vocab > num_classes:
        pad = target_vocab - y_context.size(-1)
        if pad > 0:
            y_context = torch.cat(
                [y_context, torch.zeros_like(y_context[..., :pad])], dim=-1
            )

    # Add batch dimension expected by the model/process.
    x_tgt = x_tgt.unsqueeze(0)
    x_context = x_context.unsqueeze(0)
    y_context = y_context.unsqueeze(0)

    # Start from maximally noised target tokens and denoise once.
    vocab_for_targets = target_vocab
    if generator is None:
        generator = torch.Generator(device=device)
    noisy_tokens = torch.randint(
        0, vocab_for_targets, (1, x_tgt.size(1)), device=device, generator=generator
    )
    y_tgt_noisy = F.one_hot(noisy_tokens, num_classes=vocab_for_targets).to(
        dtype=torch.float32
    )

    if t_value is None:
        t_value = process.T - 1
    t = torch.full((1,), int(t_value), device=device, dtype=torch.float32)

    logits = model(
        x_target=x_tgt,
        y_target=y_tgt_noisy,
        t=t,
        mask_target=None,
        x_context=x_context,
        y_context=y_context,
        mask_context=None,
    )

    out = logits.squeeze(0)
    if out.ndim == 2 and out.size(-1) > num_classes:
        out = out[..., :num_classes]
    return out


@torch.no_grad()
def diffusion_sample(
    model,
    process: D3PM,
    x_tgt: torch.Tensor,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    num_classes: int,
    device: torch.device,
    *,
    method: str = "ddpm",
    num_steps: Optional[int] = None,
    progress: bool = False,
    generator: Optional[torch.Generator] = None,
    log_json_path: Optional[str] = None,
    log_label: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Iterative reverse diffusion for targets conditioned on context.

    Args:
        method: "ddpm" (full schedule or subsampled) or "ddim" (subset).
        num_steps: number of steps for DDIM/DDPM (ignored for direct).
    """
    x_tgt = _ensure_matrix(x_tgt.to(device=device, dtype=torch.float32), "x_tgt")
    x_context = _ensure_matrix(
        x_context.to(device=device, dtype=torch.float32), "x_context"
    )
    y_context = _ensure_matrix(y_context.to(device=device), "y_context")

    V = process.V
    T = process.T

    y_context = _to_one_hot(y_context, num_classes)
    if V > num_classes:
        pad = V - y_context.size(-1)
        if pad > 0:
            y_context = torch.cat(
                [y_context, torch.zeros_like(y_context[..., :pad])], dim=-1
            )

    x_tgt = x_tgt.unsqueeze(0)
    x_context = x_context.unsqueeze(0)
    y_context = y_context.unsqueeze(0)

    if generator is None:
        generator = torch.Generator(device=device)

    # JSON logging setup (single-sample logging; full logits/probs/posteriors)
    log_steps = [] if log_json_path is not None else None

    # Schedule
    if method == "ddpm":
        if num_steps is None or num_steps <= 0 or num_steps >= T:
            t_schedule = list(range(T - 1, -1, -1))
        else:
            steps = int(num_steps)
            lin = torch.linspace(T - 1, 0, steps, dtype=torch.long)
            t_schedule = sorted({int(v) for v in lin.tolist()}, reverse=True)
    elif method == "ddim":
        steps = num_steps or 50
        lin = torch.linspace(T - 1, 0, steps, dtype=torch.long)
        t_schedule = sorted({int(v) for v in lin.tolist()}, reverse=True)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    # Initialise noisy targets at t=T-1
    y_tokens = torch.randint(
        0, V, (1, x_tgt.size(1)), device=device, generator=generator
    )

    iterator = enumerate(t_schedule)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = enumerate(tqdm(t_schedule, desc=f"{method.upper()} sampling"))
        except Exception:
            pass

    last_logits_x0 = None
    for step_idx, t_int in iterator:
        t_long = torch.full((1,), t_int, device=device, dtype=torch.long)
        t_float = t_long.to(dtype=torch.float32)

        y_tgt_oh = F.one_hot(y_tokens, num_classes=V).to(dtype=torch.float32)
        logits_x0 = model(
            x_target=x_tgt,
            y_target=y_tgt_oh,
            t=t_float,
            mask_target=None,
            x_context=x_context,
            y_context=y_context,
            mask_context=None,
        )
        last_logits_x0 = logits_x0
        p_x0 = torch.softmax(logits_x0, dim=-1)  # [1,N,V]

        # Determine next target timestep (or -1 to finish at 0)
        next_t = t_schedule[step_idx + 1] if step_idx + 1 < len(t_schedule) else -1
        gap = t_int - next_t

        if gap > 1:
            # Use jump posterior for multi-step skip
            p_xprev = _jump_posterior_from_x0(
                p_x0=p_x0,
                x_t=y_tokens,
                q_onestep=process.core.q_onestep_mats,
                q_mats=process.core.q_mats,
                t_int=t_int,
                next_t_int=next_t,
            )
        else:
            # Use the jump posterior even when gap==1, except for the final t=0 step
            # where next_t=-1 would clamp to 0 and create an invalid gap=0.
            if next_t < 0 and t_int == 0:
                p_xprev = process.core._p_xprev_from_xstart_probs(
                    p_xstart=p_x0,
                    x_t=y_tokens,
                    t=t_long,
                )  # [1,N,V]
            else:
                p_xprev = _jump_posterior_from_x0(
                    p_x0=p_x0,
                    x_t=y_tokens,
                    q_onestep=process.core.q_onestep_mats,
                    q_mats=process.core.q_mats,
                    t_int=t_int,
                    next_t_int=next_t,
                )

        if method == "ddpm":
            new_tokens = torch.multinomial(p_xprev.reshape(-1, V), 1).reshape_as(
                y_tokens
            )
        else:  # ddim: deterministic mode
            new_tokens = torch.argmax(p_xprev, dim=-1)
        y_tokens = new_tokens

        # Single-sample logging (take first element in batch if present)
        if log_steps is not None:
            # Current tokens
            x_t_list = y_tokens[0].tolist()
            # Jump kernel matrix
            if gap > 1:
                q_jump = torch.eye(V, device=device)
                for s in range(next_t + 1, t_int + 1):
                    q_jump = q_jump @ process.core.q_onestep_mats[s]
            else:
                q_jump = process.core.q_onestep_mats[t_int]

            row = {
                "step_idx": step_idx,
                "t_int": t_int,
                "next_t": next_t,
                "gap": gap,
                "method": method,
                "x_t_tokens": x_t_list,
                "logits_x0": logits_x0[0].detach().cpu().numpy().tolist(),
                "p_x0": p_x0[0].detach().cpu().numpy().tolist(),
                "q_jump": q_jump.detach().cpu().numpy().tolist(),
                "p_xprev": p_xprev[0].detach().cpu().numpy().tolist(),
                "new_tokens": new_tokens[0].tolist(),
                "y_true": log_label[0].tolist() if log_label is not None else None,
            }
            log_steps.append(row)

    if last_logits_x0 is None:
        last_logits_x0 = torch.zeros(
            (1, x_tgt.size(1), V), device=device, dtype=torch.float32
        )
    out = last_logits_x0.squeeze(0)
    if out.ndim == 2 and out.size(-1) > num_classes:
        out = out[..., :num_classes]
    # Write JSON log if requested
    if log_steps is not None and log_json_path is not None:
        import json

        with open(log_json_path, "w", encoding="utf-8") as f:
            json.dump(log_steps, f, indent=2)
    return out
