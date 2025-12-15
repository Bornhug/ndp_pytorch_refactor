# diffusion_categorical_torch.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Callable

import math
import torch
import torch.nn.functional as F

try:
    from config import DiffusionConfig
except ImportError:
    DiffusionConfig = None  # fallback if config is not importable


LossType = Literal["kl", "hybrid", "cross_entropy_x_start"]
ModelPrediction = Literal["x_start", "x_prev"]
TransitionMatType = Literal["uniform", "absorbing"]
ModelOutput = Literal["logits"]  # matches the JAX file comment

@dataclass(frozen=True)
class BetaSpec:
    type: Literal["linear", "cosine", "jsd"] = "cosine"
    start: float = 3e-4
    stop: float = 0.5
    num_timesteps: int = 1000


def get_diffusion_betas(spec: BetaSpec) -> torch.Tensor:
    """
    Faithful port of get_diffusion_betas() in the Google Research file:
      - linear: Ho et al. DDPM
      - cosine: Hoogeboom et al. schedule (for uniform)
      - jsd: 1/T, 1/(T-1), ..., 1 (for absorbing)
    """
    T = spec.num_timesteps
    if spec.type == "linear":
        return torch.linspace(spec.start, spec.stop, T, dtype=torch.float32)
    if spec.type == "cosine":
        steps = torch.arange(T + 1, dtype=torch.float32) / T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2)
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.minimum(betas, torch.tensor(0.999, dtype=torch.float32))
    if spec.type == "jsd":
        return 1.0 / torch.linspace(T, 1.0, T, dtype=torch.float32)
    raise NotImplementedError(spec.type)


def _log_min_exp(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Stable log(exp(a) - exp(b)) assuming a >= b (the usage in the JAX file satisfies this).
    """
    return a + torch.log(torch.clamp(1.0 - torch.exp(b - a), min=eps))


class CategoricalDiffusionTorch:
    """
    Faithful PyTorch port of google-research/d3pm/images/diffusion_categorical.py (core logic).

    Time convention matches the JAX file:
      noisy data: x_0, ..., x_{T-1}
      original data: x_start (a.k.a. x_{-1} in that convention)
    """

    def __init__(
        self,
        *,
        betas: Optional[torch.Tensor] = None,  # float32 [T]
        beta_spec: Optional[BetaSpec] = None,
        model_prediction: ModelPrediction = "x_start",
        model_output: ModelOutput = "logits",
        transition_mat_type: TransitionMatType = "uniform",
        transition_bands: Optional[int] = None,
        loss_type: LossType = "hybrid",
        hybrid_coeff: float = 1.0,
        num_bits: int = 2,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cpu")

        if betas is None:
            beta_spec = beta_spec or BetaSpec()
            betas = get_diffusion_betas(beta_spec)

        if not (betas.dtype == torch.float32 and betas.ndim == 1):
            betas = betas.to(dtype=torch.float32).flatten()

        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError("betas must be in (0, 1].")

        self.device = device
        self.eps = float(eps)

        self.betas = betas.to(device=device)  # float32
        self.num_timesteps = int(self.betas.shape[0])

        self.model_prediction = model_prediction          # 'x_start' or 'x_prev'
        self.model_output = model_output                  # 'logits' or 'logistic_pars'
        self.loss_type = loss_type                        # 'kl', 'hybrid', 'cross_entropy_x_start'
        self.hybrid_coeff = float(hybrid_coeff)

        # num_bits is treated as vocabulary size (number of classes).
        self.num_bits = int(num_bits)
        self.num_classes = int(num_bits)

        self.transition_mat_type = transition_mat_type    # 'uniform'|'absorbing'
        self.transition_bands = transition_bands

        # Precompute q(x_t | x_{t-1}) matrices and cumulative q(x_t | x_start)
        q_one_step = []
        for t in range(self.num_timesteps):
            if transition_mat_type == "uniform":
                q_one_step.append(self._get_transition_mat_uniform(t))
            elif transition_mat_type == "absorbing":
                q_one_step.append(self._get_absorbing_transition_mat(t))
            else:
                raise ValueError(f"unknown transition_mat_type: {transition_mat_type}")
        self.q_onestep_mats = torch.stack(q_one_step, dim=0)  # [T, K, K], float32

        # q_mats[t] = Q_0 Q_1 ... Q_t  (same multiplication order as JAX tensordot)
        q_mat_t = self.q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            q_mat_t = q_mat_t @ self.q_onestep_mats[t]
            q_mats.append(q_mat_t)
        self.q_mats = torch.stack(q_mats, dim=0)  # [T, K, K], float32
        self.transpose_q_onestep_mats = self.q_onestep_mats.transpose(1, 2).contiguous()

    # -------------------------
    # Transition matrices
    # -------------------------

    def _get_full_transition_mat_uniform(self, t: int) -> torch.Tensor:
        K = self.num_classes
        beta_t = float(self.betas[t].item())
        mat = torch.full((K, K), fill_value=beta_t / K, dtype=self.betas.dtype, device=self.device)
        diag_val = 1.0 - beta_t * (K - 1.0) / K
        mat.fill_diagonal_(diag_val)
        return mat

    def _get_absorbing_transition_mat(self, t: int) -> torch.Tensor:
        """
        Transition with an absorbing state at index num_classes//2:
          q(x_t = i | x_{t-1} = j) = (1-β_t) if i==j, β_t if i==abs_state, else 0
        """
        K = self.num_classes
        beta_t = float(self.betas[t].item())
        mat = torch.diag(torch.full((K,), 1.0 - beta_t, dtype=self.betas.dtype, device=self.device))
        abs_idx = K // 2
        mat[:, abs_idx] += beta_t
        return mat

    def _get_transition_mat_uniform(self, t: int) -> torch.Tensor:
        """
        Matches the JAX _get_transition_mat:
          - if transition_bands is None => full uniform
          - else banded uniform off-diagonal with row-normalized diagonal
        """
        K = self.num_classes
        if self.transition_bands is None:
            return self._get_full_transition_mat_uniform(t)



    # -------------------------
    # Forward process: q(x_t | x_start)
    # -------------------------

    def q_probs(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_start: int64 [B, ...]
        t: int64 [B]
        returns probs: float32 [B, ..., K]
        """
        B = x_start.shape[0]
        K = self.num_classes
        t = t.to(device=self.device, dtype=torch.long)
        x_start = x_start.to(device=self.device, dtype=torch.long)

        # Bounds checks to catch invalid indices early
        if x_start.min() < 0 or x_start.max() >= K:
            raise ValueError(f"x_start out of bounds: min={x_start.min().item()}, max={x_start.max().item()}, V={K}")
        if t.min() < 0 or t.max() >= self.num_timesteps:
            raise ValueError(f"t out of bounds: min={t.min().item()}, max={t.max().item()}, T={self.num_timesteps}")

        q = self.q_mats[t]  # [B, K, K] float32
        flat = x_start.reshape(B, -1)  # [B, N]

        batch_idx = torch.arange(B, device=self.device)[:, None].expand_as(flat)  # [B,N]
        rows = q[batch_idx, flat]  # [B, N, K]  (row = x_start value)
        return rows.reshape(*x_start.shape, K)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Faithful to JAX: sample via Gumbel-max on logits = log(q_probs).
        noise must be uniform in [0,1) of shape x_start.shape + (K,).
        """
        probs = self.q_probs(x_start, t)  # [B,...,K] float32
        logits = torch.log(probs + self.eps)

        if noise is None:
            noise = torch.rand(*probs.shape, device=self.device, dtype=self.betas.dtype, generator=generator)
        else:
            noise = noise.to(device=self.device, dtype=self.betas.dtype)

        noise = noise.clamp(min=torch.finfo(noise.dtype).tiny, max=1.0)
        gumbel = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel, dim=-1).to(dtype=torch.long)


    def _model_logits_xstart(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns logits over x_start: [B, ..., K]
        """
        out = model(x_t, t)  # either logits or logistic_pars
        if self.model_output == "logits":
            return out
        raise ValueError(self.model_output)

    def _model_logits_xprev(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns logits over x_{t-1}: [B, ..., K] (for t==0 this corresponds to x_start).
        """
        out = model(x_t, t)
        if self.model_output == "logits":
            return out
        raise ValueError(self.model_output)

    # -------------------------
    # Reverse distribution construction
    # -------------------------

    def _p_xprev_from_xstart_probs(
        self,
        *,
        p_xstart: torch.Tensor,   # float32 [B, N, K]
        x_t: torch.Tensor,        # long   [B, N]
        t: torch.Tensor,          # long   [B]
    ) -> torch.Tensor:
        """
        Implements Eq-style marginalization for x_start-parameterization:
          p_theta(x_{t-1} | x_t) = sum_{x_start} q(x_{t-1} | x_t, x_start) * p_theta(x_start | x_t)

        Vectorized derivation (per-sample b):
          Let A = q_mats[t-1]  (KxK), B = q_onestep[t] (KxK), Q = q_mats[t] = A@B.
          For each observed j (= x_t), define denom_cols = Q[:, j] (KxN).
          C = p_xstart^T / denom_cols  (KxN).
          S = A^T @ C  (KxN).
          B_cols = B[:, j] (KxN).
          p_xprev^T ∝ B_cols * S.
        """
        Bsz, N, K = p_xstart.shape
        device = p_xstart.device

        # For t==0, x_{t-1} is x_start: the mapping collapses; we just return p_xstart.
        if (t == 0).all():
            return p_xstart

        A = self.q_mats[(t - 1).clamp(min=0)].to(device=device)          # [B,K,K] float32
        Bmat = self.q_onestep_mats[t].to(device=device)                 # [B,K,K] float32
        Q = self.q_mats[t].to(device=device)                            # [B,K,K] float32

        # Gather columns j from Q and B for each token
        j = x_t  # [B,N]
        j_exp_KN = j.unsqueeze(1).expand(Bsz, K, N)  # [B,K,N]

        denom_cols = Q.gather(dim=2, index=j_exp_KN) + self.eps          # [B,K,N]  (k, n)
        B_cols = Bmat.gather(dim=2, index=j_exp_KN)                      # [B,K,N]

        pT = p_xstart.transpose(1, 2).contiguous()                       # [B,K,N]
        C = pT / denom_cols                                              # [B,K,N]
        S = torch.matmul(A.transpose(1, 2), C)                           # [B,K,N]

        p_xprev_T = B_cols * S                                           # [B,K,N]
        p_xprev_T = p_xprev_T / (p_xprev_T.sum(dim=1, keepdim=True) + self.eps)
        return p_xprev_T.transpose(1, 2).contiguous()                    # [B,N,K]

    # -------------------------
    # Training losses (kl / hybrid / cross_entropy_x_start)
    # -------------------------

    @torch.no_grad()
    def _q_posterior(
        self,
        *,
        x_start: torch.Tensor,  # [B,N] long
        x_t: torch.Tensor,      # [B,N] long
        t: torch.Tensor,        # [B]   long
    ) -> torch.Tensor:
        """
        True posterior q(x_{t-1} | x_t, x_start) for t>0 (and delta at t==0).

        For t>0 and per token:
          q(i | j, k) = q_onestep[t][i,j] * q_mats[t-1][k,i] / q_mats[t][k,j]
        """
        Bsz, N = x_start.shape
        K = self.num_classes
        device = x_start.device

        out = torch.zeros((Bsz, N, K), device=device, dtype=self.betas.dtype)

        # t==0 => delta at x_start
        mask0 = (t == 0)
        if mask0.any():
            b0 = torch.where(mask0)[0]
            k0 = x_start[b0]  # [b0,N]
            out[b0].scatter_(dim=2, index=k0.unsqueeze(-1), value=1.0)
            # done for those batch items

        mask = ~mask0
        if mask.any():
            b = torch.where(mask)[0]
            tb = t[b]
            xs = x_start[b]
            xt = x_t[b]

            A = self.q_mats[tb - 1].to(device=device)     # [b,K,K]
            Bmat = self.q_onestep_mats[tb].to(device=device)  # [b,K,K]
            Q = self.q_mats[tb].to(device=device)         # [b,K,K]

            # A_rows: q(x_{t-1}=i | x_start=k) = A[k,i]
            A_rows = A[torch.arange(b.numel(), device=device)[:, None], xs]  # [b,N,K]

            # B_cols: q(x_t=j | x_{t-1}=i) = B[i,j]
            j_exp = xt.unsqueeze(1)  # [b,1,N]
            B_cols = Bmat.gather(dim=2, index=j_exp.expand(-1, K, -1)).transpose(1, 2)  # [b,N,K]

            denom = Q[torch.arange(b.numel(), device=device)[:, None], xs, xt] + self.eps  # [b,N]
            qpost = (A_rows * B_cols) / denom.unsqueeze(-1)  # [b,N,K]
            qpost = qpost / (qpost.sum(dim=-1, keepdim=True) + self.eps)
            out[b] = qpost

        return out

    def training_loss(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_start: torch.Tensor,         # int64 [B,...]
        t: torch.Tensor,               # int64 [B]
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns per-batch-item loss: [B]
        loss_type in {'kl','hybrid','cross_entropy_x_start'} like the JAX file. :contentReference[oaicite:2]{index=2}
        """
        x_start = x_start.to(self.device, dtype=torch.long)
        t = t.to(self.device, dtype=torch.long)

        x_t = self.q_sample(x_start, t, generator=generator)  # [B,...]
        Bsz = x_start.shape[0]
        K = self.num_classes
        flat_start = x_start.view(Bsz, -1)  # [B,N]
        flat_xt = x_t.view(Bsz, -1)         # [B,N]
        N = flat_start.shape[1]

        vb = torch.zeros((Bsz,), device=self.device, dtype=self.betas.dtype)
        ce = torch.zeros((Bsz,), device=self.device, dtype=self.betas.dtype)

        # --- auxiliary CE on x_start (only meaningful if predicting x_start)
        if self.loss_type in ("cross_entropy_x_start", "hybrid"):
            if self.model_prediction != "x_start":
                raise ValueError("cross_entropy_x_start/hybrid expects model_prediction='x_start'.")
            logits_xstart = self._model_logits_xstart(model, x_t, t)  # [B,...,K]
            logp = F.log_softmax(logits_xstart, dim=-1)  # [B,...,K]
            ce_token = F.nll_loss(
                logp.view(-1, K),
                x_start.view(-1),
                reduction="none"
            ).view(Bsz, -1).mean(dim=1)
            ce = ce_token

        # --- KL (variational bound term per sampled t, with t==0 reducing to CE)
        if self.loss_type in ("kl", "hybrid"):
            if self.model_prediction == "x_prev":
                logits_xprev = self._model_logits_xprev(model, x_t, t).to(self.betas.dtype)  # [B,...,K]
                logp_xprev = F.log_softmax(logits_xprev, dim=-1).view(Bsz, N, K)
            else:
                # model predicts x_start -> map to p(xprev|xt)
                logits_xstart = self._model_logits_xstart(model, x_t, t).to(self.betas.dtype)  # [B,...,K]
                p_xstart = torch.softmax(logits_xstart, dim=-1).view(Bsz, N, K)
                p_xprev = self._p_xprev_from_xstart_probs(p_xstart=p_xstart, x_t=flat_xt, t=t)
                logp_xprev = torch.log(p_xprev + self.eps)

            with torch.no_grad():
                qpost = self._q_posterior(x_start=flat_start, x_t=flat_xt, t=t)  # [B,N,K]
                log_qpost = torch.log(qpost + self.eps)

            kl_token = (qpost * (log_qpost - logp_xprev)).sum(dim=-1)  # [B,N]
            vb = kl_token.mean(dim=1)

        if self.loss_type == "kl":
            total = vb
            ce = torch.zeros_like(vb)
        elif self.loss_type == "cross_entropy_x_start":
            total = ce
            vb = torch.zeros_like(ce)
        else:  # hybrid
            total = vb + self.hybrid_coeff * ce
        return total, vb, ce

    # -------------------------
    # Sampling (reverse process)
    # -------------------------

    @torch.no_grad()
    def p_sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_t: torch.Tensor,   # [B,...] long
        t: torch.Tensor,     # [B] long
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from p_theta(x_{t-1}|x_t).
        """
        x_t = x_t.to(self.device, dtype=torch.long)
        t = t.to(self.device, dtype=torch.long)

        Bsz = x_t.shape[0]
        K = self.num_classes
        flat_xt = x_t.view(Bsz, -1)
        N = flat_xt.shape[1]

        if self.model_prediction == "x_prev":
            logits = self._model_logits_xprev(model, x_t, t).to(self.betas.dtype)
            p = torch.softmax(logits, dim=-1).view(Bsz, N, K)
        else:
            logits_xstart = self._model_logits_xstart(model, x_t, t).to(self.betas.dtype)
            p_xstart = torch.softmax(logits_xstart, dim=-1).view(Bsz, N, K)
            p = self._p_xprev_from_xstart_probs(p_xstart=p_xstart, x_t=flat_xt, t=t)

        # sample categorical
        # (use multinomial for practicality; if you want exact JAX-style, use gumbel-max)
        out = torch.multinomial(p.view(-1, K), num_samples=1, generator=generator).view(Bsz, N)
        return out.view_as(x_t)

    @torch.no_grad()
    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, ...],  # (B, H, W, C) or (B, L) etc.
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate a full sample by starting from the prior and stepping t=T-1..0.
        Prior choice:
          - uniform/gaussian: uniform over states
          - absorbing: start at the absorbing state (K//2), matching that corruption’s attractor.
        """
        Bsz = shape[0]
        K = self.num_classes

        if self.transition_mat_type == "absorbing":
            x = torch.full(shape, fill_value=K // 2, device=self.device, dtype=torch.long)
        else:
            x = torch.randint(0, K, shape, device=self.device, dtype=torch.long, generator=generator)

        for tt in reversed(range(self.num_timesteps)):
            t = torch.full((Bsz,), tt, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, generator=generator)
        return x


# ----------------------------------------------------------------------
# D3PM-compatible wrapper (tabular signatures; RePaint conditional sampling)
# ----------------------------------------------------------------------

@dataclass
class D3PMSchedule:
    betas: torch.Tensor
    vocab_size: int
    device: torch.device

    @staticmethod
    def make_uniform(
        *,
        T: int | None,
        vocab_size: int,
        beta_start: Optional[float] = None,
        beta_end: Optional[float] = None,
        beta_type: Optional[Literal["cosine", "linear", "jsd"]] = None,
        device: torch.device | str = "cpu",
        dtype: Optional[torch.dtype] = None,
        transition_mat_type: TransitionMatType = "uniform",
    ) -> "D3PMSchedule":
        device_t = torch.device(device)

        # Prefer explicit args; fall back to DiffusionConfig if available; otherwise defaults.
        if DiffusionConfig is not None:
            cfg = DiffusionConfig()
            beta_start = cfg.beta_start if beta_start is None else beta_start
            beta_end = cfg.beta_end if beta_end is None else beta_end
            beta_type = cfg.schedule if beta_type is None else beta_type
            T = cfg.timesteps if T is None else T
            transition_mat_type = getattr(cfg, "transition_mat_type", transition_mat_type)

        # For absorbing, default to JSD unless explicitly overridden.
        if transition_mat_type == "absorbing":
            beta_type = "jsd" if beta_type is None else beta_type
        else:
            beta_type = "cosine" if beta_type is None else beta_type

        beta_start = 3e-4 if beta_start is None else beta_start
        beta_end = 0.5 if beta_end is None else beta_end
        T = 500 if T is None else T

        spec = BetaSpec(type=beta_type, start=beta_start, stop=beta_end, num_timesteps=T)
        betas = get_diffusion_betas(spec).to(device_t)
        if dtype is not None:
            betas = betas.to(dtype=dtype)
        return D3PMSchedule(betas=betas, vocab_size=int(vocab_size), device=device_t)


class D3PM:
    """
    Thin wrapper around CategoricalDiffusionTorch to keep the legacy D3PM API
    and the tabular model signature (t, y, x, mask).
    """

    def __init__(
        self,
        model: Callable[..., torch.Tensor],
        schedule: D3PMSchedule,
        *,
        loss_type: LossType = "hybrid",
        hybrid_coeff: float = 1.0,
        transition_mat_type: TransitionMatType = "uniform",
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.core = CategoricalDiffusionTorch(
            betas=schedule.betas,
            model_prediction="x_start",
            model_output="logits",
            transition_mat_type=transition_mat_type,
            transition_bands=None,
            loss_type=loss_type,
            hybrid_coeff=hybrid_coeff,
            num_bits=schedule.vocab_size,  # treated as vocab size
            device=schedule.device,
        )

    @property
    def T(self) -> int:
        return self.core.num_timesteps

    @property
    def V(self) -> int:
        return self.core.num_classes

    def loss(
        self,
        y_tokens: torch.Tensor,   # [B,N_tgt]
        x_target: torch.Tensor,   # [B,N_tgt,F]
        *,
        t: Optional[torch.Tensor] = None,
        x_context: torch.Tensor | None = None,
        y_context: torch.Tensor | None = None,
        mask_target: torch.Tensor | None = None,
        mask_context: torch.Tensor | None = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Per-batch mean training loss using the configured loss_type (default hybrid).
        Only target tokens are diffused; context remains conditioning information.
        """
        B = y_tokens.shape[0]
        device = y_tokens.device
        if t is None:
            t = torch.randint(0, self.T, (B,), device=device, generator=generator)

        if mask_target is None:
            mask_target = torch.zeros_like(y_tokens, dtype=torch.float32)
        else:
            mask_target = mask_target.to(device=device, dtype=torch.float32)
            if mask_target.ndim == 1:
                mask_target = mask_target.unsqueeze(0)

        if mask_context is not None:
            mask_context = mask_context.to(device=device, dtype=torch.float32)
            if mask_context.ndim == 1:
                mask_context = mask_context.unsqueeze(0)

        if x_context is not None:
            x_context = x_context.to(device=device)
        if y_context is not None:
            y_context = y_context.to(device=device)

        def model_fn(x_tokens: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            y_onehot = torch.nn.functional.one_hot(x_tokens, num_classes=self.V).to(dtype=x_target.dtype)
            t_float = t_in.to(dtype=x_target.dtype)
            return self.model(
                x_target=x_target,
                y_target=y_onehot,
                t=t_float,
                mask_target=mask_target,
                x_context=x_context,
                y_context=y_context,
                mask_context=mask_context,
            )

        total, vb, ce = self.core.training_loss(model_fn, y_tokens, t, generator=generator)
        if self.core.loss_type == "hybrid":
            return total.mean(), vb.mean(), ce.mean()
        return total.mean()

    @torch.no_grad()
    def sample(
        self,
        x_feats: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Unconditional ancestral sampling for targets matching x_feats shape.
        """
        shape = x_feats.shape[:2]  # [B,N]
        def model_fn(x_tokens: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            y_onehot = torch.nn.functional.one_hot(x_tokens, num_classes=self.V).to(dtype=x_feats.dtype)
            t_float = t_in.to(dtype=x_feats.dtype)
            mask_tgt = mask
            if mask_tgt is None:
                mask_tgt = torch.zeros(shape, device=x_feats.device, dtype=torch.float32)
            return self.model(
                x_target=x_feats,
                y_target=y_onehot,
                t=t_float,
                mask_target=mask_tgt,
                x_context=None,
                y_context=None,
                mask_context=None,
            )
        return self.core.sample(model_fn, shape, generator=generator)

    @torch.no_grad()
    def conditional_sample(
        self,
        x_target: torch.Tensor,      # [B,N,F]
        x_context: torch.Tensor,     # [B,M,F]
        y_context: torch.Tensor,     # [B,M,C] one-hot
        *,
        mask_target: torch.Tensor | None = None,
        mask_context: torch.Tensor | None = None,
        num_sample_steps: Optional[int] = None,
        num_inner_steps: int = 5,
        progress: bool = False,
        progress_desc: str | None = None,
    ) -> torch.Tensor:
        """
        RePaint-style conditional sampling is not supported for the conditional NDP.
        Use direct logits from the model instead.
        """
        raise NotImplementedError(
            "conditional_sample is disabled for the conditional NDP. "
            "Call the model directly with (x_context, y_context, x_target, y_target, t)."
        )


# ----------------------------------------------------------------------
# Module-level wrappers (keep legacy names)
# ----------------------------------------------------------------------

def loss(
    process: D3PM,
    model: Callable,
    batch,
    key: torch.Generator,
    *,
    num_timesteps: int,
    loss_type: str | None = None,
) -> torch.Tensor:
    """
    Wrapper matching the old signature; uses process/core loss internally.
    Expects batch.y_* to be one-hot; converts to token indices.
    """
    _ = loss_type  # loss_type handled by process configuration
    mask_target = batch.mask_target
    if mask_target is None:
        mask_target = torch.zeros(batch.y_target.shape[:2], device=batch.y_target.device)
    mask_context = batch.mask_context
    if mask_context is None and batch.y_context is not None:
        mask_context = torch.zeros(batch.y_context.shape[:2], device=batch.y_context.device)

    y_tokens = batch.y_target.argmax(dim=-1)  # [B,N_tgt]
    return process.loss(
        y_tokens=y_tokens,
        x_target=batch.x_target,
        t=None,
        x_context=batch.x_context,
        y_context=batch.y_context,
        mask_target=mask_target,
        mask_context=mask_context,
        generator=key,
    )


# Expose module-level conditional sampling (RePaint) for backward compatibility.
def conditional_sample(
    model: Callable,
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
    return process.conditional_sample(
        x_target=x_target,
        x_context=x_context,
        y_context=y_context,
        mask_target=mask_target,
        mask_context=mask_context,
        num_sample_steps=num_sample_steps,
        num_inner_steps=num_inner_steps,
        progress=progress,
        progress_desc=progress_desc,
    )


__all__ = [
    "BetaSpec",
    "get_diffusion_betas",
    "CategoricalDiffusionTorch",
    "D3PM",
    "D3PMSchedule",
    "loss",
    "conditional_sample",
]
