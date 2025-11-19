"""
OOP helpers to compute GP log likelihoods for evaluating NDP samples.
Global configuration (dataset name, input_dim, jitter) is carried as object state.
"""

from __future__ import annotations

import math
from typing import List, Union

import gpytorch
import numpy as np
import torch
from tqdm import tqdm

from data import _DATASET_FACTORIES, _load_npz, GPFunctionalDistribution, BaselineGP


def stable_gaussian_log_prob(
    mean: torch.Tensor,
    cov: torch.Tensor,
    y: torch.Tensor,
    base_jitter: float = 1e-6,
    max_tries: int = 5,
) -> torch.Tensor:
    """
    Numerically stable log N(y | mean, cov) using Cholesky with adaptive jitter.
    """
    n = cov.shape[0]
    eye = torch.eye(n, device=cov.device, dtype=cov.dtype)

    jitter = base_jitter
    L = None
    for _ in range(max_tries):
        try:
            cov_sym = 0.5 * (cov + cov.T)
            L = torch.linalg.cholesky(cov_sym + jitter * eye)
            break
        except RuntimeError:
            jitter *= 10.0

    if L is None:
        raise RuntimeError("Cholesky failed even with large jitter in stable_gaussian_log_prob.")

    diff = (y - mean).reshape(-1)  # [n]
    solve = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(-1)
    maha = (diff * solve).sum()

    logdet = 2.0 * torch.log(torch.diag(L)).sum()
    dim = mean.numel()
    log2pi = dim * math.log(2.0 * math.pi)

    return -0.5 * (maha + logdet + log2pi)


class BaseLikelihood:
    """
    Holds shared configuration (dataset name, input_dim, jitter) and GP builder.
    """
    def __init__(self, npz_path: str, name: str = "matern", input_dim: int = 1, jitter: float = 1e-6):
        if name not in _DATASET_FACTORIES:
            raise ValueError(f"Unknown dataset '{name}'.")
        self.name = name
        self.input_dim = input_dim
        self.jitter = jitter
        self.gp = self._build_gp()

        self.x_target, self.y_target, self.x_context, \
        self.y_context, self.mask_target, self.mask_context = self._build_data(npz_path)

    def _build_gp(self) -> GPFunctionalDistribution:
        active_dims = list(range(self.input_dim))
        fd = _DATASET_FACTORIES[self.name](active_dims)
        if not isinstance(fd, GPFunctionalDistribution):
            fd = GPFunctionalDistribution(fd.kernel, fd.params)
        return fd

    def _build_data(self, npz_path: str):
        data = _load_npz(npz_path)
        x_target = data.x_target
        y_target = data.y_target
        x_context = data.x_context
        y_context = data.y_context
        mask_target = data.mask_target
        mask_context = data.mask_context
        return x_target, y_target, x_context, y_context, mask_target, mask_context


class GPLikelihoodEvaluator(BaseLikelihood):
    """
    Methods:
      - method1_log_likelihood: use NDP-sampled y_target to score under GP.
      - method2_log_likelihood: empirical Gaussian over multiple NDP samples, score true targets.
    """

    def _summarize_ll(self, ll_vec: torch.Tensor):
        if ll_vec.numel() == 0:
            return float("nan"), float("nan"), float("nan")
        mean = ll_vec.mean().item()
        if ll_vec.numel() > 1:
            std = ll_vec.std(unbiased=True).item()
        else:
            std = 0.0
        stderr = std / math.sqrt(ll_vec.numel()) if ll_vec.numel() > 0 else float("nan")
        return mean, std, stderr

    def _print_stats(self, label: str, ll_vec: torch.Tensor):
        mean, std, stderr = self._summarize_ll(ll_vec)
        print(f"{label}: mean={mean:.4f}, std={std:.4f}, stderr={stderr:.4f}")
        return mean, std, stderr

    def compute_baselineGP(
        self
    ) -> torch.Tensor:
        '''
        Baseline for Method 1 and 2:
            Sample from Real GP posterior, there are 2 ways to do:
                1. Directly load the interpolation dataset, which itself is sampled from Real GP posterior.
                2. Given the contexts and x_target, fit GP to obtain y_target
            We use the Second.
            then evaluate under GP log likelihood.
        '''
        fd: BaselineGP
        if isinstance(self.gp, BaselineGP):
            fd = self.gp
        else:
            fd = BaselineGP(self.gp.kernel, self.gp.params)

        B = self.x_target.shape[0]
        ll_per_example: list[torch.Tensor] = []
        for b in tqdm(range(B), desc="Baseline GP", leave=False):
            mask_t = self.mask_target[b] if self.mask_target is not None else None
            if mask_t is not None:
                keep_t = mask_t == 0
                num_t = int(keep_t.sum().item())
                if num_t == 0:
                    continue
            else:
                num_t = self.x_target.shape[1]

            ll = fd.log_likelihood(
                self.x_context[b],
                self.y_context[b],
                self.x_target[b],
                self.y_target[b],
                mask_context=self.mask_context[b] if self.mask_context is not None else None,
                mask_target=mask_t,
                jitter=self.jitter,
            )
            ll_per_example.append(torch.as_tensor(ll / num_t, dtype=torch.float32))

        if not ll_per_example:
            return torch.empty(0, dtype=torch.float32)
        ll_vec = torch.stack(ll_per_example)
        self._print_stats("baselineGP", ll_vec)
        return ll_vec



    def method1_log_likelihood(
        self,
        y_target_ndp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Method 1 (NDP → GP):
            - Load x_context, y_context, x_target from `npz_path`.
            - Use `y_target_ndp` as targets sampled from the NDP model.
            - Evaluate these targets under the GP defined by the dataset factory and return the
              mean **per-target** log likelihood (average over targets within each batch, then
              over batches).
        """
        y_target = y_target_ndp

        samples = y_target
        if samples.dim() == 4:
            samples = samples.squeeze(-1)
        if samples.dim() == 3 and samples.shape[0] == self.x_target.shape[0]:  # [B,N] -> pretend S=1
            samples = samples.unsqueeze(0)
        elif samples.dim() == 2:
            samples = samples.unsqueeze(0)

        S, B, N = samples.shape

        ll_per_example: list[torch.Tensor] = []
        for b in tqdm(range(B), desc="Method 1 GP ll", leave=False):
            mask_t = self.mask_target[b] if self.mask_target is not None else None
            if mask_t is not None:
                keep_t = mask_t == 0
                num_t = int(keep_t.sum().item())
                if num_t == 0:
                    continue
            else:
                num_t = N

            ll_samples: list[float] = []
            for s in range(S):
                y_s = samples[s, b]  # [N]
                y_s = y_s.unsqueeze(-1)
                ll = self.gp.compute_log_likelihood(
                    self.x_context[b],
                    self.y_context[b],
                    self.x_target[b],
                    y_s,
                    mask_train=self.mask_context[b] if self.mask_context is not None else None,
                    mask_test=mask_t,
                    jitter=self.jitter,
                )
                ll_samples.append(ll / num_t)

            ll_per_example.append(torch.as_tensor(ll_samples, dtype=torch.float32).mean())

        if not ll_per_example:
            return torch.empty(0, dtype=torch.float32)
        ll_vec = torch.stack(ll_per_example)
        self._print_stats("method1", ll_vec)
        return ll_vec



    def method2_log_likelihood(
        self,
        y_target_ndp: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Method 2:
            Sample from Real GP posterior (using the interpolation set, which contains conditional GP samples
                from the same posterior distribution as in the training set),
            Then evaluate under "constructed" GP log likelihood.

            The construction is done by:
                1. y_target(s)  ∼ qθ( · | XT , C), where qθ is NDP predictive distribution
                2. Utilising y_target(s), the empirical mean and covariance are computed
                3. Fit a multivariate Gaussian using the empirical mean and covariance.
                4. Compute GP log likelihood, where the posterior mean is the empirical mean and
                    the posterior covariance is the empirical covariance.
        '''
        # Normalize input to torch.Tensor of shape [S,B,N]
        samples = y_target_ndp
        if samples.dim() == 4:
            samples = samples.squeeze(-1)
        if samples.dim() == 3 and samples.shape[0] == self.x_target.shape[0]:  # [B,N] -> pretend S=1
            samples = samples.unsqueeze(0)
        elif samples.dim() == 2:
            samples = samples.unsqueeze(0)

        S, B, N = samples.shape
        y_true_full = self.y_target.squeeze(-1)  # [B,N]

        ll_per_example: list[torch.Tensor] = []
        for b in tqdm(range(B), desc="Method 2 GP ll", leave=False):
            mask_t = self.mask_target[b] if self.mask_target is not None else None
            if mask_t is not None:
                keep_t = mask_t == 0
                num_t = int(keep_t.sum().item())
                if num_t == 0:
                    continue
                s_b = samples[:, b, :][:, keep_t]  # [S,num_t]
                y_true = y_true_full[b, keep_t]    # [num_t]
            else:
                num_t = N
                s_b = samples[:, b, :]             # [S,N]
                y_true = y_true_full[b]            # [N]

            mean = s_b.mean(dim=0)  # [num_t]
            if S > 1:
                centered = s_b - mean.unsqueeze(0)
                cov = torch.einsum("sn,sm->nm", centered, centered) / (S - 1)
            else:
                cov = torch.zeros(num_t, num_t, device=s_b.device, dtype=s_b.dtype)
            cov = cov + torch.eye(num_t, device=s_b.device, dtype=s_b.dtype) * self.jitter

            ll = stable_gaussian_log_prob(mean, cov, y_true, base_jitter=self.jitter)
            ll_per_example.append(ll / num_t)

        if not ll_per_example:
            return torch.empty(0, dtype=torch.float32)
        ll_vec = torch.stack(ll_per_example)
        self._print_stats("method2", ll_vec)
        return ll_vec



    '''
    Method 3:
        Sample from Real GP posterior.
        Evaluate under score matching likelihood.
    '''
