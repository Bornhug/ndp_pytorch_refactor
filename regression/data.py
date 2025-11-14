# ===== data.py =====
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import math
import torch
from torch import Tensor
import inspect
import gpytorch

from neural_diffusion_processes.types import Batch



__all__ = [
    "DATASETS",
    "TASKS",
    "get_batch",
    "_DATASET_CONFIGS",
    "_TASK_CONFIGS",
]


# ---------------- Small distributions ------------------

@dataclass
class UniformDiscrete:
    low: int
    high: int
    def sample(self, shape: Tuple[int, ...], *, g: torch.Generator) -> Tensor:
        if self.low == self.high:
            return torch.full(shape, self.low, dtype=torch.int64)
        return torch.randint(self.low, self.high + 1, shape, generator=g)


# ---------------- Public constants ---------------------

DATASETS = ["se", "matern", "sawtooth", "step"]
TASKS = ["training", "interpolation"]

# ---------------- Configs ------------------------------

@dataclass
class TaskConfig:
    x_context_dist: torch.distributions.Distribution
    x_target_dist: torch.distributions.Distribution

@dataclass
class DatasetConfig:
    max_input_dim: int
    is_gp: bool
    eval_num_target: UniformDiscrete = UniformDiscrete(50, 50)
    eval_num_context: UniformDiscrete = UniformDiscrete(1, 10)

_NOISE_VAR   = 0.05 ** 2
_KERNEL_VAR  = 1.0
_LENGTHSCALE = 0.25
_JITTER      = 1e-6

_DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "se":     DatasetConfig(max_input_dim=3, is_gp=True),
    "matern": DatasetConfig(max_input_dim=3, is_gp=True),
    "sawtooth": DatasetConfig(max_input_dim=1, is_gp=False),
    "step":     DatasetConfig(max_input_dim=1, is_gp=False),
}

_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "training": TaskConfig(
        x_context_dist=torch.distributions.Uniform(-2, 2),
        x_target_dist=torch.distributions.Uniform(-2, 2)
    ),

    "interpolation": TaskConfig(
        x_context_dist=torch.distributions.Uniform(-2, 2),
        x_target_dist=torch.distributions.Uniform(-2, 2)
    ),}

# Light defaults for training sizes (kept local to avoid changing config API)
TRAIN_NUM_TARGET  = UniformDiscrete(32, 64)
TRAIN_NUM_CONTEXT = UniformDiscrete(0, 32)

import abc
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping

import torch
import gpytorch


# ----------------------------- base class ---------------------------------
import math
from dataclasses import dataclass
from typing import Mapping

import torch
import gpytorch


# ---------------- base class ----------------

@dataclass
class FunctionalDistribution:
    is_data_naturally_normalized: bool = True
    normalize: bool = False


# ---------------- Exact GP model ----------------

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,      # [N, D]
        train_y: torch.Tensor,      # [N]
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: gpytorch.kernels.Kernel,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x: torch.Tensor):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# --------------- GP functional distribution ---------------

class GPFunctionalDistribution(FunctionalDistribution):
    def __init__(self, kernel: gpytorch.kernels.Kernel, params: Mapping):
        """
        kernel : base GPyTorch kernel (e.g. ScaleKernel(RBFKernel(...)))
                 with correct active_dims, but NOT hyperparameters set.
        params : {
            "kernel": {
                "lengthscale": float,
                "variance": float,
            },
            "noise_variance": float,
        }
        """
        super().__init__(is_data_naturally_normalized=True, normalize=False)
        self.kernel = kernel
        self.params = params

        # Likelihood used for the posterior
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = float(params["noise_variance"])

    def _apply_kernel_params(self):
        """Apply lengthscale / variance from self.params to self.kernel."""
        kparams = self.params["kernel"]
        lengthscale = float(kparams["lengthscale"])
        variance    = float(kparams["variance"])

        # Assumes ScaleKernel(base_kernel=RBF/Matern/etc.)
        self.kernel.base_kernel.lengthscale = lengthscale
        self.kernel.outputscale             = variance

    # -------- PRIOR: f(x) ~ GP(0, k), then y = f + ε --------
    @torch.no_grad()
    def sample_prior(
        self,
        key: torch.Generator,   # NOTE: GPyTorch MVN ignores generator; used only for eps
        x: torch.Tensor,        # [N, D]
    ) -> torch.Tensor:         # [N, 1]
        N, device, dtype = x.size(0), x.device, x.dtype

        # Sync kernel hyperparams from self.params
        self._apply_kernel_params()

        # Lazy covariance for GP prior
        K_lazy = self.kernel(x)                         # LazyTensor [N, N]
        mean   = torch.zeros(N, device=device, dtype=dtype)

        prior_dist = gpytorch.distributions.MultivariateNormal(mean, K_lazy)
        f = prior_dist.rsample().unsqueeze(-1)          # [N, 1]  (latent function)

        # Add observation noise: y = f + ε
        sigma2 = float(self.params["noise_variance"])
        eps    = torch.randn_like(f, generator=key) * math.sqrt(sigma2)
        return f + eps                                  # [N, 1]

    # -------- POSTERIOR: f_* | (X,y), X_* via ExactGP --------
    @torch.no_grad()
    def sample_posterior(
        self,
        x_train: torch.Tensor,   # [N, D]
        y_train: torch.Tensor,   # [N, 1] or [N]
        x_test: torch.Tensor,    # [M, D]
        *,
        return_obs: bool = False,  # False → f_*, True → y_* (with noise)
    ) -> torch.Tensor:           # [M, 1]
        device, dtype = x_train.device, x_train.dtype
        y_train_vec = y_train.view(-1)                  # [N]

        # Build ExactGP model that encodes prior + data
        model = ExactGPModel(x_train, y_train_vec, self.likelihood, self.kernel)
        model = model.to(device=device, dtype=dtype)

        model.eval()
        self.likelihood.eval()

        # Posterior over latent f_* at x_test
        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            latent_dist = model(x_test)                 # MultivariateNormal over f_*
            f_samples   = latent_dist.rsample().unsqueeze(-1)  # [M, 1]

            if not return_obs:
                return f_samples

            # Predictive distribution over y_* (includes noise)
            pred_dist = self.likelihood(latent_dist)
            y_samples = pred_dist.rsample().unsqueeze(-1)      # [M, 1]
            return y_samples



# ------------- factory registry (same pattern) -------------

DatasetFactory = Callable[[List[int]], FunctionalDistribution]
_DATASET_FACTORIES: Dict[str, DatasetFactory] = {}


def register_dataset_factory(name: str):
    def wrap(f: DatasetFactory):
        _DATASET_FACTORIES[name] = f
        return f
    return wrap


# ---------------- SE (RBF) dataset ----------------

@register_dataset_factory("se")
def _se_dataset_factory(active_dim: List[int]) -> FunctionalDistribution:
    input_dim = len(active_dim)
    factor    = math.sqrt(input_dim)

    # Kernel with correct active_dims; hyperparams live in params["kernel"]
    base_kernel = gpytorch.kernels.RBFKernel(active_dims=active_dim)
    kernel      = gpytorch.kernels.ScaleKernel(base_kernel)

    params = {
        "kernel": {
            "lengthscale": _LENGTHSCALE * factor,
            "variance":    _KERNEL_VAR,
        },
        "noise_variance": _NOISE_VAR,
    }
    return GPFunctionalDistribution(kernel, params)


# ------------- Matérn-5/2 dataset -------------

@register_dataset_factory("matern")
def _matern_dataset_factory(active_dim: List[int]) -> FunctionalDistribution:
    input_dim = len(active_dim)
    factor    = math.sqrt(input_dim)

    base_kernel = gpytorch.kernels.MaternKernel(
        nu=2.5,
        active_dims=active_dim,
    )
    kernel = gpytorch.kernels.ScaleKernel(base_kernel)

    params = {
        "kernel": {
            "lengthscale": _LENGTHSCALE * factor,
            "variance":    _KERNEL_VAR,
        },
        "noise_variance": _NOISE_VAR,
    }
    return GPFunctionalDistribution(kernel, params)




# ---------------- Main API ------------------------------

def get_batch(
    g: torch.Generator,
    *,
    batch_size: int,
    name: str,  # "se" | "matern" | "sawtooth" | "step"
    task: str,  # "training" | "interpolation"
    input_dim: int,
    gp_conditional_targets: bool = False,
    p_drop_ctx: float = 0.0,
) -> Batch:
    if name not in DATASETS: raise ValueError(f"Unknown dataset: {name}")
    if task not in TASKS:    raise ValueError(f"Unknown task: {task}")

    dataset = _DATASET_CONFIGS[name]
    if input_dim > dataset.max_input_dim:
        raise ValueError(f"input_dim {input_dim} > max_input_dim {dataset.max_input_dim} for {name}")

    if task == "training":
        min_n_target = dataset.eval_num_target.lower
        max_n_target = (
                dataset.eval_num_target.upper
                + dataset.eval_num_context.upper * input_dim
        )  # input_dim * num_context + num_target
        max_n_context = 0
    else:
        max_n_target = dataset.eval_num_target.upper
        max_n_context = dataset.eval_num_context.upper * input_dim


    # Inputs, sample from Uniform (-2,2)
    x_context = _TASK_CONFIGS[task].x_context_dist.sample(
        sample_shape=(batch_size, max_n_context, input_dim), generator=g
    )
    x_target = _TASK_CONFIGS[task].x_target_dist.sample(
        sample_shape=(batch_size, max_n_target, input_dim), generator=g
    )
    x_all = torch.cat([x_context, x_target], dim=1)


    if task == "training":
        num_keep_target = jax.random.randint(mkey, (), minval=min_n_target, maxval=max_n_target)
        mask_target = jnp.where(
            jnp.arange(max_n_target)[None, :] < num_keep_target,
            jnp.zeros_like(x_target)[..., 0],  # keep
            jnp.ones_like(x_target)[..., 0]  # ignore
        )
        mask_context = jnp.zeros_like(x_context[..., 0])
    elif task == "interpolation":
        num_keep_context = jax.random.randint(mkey, (), minval=1, maxval=max_n_context)
        mask_context = jnp.where(
            jnp.arange(max_n_context)[None, :] < num_keep_context,
            jnp.zeros_like(x_context)[..., 0],  # keep
            jnp.ones_like(x_context)[..., 0]  # ignore
        )
        mask_target = jnp.zeros_like(x_target[..., 0])


    active_dims = list(range(input_dim))
    dataset_factory = _DATASET_FACTORIES[name]
    function_distribution = dataset_factory(active_dims)


    if (gp_conditional_targets and _DATASET_CONFIGS[name].is_gp and hasattr(function_distribution, "kernel_fn")) is:


    # Build y_all, then split
    y_all = torch.stack([function_distribution.sample(x_all[b], g) for b in range(batch_size)], dim=0)
    y_context, y_target = y_all[:, :max_n_context], y_all[:, max_n_context:]

    # Optional GP conditional resampling of targets
    if gp_conditional_targets and _DATASET_CONFIGS[name].is_gp and hasattr(function_distribution, "kernel_fn"):
        K_fn = function_distribution.kernel_fn
        new_targets = []
        for b in range(batch_size):
            x_c = x_context[b]  # [M,D]
            y_c = y_context[b].squeeze(-1)  # [M]
            x_t = x_target[b]   # [N,D]
            M = x_c.size(0); N = x_t.size(0)
            if M == 0:
                Kxx = K_fn(x_t, x_t) + (_JITTER) * torch.eye(N, device=x_t.device, dtype=x_t.dtype)
                L = torch.linalg.cholesky(Kxx)
                z = torch.randn(N, 1, device=x_t.device, dtype=x_t.dtype, generator=g)
                y_samp = L @ z
                new_targets.append(y_samp)
                continue
            Kcc = K_fn(x_c, x_c) + (_NOISE_VAR + _JITTER) * torch.eye(M, device=x_c.device, dtype=x_c.dtype)
            Kxc = K_fn(x_t, x_c)
            Kxx = K_fn(x_t, x_t) + _JITTER * torch.eye(N, device=x_t.device, dtype=x_t.dtype)
            Lc = torch.linalg.cholesky(Kcc)
            alpha = torch.cholesky_solve(y_c.unsqueeze(-1), Lc)  # (M,1)
            mu = Kxc @ alpha                                     # (N,1)
            v = torch.cholesky_solve(Kxc.T, Lc)                  # (M,N)
            cov = Kxx - Kxc @ v                                  # (N,N)
            L = torch.linalg.cholesky(cov + _JITTER * torch.eye(N, device=cov.device, dtype=cov.dtype))
            z = torch.randn(N, 1, device=x_t.device, dtype=x_t.dtype, generator=g)
            y_samp = mu + L @ z
            new_targets.append(y_samp)
        y_target = torch.stack(new_targets, dim=0) # Posterior y_target

    # Masks (1 = missing/padded)
    mask_context = torch.zeros(batch_size, n_context, dtype=torch.float32)
    mask_target  = torch.zeros(batch_size, n_target,  dtype=torch.float32)

    # Classifier-free style drop AFTER targets are formed
    if p_drop_ctx > 0.0 and n_context > 0:
        drop = torch.rand(batch_size, generator=g) < p_drop_ctx
        for b in range(batch_size):
            if drop[b]:
                mask_context[b].fill_(1.0)

        # zero-out masked positions with proper broadcast
        mc = mask_context.bool()[..., None]         # [B,M,1]
        x_context = x_context.masked_fill(mc, 0.0)
        y_context = y_context.masked_fill(mc, 0.0)

    if device is not None:
        x_context, y_context, x_target, y_target = (
            x_context.to(device),
            y_context.to(device),
            x_target.to(device),
            y_target.to(device),
        )
        mask_context = mask_context.to(device)
        mask_target  = mask_target.to(device)

    return Batch(
        x_target=x_target,
        y_target=y_target,
        x_context=x_context,
        y_context=y_context,
        mask_target=mask_target,
        mask_context=mask_context,
    )




class Sawtooth(FunctionalDistribution):
    A = 1.0
    K_max = 20
    mean = 0.5
    variance = 0.07965
    def sample(self, x: Tensor, g: torch.Generator) -> Tensor:
        # Use first dim only; return (N,1)
        x1 = x[..., 0:1]
        f = 3.0 + 2.0 * torch.rand((), generator=g, device=x.device, dtype=x.dtype)
        s = -5.0 + 10.0 * torch.rand((), generator=g, device=x.device, dtype=x.dtype)
        ks = torch.arange(1, self.K_max + 1, dtype=x.dtype, device=x.device)[None, :]
        vals = (-1.0) ** ks * torch.sin(2 * math.pi * ks * f * (x1 - s)) / ks
        k = torch.randint(10, self.K_max + 1, (), generator=g)
        mask = (ks < k).float()
        fs = self.A / 2 + self.A / math.pi * (vals * mask).sum(dim=1, keepdim=True)
        fs = fs - self.mean
        return fs  # (N,1)

@register_dataset_factory("sawtooth")
def _sawtooth_dataset_factory(*args):
    return Sawtooth()

class Step(FunctionalDistribution):
    def sample(self, x: Tensor, g: torch.Generator) -> Tensor:
        # Use first dim only; return (N,1)
        x1 = x[..., 0:1]
        s = -2.0 + 4.0 * torch.rand((), generator=g, device=x.device, dtype=x.dtype)
        return torch.where(x1 < s, torch.zeros_like(x1), torch.ones_like(x1))  # (N,1)

@register_dataset_factory("step")
def _step_dataset_factory(*args):
    return Step()


