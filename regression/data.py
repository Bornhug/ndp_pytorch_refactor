# ===== data.py =====
from __future__ import annotations

from torch import Tensor

from neural_diffusion_processes.types import Batch

from typing import Callable, Dict, List, Mapping, Tuple

import math
from dataclasses import dataclass

import torch
import gpytorch

import random
import numpy as np


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

@dataclass
class UniformContinuous:
    low: float
    high: float
    def sample(self, sample_shape: Tuple[int, ...], *, g: torch.Generator) -> Tensor:
        return self.low + (self.high - self.low) * torch.rand(sample_shape, generator=g)

# ---------------- Public constants ---------------------

# DATASETS = ["se", "matern", "sawtooth", "step"]
DATASETS = ["se", "matern"]
TASKS = ["training", "interpolation"]

# ---------------- Configs ------------------------------

@dataclass
class TaskConfig:
    x_context_dist: UniformContinuous
    x_target_dist: UniformContinuous

@dataclass
class DatasetConfig:
    max_input_dim: int
    is_gp: bool
    eval_num_target: UniformDiscrete = UniformDiscrete(50, 50)
    eval_num_context: UniformDiscrete = UniformDiscrete(1, 10)

_NOISE_VAR   = 0.05 ** 2
_KERNEL_VAR  = 1.0
_LENGTHSCALE = 0.25


_DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "se":     DatasetConfig(max_input_dim=3, is_gp=True),
    "matern": DatasetConfig(max_input_dim=3, is_gp=True),
    "sawtooth": DatasetConfig(max_input_dim=1, is_gp=False),
    "step":     DatasetConfig(max_input_dim=1, is_gp=False),
}

_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "training": TaskConfig(
        x_context_dist = UniformContinuous(-2, 2),
        x_target_dist = UniformContinuous(-2, 2)
    ),

    "interpolation": TaskConfig(
        x_context_dist = UniformContinuous(-2, 2),
        x_target_dist = UniformContinuous(-2, 2)
    ),}







# ---------------- base class ----------------

@dataclass
class FunctionalDistribution:
    is_data_naturally_normalized: bool = True
    normalize: bool = False


# ------------- GP utilities (moved from gp.py) -------------


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
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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
        variance = float(kparams["variance"])

        self.kernel.base_kernel.lengthscale = lengthscale
        self.kernel.outputscale = variance

    @torch.no_grad()
    def sample_prior(
        self,
        key: torch.Generator,
        x: torch.Tensor,        # [N, D]
    ) -> torch.Tensor:         # [N, 1]
        N, device, dtype = x.size(0), x.device, x.dtype

        self._apply_kernel_params()

        K = self.kernel(x)  # LazyTensor [N, N]
        mean = torch.zeros(N, device=device, dtype=dtype)

        prior_dist = gpytorch.distributions.MultivariateNormal(mean, K)
        f = prior_dist.rsample().unsqueeze(-1)  # [N, 1]

        sigma2 = float(self.params["noise_variance"])
        eps = torch.randn_like(f) * math.sqrt(sigma2)
        return f + eps

    @torch.no_grad()
    def sample_posterior(
        self,
        x_train: torch.Tensor,   # [N, D]
        y_train: torch.Tensor,   # [N, 1] or [N]
        x_test: torch.Tensor,    # [M, D]
        *,
        return_obs: bool = False,
    ) -> torch.Tensor:           # [M, 1]
        device, dtype = x_train.device, x_train.dtype
        y_train_vec = y_train.view(-1)

        model = ExactGPModel(x_train, y_train_vec, self.likelihood, self.kernel)
        model = model.to(device=device, dtype=dtype)

        model.eval()
        self.likelihood.eval()

        self._apply_kernel_params()

        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            latent_dist = model(x_test)                 # MultivariateNormal over f_*
            f_samples = latent_dist.rsample().unsqueeze(-1)  # [M, 1]

            if not return_obs:
                return f_samples

            pred_dist = self.likelihood(latent_dist)
            y_samples = pred_dist.rsample().unsqueeze(-1)
            return y_samples

    @torch.no_grad()
    def compute_log_likelihood(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        mask_train: torch.Tensor | None = None,
        mask_test: torch.Tensor | None = None,
        jitter: float = 1e-6,
    ) -> float:
        """
        Predictive log likelihood log p(y_test | x_test, (x_train, y_train))
        using this GPFunctionalDistribution's kernel and noise settings.
        Masks (0=keep, 1=ignore) are applied if provided.
        """

        clean_data = get_batches_from_existing(
            x_context=x_train,
            y_context=y_train,
            x_target=x_test,
            y_target=y_test,
            mask_context=mask_train,
            mask_target=mask_test,
        )
        x_train = clean_data.x_context
        y_train = clean_data.y_context
        x_test = clean_data.x_target
        y_test = clean_data.y_target

        device, dtype = x_train.device, x_train.dtype
        y_train_vec = y_train.view(-1).to(device=device, dtype=dtype)
        y_test_vec = y_test.view(-1).to(device=device, dtype=dtype)

        self._apply_kernel_params()

        model = ExactGPModel(x_train, y_train_vec, self.likelihood, self.kernel).to(device=device, dtype=dtype)
        model.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_computations(covar_root_decomposition=True):
            latent_dist = model(x_test)
            pred_dist = self.likelihood(latent_dist)
            # Let GPyTorch handle numerical stabilisation in log_prob.
            ll = pred_dist.log_prob(y_test_vec)
            return float(ll)


class BaselineGP(GPFunctionalDistribution):
    """
    Simple wrapper around GPFunctionalDistribution for baseline GP scoring.
    Usage:
        gp = BaselineGP(kernel, params)
        y_hat = gp.sample_posterior(x_context, y_context, x_target)
        ll = gp.log_likelihood(x_context, y_context, x_target, y_target)
    """

    def __init__(self, kernel: gpytorch.kernels.Kernel, params: Mapping):
        super().__init__(kernel, params)

    def sample_targets(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        return self.sample_posterior(x_context, y_context, x_target)

    def log_likelihood(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        y_target: torch.Tensor,
        mask_context: torch.Tensor | None = None,
        mask_target: torch.Tensor | None = None,
        jitter: float = 1e-6,
    ) -> float:
        return self.compute_log_likelihood(
            x_context,
            y_context,
            x_target,
            y_target,
            mask_train=mask_context,
            mask_test=mask_target,
            jitter=jitter,
        )


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


# ------------- Mat茅rn-5/2 dataset -------------

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




def get_batch(
        g: torch.Generator,
        *,
        batch_size: int,
        name: str,  # "se" | "matern" | "sawtooth" | "step"
        task: str,  # "training" | "interpolation"
        input_dim: int,
        gp_conditional_targets: bool = False,
) -> Batch:
    if name not in DATASETS: raise ValueError(f"Unknown dataset: {name}")
    if task not in TASKS:    raise ValueError(f"Unknown task: {task}")

    dataset = _DATASET_CONFIGS[name]
    if input_dim > dataset.max_input_dim:
        raise ValueError(f"input_dim {input_dim} > max_input_dim {dataset.max_input_dim} for {name}")

    if task == "training" and not gp_conditional_targets:
        min_n_target = dataset.eval_num_target.low
        max_n_target = (
                dataset.eval_num_target.high
                + dataset.eval_num_context.high * input_dim
        )  # input_dim * num_context + num_target

        max_n_context = dataset.eval_num_context.high


        # Inputs, sample from Uniform (-2,2)
        x_context = _TASK_CONFIGS[task].x_context_dist.sample(
            sample_shape=(batch_size, max_n_context, input_dim), g=g
        )
        x_target = _TASK_CONFIGS[task].x_target_dist.sample(
            sample_shape=(batch_size, max_n_target, input_dim), g=g
        )


        # Sample how many target points to keep
        num_keep_target = torch.randint(min_n_target, max_n_target, (1,), generator=g)[0]  # scalar tensor, always 50
        arange_target = torch.arange(max_n_target, device=x_target.device)[None, :]  # [1, max_n_target]

        # Where keep_mask is True -> 0, else 1
        # [B, max_n_target]
        ''' 
        In this case, arrange_target is array [0,...,50 + 10 * input_dim] ;
                    num_keep_target is one element of [50, 50 + 10 * input_dim), 
        hence at least one element mask_target [50 + 10 * input_dim] is masked.

        Reason for doing this: 
            Regularization?
            you want a variable number of target points while still using a fixed tensor shape (batch_size, max_n_target, input_dim)
        '''
        mask_target = torch.where(arange_target < num_keep_target,  # [1, max_n_target], broadcasts to [B, max_n_target]
                                  torch.zeros_like(x_target[..., 0]),  # keep
                                  torch.ones_like(x_target[..., 0]))  # ignore

        # All context points are ignored
        mask_context = torch.ones_like(x_context[..., 0])  # [B, max_n_context]

        active_dims = list(range(input_dim))
        dataset_factory = _DATASET_FACTORIES[name]
        function_distribution = dataset_factory(active_dims)

        # Prior over context points, looped over batch
        y_context = torch.stack(
            [function_distribution.sample_prior(g, x_context[b]) for b in range(batch_size)],
            dim=0,
        )
        y_target = torch.stack(
            [function_distribution.sample_prior(g, x_target[b]) for b in range(batch_size)],
            dim=0,
        )


    elif task == "training" and gp_conditional_targets:
        min_n_target = dataset.eval_num_target.low
        max_n_target = (
                dataset.eval_num_target.high
                + dataset.eval_num_context.high * input_dim
        )  # input_dim * num_context + num_target

        max_n_context = dataset.eval_num_context.high

        # Inputs, sample from Uniform (-2,2)
        x_context = _TASK_CONFIGS[task].x_context_dist.sample(
            sample_shape=(batch_size, max_n_context, input_dim), g=g
        )
        x_target = _TASK_CONFIGS[task].x_target_dist.sample(
            sample_shape=(batch_size, max_n_target, input_dim), g=g
        )

        # Sample how many target points to keep
        num_keep_target = torch.randint(min_n_target, max_n_target, (1,), generator=g)[0]  # scalar tensor, always 50
        arange_target = torch.arange(max_n_target, device=x_target.device)[None, :]  # [1, max_n_target]

        # Where keep_mask is True -> 0, else 1
        # [B, max_n_target]
        ''' 
        In this case, arrange_target is array [0,...,50 + 10 * input_dim] ;
                    num_keep_target is one element of [50, 50 + 10 * input_dim), 
        hence at least one element mask_target [50 + 10 * input_dim] is masked.

        Reason for doing this: 
            Regularization?
            you want a variable number of target points while still using a fixed tensor shape (batch_size, max_n_target, input_dim)
        '''
        mask_target = torch.where(arange_target < num_keep_target,  # [1, max_n_target], broadcasts to [B, max_n_target]
                                  torch.zeros_like(x_target[..., 0]),  # keep
                                  torch.ones_like(x_target[..., 0]))  # ignore

        # All context points are ***KEPT***
        mask_context = torch.zeros_like(x_context[..., 0])  # [B, max_n_context]

        active_dims = list(range(input_dim))
        dataset_factory = _DATASET_FACTORIES[name]
        function_distribution = dataset_factory(active_dims)

        # Prior over context points, looped over batch
        y_context = torch.stack(
            [function_distribution.sample_prior(g, x_context[b]) for b in range(batch_size)],
            dim=0,
        )

        # Conditional sample: for each batch element, build a GP posterior
        y_target = torch.stack(
            [function_distribution.sample_posterior(x_context[b], y_context[b], x_target[b]) for b in
             range(batch_size)],
            dim=0
        )  # [B, N_tgt, input_dim]


    elif task == "interpolation" and gp_conditional_targets:
        max_n_target = dataset.eval_num_target.high
        max_n_context = dataset.eval_num_context.high * input_dim

        # Inputs, sample from Uniform (-2,2)
        x_context = _TASK_CONFIGS[task].x_context_dist.sample(
            sample_shape=(batch_size, max_n_context, input_dim), g=g
        )
        x_target = _TASK_CONFIGS[task].x_target_dist.sample(
            sample_shape=(batch_size, max_n_target, input_dim), g=g
        )

        # Sample how many context points to keep
        # dataset.eval_num_context.low = 1 here
        num_keep_context = torch.randint(dataset.eval_num_context.low, max_n_context, (1,), generator=g)[0]  # scalar tensor
        arange_context = torch.arange(max_n_context, device=x_context.device)[None, :]  # [1, max_n_context]

        # Where keep_mask is True -> 0, else 1
        # [B, max_n_context]
        ''' 
        In this case, arrange_context [0,...,9] 
                    num_keep_context is one element of [1, 10), 
        hence at least one element mask_context[9] is masked.

        Reason for doing this: 
            Regularization?
            Missing Value in real scenario
            you want a variable number of context points while still using a fixed tensor shape (batch_size, max_n_context, input_dim)
        '''
        mask_context = torch.where(arange_context < num_keep_context,
                                   # [1, max_n_target], broadcasts to [B, max_n_target]
                                   torch.zeros_like(x_context[..., 0]),  # keep
                                   torch.ones_like(x_context[..., 0]))  # ignore

        # All context points are kept (no padding)
        mask_target = torch.zeros_like(x_target[..., 0])  # [B, max_n_context]

        active_dims = list(range(input_dim))
        dataset_factory = _DATASET_FACTORIES[name]
        function_distribution = dataset_factory(active_dims)

        # Prior over context points, looped over batch
        y_context = torch.stack(
            [function_distribution.sample_prior(g, x_context[b]) for b in range(batch_size)],
            dim=0,
        )

        # Conditional sample: for each batch element, build a GP posterior
        y_target = torch.stack(
            [function_distribution.sample_posterior(x_context[b], y_context[b], x_target[b]) for b in
             range(batch_size)],
            dim=0
        )  # [B, N_tgt, input_dim]

    return Batch(
        x_target=x_target,
        y_target=y_target,
        x_context=x_context,
        y_context=y_context,
        mask_target=mask_target,
        mask_context=mask_context,
    )





def get_batch_with_prob(
        g: torch.Generator,
        prob: float = 0.2,  # The ratio between uncond and cond samples
        *,
        batch_size: int,
        name: str,  # "se" | "matern" | "sawtooth" | "step"
        task: str,  # "training" | "interpolation"
        input_dim: int,
)-> Batch:

    if random.random() < prob:
        return get_batch(g,
                  batch_size=batch_size,
                  name=name,
                  task=task,
                  input_dim=input_dim,
                  gp_conditional_targets=False)
    else:
        return get_batch(g,
                  batch_size=batch_size,
                  name=name,
                  task=task,
                  input_dim=input_dim,
                  gp_conditional_targets=True)





# ------------- For Interpolation / Evaluation -------------


def get_batches_from_existing(
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        y_target: torch.Tensor,
        mask_context: torch.Tensor | None = None,
        mask_target: torch.Tensor | None = None,
):
    '''
    Once we generate the synthetic dataset / real dataset, we extract the data from them for visualization
        or computing likelihood.
    Notice that we may encounter the situation that the contexts/targets have exact values on their columns,
        but they are acutally masked (the mask_context / mask_target is 1).
        We need to handle this by filtering out the masked contexts/targets.
    '''
    if mask_context is not None:  # mask_context is not None
        keep_c = mask_context == 0
        x_context = x_context[keep_c]
        y_context = y_context[keep_c]
    if mask_target is not None:  # mask_target is not None
        keep_t = mask_target == 0
    else:  # mask_target is None, so keep all targets by setting keep_t to 1
        keep_t = torch.ones(x_target.shape[0], dtype=torch.bool, device=x_target.device)

    if keep_t.sum() == 0:
        # The mask_target says 鈥渋gnore everything鈥?(all entries are 1, so keep_t.sum() == 0
        raise ValueError("All targets are masked; at least one unmasked target is required.")

    x_target = x_target[keep_t]
    y_target = y_target[keep_t]
    return Batch(
        x_target=x_target,
        y_target=y_target,
        x_context=x_context,
        y_context=y_context
    )

def _load_npz(npz_path: str):
    data = np.load(npz_path)
    x_context = torch.from_numpy(data["x_context"].astype(np.float32))
    y_context = torch.from_numpy(data["y_context"].astype(np.float32))
    x_target = torch.from_numpy(data["x_target"].astype(np.float32))
    y_target = torch.from_numpy(data["y_target"].astype(np.float32))
    mask_target = torch.from_numpy(data["mask_target"].astype(np.float32))
    mask_context = torch.from_numpy(data["mask_context"].astype(np.float32))

    return Batch(
        x_target=x_target,
        y_target=y_target,
        x_context=x_context,
        y_context=y_context,
        mask_target=mask_target,
        mask_context=mask_context
    )

def load_batch_from_npz(npz_path: str):
    '''
        Load all the batches from the npz file;
        Then extract the contexts and targets which are unmasked.
    '''

    batch = _load_npz(npz_path)
    return get_batches_from_existing(
        x_context=batch.x_context,
        y_context=batch.y_context,
        x_target=batch.x_target,
        y_target=batch.y_target,
        mask_target=batch.mask_target,
        mask_context=batch.mask_context
    )


# class Sawtooth(FunctionalDistribution):
#     A = 1.0
#     K_max = 20
#     mean = 0.5
#     variance = 0.07965
#     def sample(self, x: Tensor, g: torch.Generator) -> Tensor:
#         # Use first dim only; return (N,1)
#         x1 = x[..., 0:1]
#         f = 3.0 + 2.0 * torch.rand((), generator=g, device=x.device, dtype=x.dtype)
#         s = -5.0 + 10.0 * torch.rand((), generator=g, device=x.device, dtype=x.dtype)
#         ks = torch.arange(1, self.K_max + 1, dtype=x.dtype, device=x.device)[None, :]
#         vals = (-1.0) ** ks * torch.sin(2 * math.pi * ks * f * (x1 - s)) / ks
#         k = torch.randint(10, self.K_max + 1, (), generator=g)
#         mask = (ks < k).float()
#         fs = self.A / 2 + self.A / math.pi * (vals * mask).sum(dim=1, keepdim=True)
#         fs = fs - self.mean
#         return fs  # (N,1)
#
# @register_dataset_factory("sawtooth")
# def _sawtooth_dataset_factory(*args):
#     return Sawtooth()
#
# class Step(FunctionalDistribution):
#     def sample(self, x: Tensor, g: torch.Generator) -> Tensor:
#         # Use first dim only; return (N,1)
#         x1 = x[..., 0:1]
#         s = -2.0 + 4.0 * torch.rand((), generator=g, device=x.device, dtype=x.dtype)
#         return torch.where(x1 < s, torch.zeros_like(x1), torch.ones_like(x1))  # (N,1)
#
# @register_dataset_factory("step")
# def _step_dataset_factory(*args):
#     return Step()



