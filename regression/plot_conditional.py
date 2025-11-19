from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Optional

import numpy as np
import torch

# Ensure project root (containing `neural_diffusion_processes`) is on sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import Config
from neural_diffusion_processes.model import BiDimensionalAttentionModel
from neural_diffusion_processes.process import GaussianDiffusion, cosine_schedule
from neural_diffusion_processes.types import Batch
from likelihood import GPLikelihoodEvaluator
from data import BaselineGP, get_batches_from_existing


HERE = pathlib.Path(__file__).parent


def set_seed(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_model_and_process(config: Config, device: torch.device | str):
    device = torch.device(device)
    model = BiDimensionalAttentionModel(
        n_layers=config.network.n_layers,
        hidden_dim=config.network.hidden_dim,
        num_heads=config.network.num_heads,
    ).to(device)
    betas = cosine_schedule(
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        timesteps=config.diffusion.timesteps,
    ).to(device)
    process = GaussianDiffusion(betas)
    return model, process


def load_checkpoint(model: torch.nn.Module, ckpt_path: pathlib.Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_key = "ema_state" if "ema_state" in ckpt else "model_state"
    model.load_state_dict(ckpt[state_key])
    return ckpt.get("config", None)


@torch.no_grad()
def sample_ndp_conditionals(
    process: GaussianDiffusion,
    model: BiDimensionalAttentionModel,
    batch: Batch,
    num_samples: int,
    seed: int = 0,
) -> torch.Tensor:
    """
    Draw `num_samples` conditional samples for the entire batch.
    Returns tensor of shape [num_samples, B, N_tgt, 1].
    """
    device = batch.x_target.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    samples = []
    for _ in range(num_samples):
        y_hat = process.sample(
            key=g,
            x=batch.x_target,
            mask=batch.mask_target,
            x_context=batch.x_context,
            y_context=batch.y_context,
            mask_context=batch.mask_context,
            model=model,
        )  # [B, N_tgt, 1]
        samples.append(y_hat.unsqueeze(0))
    return torch.cat(samples, dim=0)


def plot_samples(
    x: torch.Tensor,
    y_true: torch.Tensor,
    x_context: Optional[torch.Tensor],
    y_context: Optional[torch.Tensor],
    mask_context: Optional[torch.Tensor],
    samples: torch.Tensor,
    out_path: pathlib.Path,
    title: str,
):
    """
    Plot sampled functions, their mean and 95% CI, plus ground truth and context points.
    """
    import matplotlib.pyplot as plt

    x_np = x.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    xc_np = x_context.cpu().numpy() if x_context is not None else None
    yc_np = y_context.cpu().numpy() if y_context is not None else None
    mc_np = mask_context.cpu().numpy() if mask_context is not None else None

    # Sort by x so curves look like smooth GP functions
    order = np.argsort(x_np)
    x_np = x_np[order]
    y_true_np = y_true_np[order]

    # samples: [S,N] (already squeezed before calling)
    s_np = samples.cpu().numpy()[:, order]  # reorder along N
    mean = s_np.mean(axis=0)
    var = s_np.var(axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_np, s_np.T, color="C0", alpha=0.3, lw=1)
    ax.plot(x_np, mean, color="k", lw=2, label="mean")
    ax.fill_between(
        x_np,
        mean - 1.96 * np.sqrt(var),
        mean + 1.96 * np.sqrt(var),
        color="k",
        alpha=0.1,
        label="95% CI",
    )
    ax.plot(x_np, y_true_np, "g--", lw=1, label="true")
    if xc_np is not None and mc_np is not None:
        keep_c = mc_np == 0
        ax.plot(xc_np[keep_c], yc_np[keep_c], "ko", label="context")
    ax.legend()
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot NDP vs baseline GP conditional samples for a single batch."
    )
    default_ckpt = (
        HERE / "logs" / "regression-pytorch" / "Nov16_183218_wbfp" / "checkpoints" / "model_step_128000.pt"
    )
    parser.add_argument(
        "--ckpt",
        default=str(default_ckpt),
        help="Path to checkpoint .pt file (from train.py).",
    )
    parser.add_argument("--dataset", default="matern")
    parser.add_argument("--input-dim", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to draw.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--npz-path",
        type=str,
        default=None,
        help=(
            "Path to interpolation .npz in regression/data. "
            "If not set, will use regression/data/{dataset}_{input_dim}_interpolation.npz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(HERE / "visualization"),
        help="Directory to save the output plots.",
    )
    args = parser.parse_args()

    config = Config()
    config.dataset = args.dataset
    config.input_dim = args.input_dim
    config.eval.num_samples = args.num_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    model, process = make_model_and_process(config, device)
    load_checkpoint(model, pathlib.Path(args.ckpt), device)

    # Load evaluation data from regression/data npz
    if args.npz_path is None:
        npz_path = HERE / "data" / f"{config.dataset}_{config.input_dim}_interpolation.npz"
    else:
        npz_path = pathlib.Path(args.npz_path)

    # GP evaluator (loads data and GP definition)
    evaluator = GPLikelihoodEvaluator(
        npz_path=str(npz_path),
        name=config.dataset,
        input_dim=config.input_dim,
        jitter=1e-6,
    )

    # Build first batch for NDP sampling on device (batch index 0)
    batch_first = Batch(
        x_target=torch.as_tensor(evaluator.x_target[:1], device=device, dtype=torch.float32),
        y_target=torch.as_tensor(evaluator.y_target[:1], device=device, dtype=torch.float32),
        x_context=torch.as_tensor(evaluator.x_context[:1], device=device, dtype=torch.float32),
        y_context=torch.as_tensor(evaluator.y_context[:1], device=device, dtype=torch.float32),
        mask_target=torch.as_tensor(evaluator.mask_target[:1], device=device, dtype=torch.float32),
        mask_context=torch.as_tensor(evaluator.mask_context[:1], device=device, dtype=torch.float32),
    )

    # NDP samples: [S,1,N,1] -> [S,N]
    ndp_samples = sample_ndp_conditionals(
        process,
        model,
        batch=batch_first,
        num_samples=config.eval.num_samples,
        seed=args.seed,
    )
    ndp_samples_2d = ndp_samples[:, 0, :, 0].cpu()  # [S,N]

    # Baseline GP samples for the corresponding example (CPU tensors via evaluator)
    b0 = 0
    mask_ctx0 = evaluator.mask_context[b0] if evaluator.mask_context is not None else None
    mask_tgt0 = evaluator.mask_target[b0] if evaluator.mask_target is not None else None
    clean0 = get_batches_from_existing(
        x_context=evaluator.x_context[b0],
        y_context=evaluator.y_context[b0],
        x_target=evaluator.x_target[b0],
        y_target=evaluator.y_target[b0],
        mask_context=mask_ctx0,
        mask_target=mask_tgt0,
    )

    if isinstance(evaluator.gp, BaselineGP):
        baseline_gp = evaluator.gp
    else:
        baseline_gp = BaselineGP(evaluator.gp.kernel, evaluator.gp.params)

    baseline_samples_list = []
    for _ in range(config.eval.num_samples):
        y_gp = baseline_gp.sample_targets(
            clean0.x_context,
            clean0.y_context,
            clean0.x_target,
        )  # [N,1]
        baseline_samples_list.append(y_gp.unsqueeze(0))  # [1,N,1]
    baseline_samples = torch.cat(baseline_samples_list, dim=0)  # [S,N,1]
    baseline_samples_2d = baseline_samples[:, :, 0]  # [S,N]

    # Data for plotting (first batch, CPU tensors)
    x_first = batch_first.x_target[0, :, 0].cpu()
    y_true_first = batch_first.y_target[0, :, 0].cpu()
    x_context_first = batch_first.x_context[0, :, 0].cpu()
    y_context_first = batch_first.y_context[0, :, 0].cpu()
    mask_context_first = batch_first.mask_context[0].cpu() if batch_first.mask_context is not None else None

    out_dir = pathlib.Path(args.out_dir)
    ndp_path = out_dir / f"{config.dataset}_{config.input_dim}_ndp_samples.png"
    gp_path = out_dir / f"{config.dataset}_{config.input_dim}_baseline_gp_samples.png"

    plot_samples(
        x_first,
        y_true_first,
        x_context_first,
        y_context_first,
        mask_context_first,
        ndp_samples_2d,
        ndp_path,
        title="NDP conditional samples (batch 0)",
    )

    plot_samples(
        x_first,
        y_true_first,
        x_context_first,
        y_context_first,
        mask_context_first,
        baseline_samples_2d,
        gp_path,
        title="Baseline GP conditional samples (batch 0)",
    )


if __name__ == "__main__":
    main()
