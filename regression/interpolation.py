from __future__ import annotations

import argparse
import pathlib
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from config import Config
from neural_diffusion_processes.model import BiDimensionalAttentionModel
from neural_diffusion_processes.process import GaussianDiffusion, cosine_schedule
from neural_diffusion_processes.types import Batch
from likelihood import GPLikelihoodEvaluator

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
def sample_conditionals(
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
    for _ in tqdm(range(num_samples), desc="Sampling NDP conditionals"):
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


def plot(batch: Batch, samples: torch.Tensor, out_path: Optional[str] = None):
    """
    Build a matplotlib figure showing conditional samples for batch 0.
    This function no longer saves to disk; the caller is responsible for logging
    or saving the returned figure if desired.
    """
    if batch.x_target.shape[-1] != 1:
        return None
    import matplotlib.pyplot as plt

    x = batch.x_target[0, :, 0].cpu().numpy()
    y_true = batch.y_target[0, :, 0].cpu().numpy()
    mc = batch.mask_context[0].cpu().numpy() if batch.mask_context is not None else None
    xc = batch.x_context[0, :, 0].cpu().numpy() if batch.x_context is not None else None
    yc = batch.y_context[0, :, 0].cpu().numpy() if batch.y_context is not None else None

    s = samples[:, 0, :, 0].cpu().numpy()  # [S,N]
    mean = s.mean(axis=0)
    var = s.var(axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, s.T, color="C0", alpha=0.3, lw=1)
    ax.plot(x, mean, color="k", lw=2, label="mean")
    ax.fill_between(x, mean - 1.96 * np.sqrt(var), mean + 1.96 * np.sqrt(var), color="k", alpha=0.1, label="95% CI")
    ax.plot(x, y_true, "g--", lw=1, label="true")
    if xc is not None and mc is not None:
        keep_c = mc == 0
        ax.plot(xc[keep_c], yc[keep_c], "ko", label="context")
    ax.legend()
    ax.set_title("Conditional samples (batch 0)")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Evaluate conditional interpolation with PyTorch diffusion model.")
    default_ckpt = HERE / "logs" / "regression-pytorch" / "Nov16_183218_wbfp" / "checkpoints" / "model_step_128000.pt"
    parser.add_argument(
        "--ckpt",
        default=str(default_ckpt),
        help="Path to checkpoint .pt file (from train.py).",
    )
    parser.add_argument("--dataset", default="matern")
    parser.add_argument("--input-dim", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=8, help="Conditional samples per batch.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--npz-path",
        type=str,
        default=None,
        help="Path to interpolation .npz in regression/data. If not set, will use regression/data/{dataset}_{input_dim}_interpolation.npz",
    )
    args = parser.parse_args()

    config = Config()
    config.dataset = args.dataset
    config.input_dim = args.input_dim
    config.eval.batch_size = args.batch_size
    config.eval.num_samples = args.num_samples

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    model, process = make_model_and_process(config, device)
    load_checkpoint(model, pathlib.Path(args.ckpt), device)

    # Load evaluation data from regression/data npz
    if args.npz_path is None:
        npz_path = pathlib.Path(__file__).parent / "data" / f"{config.dataset}_{config.input_dim}_interpolation.npz"
    else:
        npz_path = pathlib.Path(args.npz_path)

    # GP evaluator (loads data and GP definition)
    evaluator = GPLikelihoodEvaluator(
        npz_path=str(npz_path),
        name=config.dataset,
        input_dim=config.input_dim,
        jitter=1e-6,
    )

    # Build full batch for NDP sampling on device
    batch_all = Batch(
        x_target=torch.as_tensor(evaluator.x_target, device=device, dtype=torch.float32),
        y_target=torch.as_tensor(evaluator.y_target, device=device, dtype=torch.float32),
        x_context=torch.as_tensor(evaluator.x_context, device=device, dtype=torch.float32),
        y_context=torch.as_tensor(evaluator.y_context, device=device, dtype=torch.float32),
        mask_target=torch.as_tensor(evaluator.mask_target, device=device, dtype=torch.float32),
        mask_context=torch.as_tensor(evaluator.mask_context, device=device, dtype=torch.float32),
    )

    print(f"[eval] Sampling {config.eval.num_samples} NDP draws from NDP...")
    y_samples = sample_conditionals(
        process,
        model,
        batch=batch_all,
        num_samples=config.eval.num_samples,
        seed=args.seed,
    )  # [S,B,N,1]
    y_target_ndp = y_samples.squeeze(-1).cpu()  # [S,B,N]

    # Compute GP likelihood-based metrics
    print("[eval] Computing baseline GP log-likelihood...")
    ll_baseline = evaluator.compute_baselineGP()                   # [B]
    print("[eval] Computing Method 1 GP log-likelihood...")
    ll_m1 = evaluator.method1_log_likelihood(y_target_ndp)        # [B]
    print("[eval] Computing Method 2 GP log-likelihood...")
    ll_m2 = evaluator.method2_log_likelihood(y_target_ndp)        # [B]

    # Log to wandb (if available)
    if wandb is not None:
        wandb.init(
            project="ndp-regression-eval",
            config={
                "dataset": config.dataset,
                "input_dim": config.input_dim,
                "num_samples": config.eval.num_samples,
                "ckpt": args.ckpt,
                "npz_path": str(npz_path),
            },
        )

        def _log_ll(label: str, ll_vec: torch.Tensor):
            mean, std, stderr = evaluator._summarize_ll(ll_vec)
            wandb.log(
                {
                    f"{label}/mean": mean,
                    f"{label}/std": std,
                    f"{label}/stderr": stderr,
                }
            )

        _log_ll("ll_baseline", ll_baseline)
        _log_ll("ll_m1", ll_m1)
        _log_ll("ll_m2", ll_m2)

    # Always produce and (if wandb is enabled) log a conditional plot for batch 0 in 1D.
    first_batch = Batch(
        x_target=batch_all.x_target[:1],
        y_target=batch_all.y_target[:1],
        x_context=batch_all.x_context[:1],
        y_context=batch_all.y_context[:1],
        mask_target=batch_all.mask_target[:1],
        mask_context=batch_all.mask_context[:1],
    )
    samples_plot = sample_conditionals(
        process,
        model,
        batch=first_batch,
        num_samples=config.eval.num_samples,
        seed=args.seed,
    )
    fig = plot(first_batch, samples_plot)

    # Log the plot image to Weights & Biases (no local save)
    if wandb is not None and fig is not None:
        print("plots are uploading to wandb")
        wandb.log({"eval/conditional_plot": wandb.Image(fig)})


if __name__ == "__main__":
    main()
