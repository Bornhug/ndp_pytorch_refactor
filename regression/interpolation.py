from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root (one level up from regression/) is on PYTHONPATH for local imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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


def plot_ndp_vs_baseline(
    x: torch.Tensor,
    y_true: torch.Tensor,
    x_context: Optional[torch.Tensor],
    y_context: Optional[torch.Tensor],
    mask_context: Optional[torch.Tensor],
    ndp_samples: torch.Tensor,        # [S,N]
    gp_samples: torch.Tensor,         # [S,N]
    out_path: pathlib.Path,
) -> None:
    """
    Plot NDP and baseline GP conditional samples for a single 1D batch.
    """
    import matplotlib.pyplot as plt

    x_np = x.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    xc_np = x_context.cpu().numpy() if x_context is not None else None
    yc_np = y_context.cpu().numpy() if y_context is not None else None
    mc_np = mask_context.cpu().numpy() if mask_context is not None else None

    # Sort by x for smooth curves
    order = np.argsort(x_np)
    x_np = x_np[order]
    y_true_np = y_true_np[order]

    ndp_np = ndp_samples.cpu().numpy()[:, order]  # [S,N]
    ndp_mean = ndp_np.mean(axis=0)
    ndp_var = ndp_np.var(axis=0)

    gp_np = gp_samples.cpu().numpy()[:, order]    # [S,N]
    gp_mean = gp_np.mean(axis=0)
    gp_var = gp_np.var(axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    # NDP samples
    ax.plot(x_np, ndp_np.T, color="C0", alpha=0.3, lw=1)
    ax.plot(x_np, ndp_mean, color="C0", lw=2, label="NDP mean")
    ax.fill_between(
        x_np,
        ndp_mean - 1.96 * np.sqrt(ndp_var),
        ndp_mean + 1.96 * np.sqrt(ndp_var),
        color="C0",
        alpha=0.1,
        label="NDP 95% CI",
    )

    # Baseline GP mean/CI
    ax.plot(x_np, gp_mean, color="C3", lw=2, label="Baseline GP mean")
    ax.fill_between(
        x_np,
        gp_mean - 1.96 * np.sqrt(gp_var),
        gp_mean + 1.96 * np.sqrt(gp_var),
        color="C3",
        alpha=0.1,
        label="Baseline GP 95% CI",
    )

    # Context only (omit plotting the full true function)
    if xc_np is not None and mc_np is not None:
        keep_c = mc_np == 0
        ax.plot(xc_np[keep_c], yc_np[keep_c], "ko", label="context")

    ax.legend()
    ax.set_title("NDP vs baseline GP conditional samples (batch 0)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate conditional interpolation with PyTorch diffusion model.")
    default_ckpt = HERE / "logs" / "regression-pytorch" / "Nov16_183218_wbfp" / "checkpoints" / "model_step_128000.pt"
    parser.add_argument(
        "--ckpt",
        default=str(default_ckpt),
        help="Path to checkpoint .pt file (from train.py).",
    )
    parser.add_argument("--dataset", default="se")
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

    ckpt_path = pathlib.Path(args.ckpt)
    model, process = make_model_and_process(config, device)
    load_checkpoint(model, ckpt_path, device)

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
    y_samples_full = sample_conditionals(  # handle mask_contexts inside
        process,
        model,
        batch=batch_all,
        num_samples=config.eval.num_samples,
        seed=args.seed,
    )  # [S,B,N,1]
    y_target_ndp = y_samples_full.squeeze(-1).cpu()  # [S,B,N]

    # Compute GP likelihood-based metrics
    print("[eval] Computing baseline GP log-likelihood...")
    ll_baseline = evaluator.compute_baselineGP()                   # [B]
    print("[eval] Computing Method 1 GP log-likelihood...")
    ll_m1 = evaluator.method1_log_likelihood(y_target_ndp)        # [B]
    print("[eval] Computing Method 2 GP log-likelihood...")
    ll_m2 = evaluator.method2_log_likelihood(y_target_ndp)        # [B]

    # ---- Save metrics locally next to the checkpoint run directory ----
    run_dir = ckpt_path.parent.parent  # .../run_name
    metrics_path = run_dir / f"interpolation_metrics_{config.dataset}_{config.input_dim}.json"

    b_mean, b_std, b_stderr = evaluator._summarize_ll(ll_baseline)
    m1_mean, m1_std, m1_stderr = evaluator._summarize_ll(ll_m1)
    m2_mean, m2_std, m2_stderr = evaluator._summarize_ll(ll_m2)

    metrics = {
        "dataset": config.dataset,
        "input_dim": config.input_dim,
        "num_samples": config.eval.num_samples,
        "ll_baseline": {
            "values": ll_baseline.cpu().tolist(),
            "mean": b_mean,
            "std": b_std,
            "stderr": b_stderr,
        },
        "ll_m1": {
            "values": ll_m1.cpu().tolist(),
            "mean": m1_mean,
            "std": m1_std,
            "stderr": m1_stderr,
        },
        "ll_m2": {
            "values": ll_m2.cpu().tolist(),
            "mean": m2_mean,
            "std": m2_std,
            "stderr": m2_stderr,
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ---- Plot NDP vs baseline GP for batch 0 and save alongside metrics ----
    first_batch = Batch(
        x_target=batch_all.x_target[:1],
        y_target=batch_all.y_target[:1],
        x_context=batch_all.x_context[:1],
        y_context=batch_all.y_context[:1],
        mask_target=batch_all.mask_target[:1],
        mask_context=batch_all.mask_context[:1],
    )

    # NDP samples for batch 0: reuse the first-batch slice from the full sampling pass
    ndp_samples_plot = y_samples_full[:, 0, :, 0].cpu()  # [S,N]

    # Baseline GP samples for corresponding example (CPU tensors via evaluator)
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

    gp_samples_list = []
    for _ in range(config.eval.num_samples):
        y_gp = baseline_gp.sample_targets(
            clean0.x_context,
            clean0.y_context,
            clean0.x_target,
        )  # [N,1]
        gp_samples_list.append(y_gp[:, 0].unsqueeze(0))  # [1,N]
    gp_samples = torch.cat(gp_samples_list, dim=0)  # [S,N]

    x_first = first_batch.x_target[0, :, 0].cpu()
    y_true_first = first_batch.y_target[0, :, 0].cpu()
    x_context_first = first_batch.x_context[0, :, 0].cpu()
    y_context_first = first_batch.y_context[0, :, 0].cpu()
    mask_context_first = first_batch.mask_context[0].cpu() if first_batch.mask_context is not None else None

    plot_path = run_dir / f"interpolation_plot_{config.dataset}_{config.input_dim}.png"
    plot_ndp_vs_baseline(
        x_first,
        y_true_first,
        x_context_first,
        y_context_first,
        mask_context_first,
        ndp_samples_plot,
        gp_samples,
        plot_path,
    )


if __name__ == "__main__":
    main()
