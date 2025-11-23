from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

# Make the project root importable so we can access neural_diffusion_processes.
import sys

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from neural_diffusion_processes.model import BiDimensionalAttentionModel  # noqa: E402
from neural_diffusion_processes.process import (  # noqa: E402
    GaussianDiffusion,
    cosine_schedule,
    loss as diffusion_loss,
)
from neural_diffusion_processes.types import Batch  # noqa: E402
from nanoTabPFN.config import Config  # noqa: E402
from nanoTabPFN.data import PriorDumpDataLoader  # noqa: E402


class BiDimClassificationEpsNet(BiDimensionalAttentionModel):
    """
    Bi-dimensional epsilon network for classification.

    Extends `BiDimensionalAttentionModel` to operate on one-hot labels with
    `output_dim` channels, while reusing the same bidimensional attention
    blocks internally.
    """

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        num_heads: int,
        output_dim: int,
        init_zero: bool = True,
    ) -> None:
        super().__init__(n_layers=n_layers, hidden_dim=hidden_dim, num_heads=num_heads, init_zero=False)
        self.output_dim = int(output_dim)

        # Last axis now has size 1 (x) + output_dim (one-hot y).
        self.input_linear = nn.Linear(1 + self.output_dim, hidden_dim)
        self.proj_eps = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, self.output_dim)
        if init_zero:
            nn.init.zeros_(self.output_linear.weight)
            nn.init.zeros_(self.output_linear.bias)

    def process_inputs(self, x, y):
        # x : [B, N, D] or [N, D]
        # y : [B, N, C] or [N, C] with C = output_dim (one-hot labels / signals)
        if x is None or y is None:
            return None

        if x.ndim == 2:  # [N,D] -> [1,N,D]
            x = x.unsqueeze(0)
        if y.ndim == 2:  # [N,C] -> [1,N,C]
            y = y.unsqueeze(0)  # [1,N,C]

        # x: [B,N,D] -> [B,N,D,1]
        if x.ndim == 3:
            x = x.unsqueeze(-1)

        # y: [B,N,C] -> [B,N,1,C] -> [B,N,D,C]
        if y.ndim == 3:
            y = y.unsqueeze(2)  # [B,N,1,C]
            y = y.expand(-1, -1, x.size(2), -1)  # match D

        return torch.cat([x, y], dim=-1)  # [B,N,D,1+C]


class NanoTabPFNModel(nn.Module):
    """
    Diffusion epsilon-model for the nanoTabPFN prior, wrapping a
    classification-aware bidimensional attention model with feature
    z-score normalisation.

    The forward signature matches `BiDimensionalAttentionModel` so it can
    be used with `neural_diffusion_processes.process.loss`.
    """

    def __init__(
        self,
        num_features: int,
        num_outputs: int,
        *,
        embedding_size: int,
        num_attention_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.num_features = int(num_features)
        self.num_outputs = int(num_outputs)

        # Core bidimensional attention epsilon-network for C-dimensional labels.
        self.core = BiDimClassificationEpsNet(
            n_layers=num_layers,
            hidden_dim=embedding_size,
            num_heads=num_attention_heads,
            output_dim=self.num_outputs,
        )

    def _compute_normalization_stats(
        self, x_tgt: torch.Tensor | None, x_context: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """
        Compute per-task normalization statistics from the provided features.

        We pool across context and target points so that both branches share the
        same normalization parameters for a given task.
        """
        tensors: list[torch.Tensor] = []
        if x_tgt is not None:
            tensors.append(x_tgt)
        if x_context is not None:
            tensors.append(x_context)

        if not tensors:
            return None, None

        stacked = torch.cat(tensors, dim=1)  # [B, N_total, F]
        mean = stacked.mean(dim=1, keepdim=True)
        var = stacked.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var.clamp_min(1e-6))
        return mean, std

    def _normalize(
        self,
        x: torch.Tensor | None,
        mean: torch.Tensor | None,
        std: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if x is None or mean is None or std is None:
            return x
        return (x - mean) / std

    def forward(
        self,
        x_tgt: torch.Tensor,
        y_tgt: torch.Tensor,
        t: torch.Tensor,
        mask_tgt: torch.Tensor | None = None,
        x_context: torch.Tensor | None = None,
        y_context: torch.Tensor | None = None,
        mask_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass used inside the diffusion loss.

        x_*: [B, N_*, F]
        y_*: [B, N_*, C] where C = num_outputs
        t:   [B]
        """
        mean, std = self._compute_normalization_stats(x_tgt, x_context)
        x_tgt_norm = self._normalize(x_tgt, mean, std)
        x_ctx_norm = self._normalize(x_context, mean, std)

        return self.core(
            x_tgt=x_tgt_norm,
            y_tgt=y_tgt,
            t=t,
            mask_tgt=mask_tgt,
            x_context=x_ctx_norm,
            y_context=y_context,
            mask_context=mask_context,
        )


def get_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_lr(config: Config, step: int) -> float:
    """
    Cosine schedule with linear warmup, mirroring the regression config.
    """
    warmup_steps = config.training.steps_per_epoch * config.optim.num_warmup_epochs
    decay_steps = config.training.steps_per_epoch * config.optim.num_decay_epochs
    init_lr = config.optim.init_lr
    peak_lr = config.optim.peak_lr
    end_lr = config.optim.end_lr

    if step < warmup_steps:
        if warmup_steps == 0:
            return peak_lr
        alpha = step / float(warmup_steps)
        return init_lr + (peak_lr - init_lr) * alpha

    t = min(step - warmup_steps, decay_steps)
    if decay_steps <= 0:
        return end_lr

    cosine = 0.5 * (1.0 + np.cos(np.pi * t / float(decay_steps)))
    return end_lr + (peak_lr - end_lr) * cosine


def train(
    model: NanoTabPFNModel,
    prior: PriorDumpDataLoader,
    process: GaussianDiffusion,
    *,
    num_timesteps: int,
    num_classes: int,
    config: Config,
    device: torch.device | None = None,
    loss_type: str = "l1",
) -> Tuple[NanoTabPFNModel, Dict[str, List[float]]]:
    """
    Diffusion-based training loop over the prior dump.
    """
    device = device or get_default_device()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.init_lr,
        weight_decay=config.optim.weight_decay,
    )
    losses: List[float] = []
    grad_norms: List[float] = []
    lrs: List[float] = []

    total_steps = config.training.total_steps

    progress = tqdm(enumerate(prior, start=1), total=total_steps, desc="training")

    for step, batch in progress:
        optimizer.zero_grad(set_to_none=True)

        # Move to device and build one-hot labels for diffusion in class space.
        x_tgt = torch.as_tensor(batch.x_target, device=device, dtype=torch.float32)
        x_ctx = torch.as_tensor(batch.x_context, device=device, dtype=torch.float32)

        y_tgt_int = torch.as_tensor(batch.y_target, device=device).long().squeeze(-1)
        y_ctx_int = torch.as_tensor(batch.y_context, device=device).long().squeeze(-1)

        y_tgt_oh = F.one_hot(y_tgt_int, num_classes=num_classes).float()
        y_ctx_oh = F.one_hot(y_ctx_int, num_classes=num_classes).float()

        batch_t = Batch(
            x_target=x_tgt,
            y_target=y_tgt_oh,
            x_context=x_ctx,
            y_context=y_ctx_oh,
            mask_target=None,
            mask_context=None,
        )

        lr = compute_lr(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss = diffusion_loss(
            process,
            model,
            batch_t,
            None,
            num_timesteps=num_timesteps,
            loss_type=loss_type,
        )

        # --- DEBUG: inspect non-finite values ---
        if not torch.isfinite(loss) or (step <= 3):
            print(f"\n{'='*60}")
            print(f"Step {step} - Loss: {loss.item():.6f}, LR: {lr:.6e}")
            print(f"{'='*60}")

            # Check input data
            with torch.no_grad():
                print("Input checks:")
                print(f"  x_tgt finite: {torch.isfinite(batch_t.x_target).all().item()}")
                print(f"  y_tgt finite: {torch.isfinite(batch_t.y_target).all().item()}")
                print(f"  x_ctx finite: {torch.isfinite(batch_t.x_context).all().item()}")
                print(f"  y_ctx finite: {torch.isfinite(batch_t.y_context).all().item()}")
                print(f"  x_tgt range: [{batch_t.x_target.min().item():.4f}, {batch_t.x_target.max().item():.4f}]")
                print(f"  y_tgt sum per sample: {batch_t.y_target.sum(dim=-1).mean().item():.4f}")
                
                # Check diffusion outputs
                B, N, C = batch_t.y_target.shape
                from neural_diffusion_processes.process import stratified_timesteps
                t = stratified_timesteps(B, num_timesteps, device=device)
                t_b = t.view(B, 1, 1)
                y_t, eps_true = process.forward(None, batch_t.y_target, t_b)
                
                print(f"\nDiffusion forward:")
                print(f"  timesteps t: {t[:5].tolist()}")
                print(f"  y_t finite: {torch.isfinite(y_t).all().item()}")
                print(f"  y_t range: [{y_t.min().item():.4f}, {y_t.max().item():.4f}]")
                print(f"  eps_true finite: {torch.isfinite(eps_true).all().item()}")
                print(f"  eps_true range: [{eps_true.min().item():.4f}, {eps_true.max().item():.4f}]")
                
                eps_hat = model(
                    x_tgt=batch_t.x_target,
                    y_tgt=y_t,
                    t=t,
                    mask_tgt=None,
                    x_context=batch_t.x_context,
                    y_context=batch_t.y_context,
                    mask_context=None,
                )
                print(f"\nModel output:")
                print(f"  eps_hat finite: {torch.isfinite(eps_hat).all().item()}")
                print(f"  eps_hat range: [{eps_hat.min().item():.4f}, {eps_hat.max().item():.4f}]")
                print(f"  eps_hat mean/std: {eps_hat.mean().item():.4f} / {eps_hat.std().item():.4f}")

            # Check parameters for NaN
            if not torch.isfinite(loss):
                for name, p in model.named_parameters():
                    if not torch.isfinite(p).all():
                        print(f"  NON-FINITE PARAM: {name}")
                        print(f"    shape: {p.shape}, non-finite count: {(~torch.isfinite(p)).sum().item()}")
                        break
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        print(f"  NON-FINITE GRAD: {name}")
                        print(f"    shape: {p.grad.shape}, non-finite count: {(~torch.isfinite(p.grad)).sum().item()}")
                        break
                
                print("\nStopping due to NaN loss.")
                return model, {"loss": losses, "grad_norm": grad_norms, "lr": lrs}
        # --- END DEBUG ---
        
        loss.backward()
        
        # Check for NaN gradients before clipping
        has_nan_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print(f"NaN gradient detected in {name} before clipping")
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print("Skipping optimizer step due to NaN gradients")
            continue
            
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = float(loss.item())
        grad_norm_val = float(grad_norm)
        lr_val = float(lr)

        losses.append(loss_val)
        grad_norms.append(grad_norm_val)
        lrs.append(lr_val)

        if step == 1 or step % 10 == 0:
            progress.set_description(f"step {step}/{total_steps or '?'} loss={loss_val:.4f}")

        # Log to Weights & Biases if a run is active.
        try:
            wandb.log(
                {
                    "step": step,
                    "train/loss": loss_val,
                    "train/lr": lr_val,
                    "train/grad_norm": grad_norm_val,
                }
            )
        except Exception:
            # Keep training even if wandb is not initialized or fails.
            pass

    return model, {"loss": losses, "grad_norm": grad_norms, "lr": lrs}


def main() -> None:
    config = Config()
    device = get_default_device()

    run = wandb.init(
        project="nanoTabPFN",
        name="nanoTabPFN-train",
        config=asdict(config),
        dir=str(HERE),
    )
    run_dir = Path(run.dir) if run is not None else HERE

    set_seed(config.training.seed)

    h5_path = config.training.resolved_h5_path()
    print(f"\nScanning H5 file for corrupted tasks: {h5_path}")
    
    from nanoTabPFN.data import find_corrupted_tasks
    corrupted_tasks = find_corrupted_tasks(h5_path)
    
    if corrupted_tasks:
        print(f"Found {len(corrupted_tasks)} corrupted tasks that will be skipped:")
        print(f"  Task indices: {sorted(corrupted_tasks)}")
    else:
        print("No corrupted tasks found.")
    
    model = NanoTabPFNModel(
        num_features=config.model.num_features,
        num_outputs=config.model.num_outputs,
        embedding_size=config.model.embedding_size,
        num_attention_heads=config.model.num_attention_heads,
        num_layers=config.model.num_layers,
    )

    # Build diffusion process (cosine schedule) on the same device.
    betas = cosine_schedule(
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        timesteps=config.diffusion.timesteps,
    ).to(device)
    process = GaussianDiffusion(betas)

    prior = PriorDumpDataLoader(
        h5_path,
        num_steps=config.training.total_steps,
        batch_size=config.training.batch_size,
        p_ctx=config.training.p_ctx,
        device=device,
        seed=config.training.seed,
    )

    model, logs = train(
        model,
        prior,
        process,
        num_timesteps=config.diffusion.timesteps,
        num_classes=config.model.num_outputs,
        config=config,
        device=device,
        loss_type="l1",
    )

    # Save a checkpoint into the current W&B run directory.
    ckpt_path = run_dir / "model.pt"
    torch.save(
        {
            "config": asdict(config),
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
