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

# Make the project root importable so we can access local modules.
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ndp_discrete.process import (  # noqa: E402
    D3PM,
    D3PMSchedule,
    loss as diffusion_loss,
)
from ndp_discrete.types import Batch  # noqa: E402
from ndp_discrete.model import BiDimensionalAttentionModel  # noqa: E402
from config import Config  # noqa: E402
from data import PriorDumpDataLoader, find_corrupted_tasks  # noqa: E402


class BiDimClassificationEpsNet(BiDimensionalAttentionModel):
    """
    Bi-dimensional epsilon network for classification.

    Uses the base bidimensional attention model with scalar labels; we keep
    track of the configured class count for metadata only.
    """

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        num_heads: int,
        output_dim: int,
        init_zero: bool = True,
    ) -> None:
        super().__init__(
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            output_dim=output_dim,
            label_dim=output_dim,
            init_zero=init_zero,
        )
        self.num_classes = int(output_dim)


class NanoTabPFNModel(nn.Module):
    """
    Diffusion epsilon-model for the nanoTabPFN prior, wrapping a
    classification-aware bidimensional attention model with feature
    z-score normalisation.

    The forward signature matches `BiDimensionalAttentionModel` so it can
    be used with `neural_diffusion_processes_original.process.loss`.
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
        self.num_classes = self.num_outputs

        # Core bidimensional attention epsilon-network for scalar labels.
        self.core = BiDimClassificationEpsNet(
            n_layers=num_layers,
            hidden_dim=embedding_size,
            num_heads=num_attention_heads,
            output_dim=self.num_outputs,
        )

    def _compute_normalization_stats(
        self, x_target: torch.Tensor | None, x_context: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """
        Compute per-task normalization statistics from both context and target features.
        """
        if x_target is None and x_context is None:
            return None, None

        to_stack = []
        if x_target is not None:
            if x_target.ndim == 2:
                x_target = x_target.unsqueeze(0)
            to_stack.append(x_target)
        if x_context is not None:
            if x_context.ndim == 2:
                x_context = x_context.unsqueeze(0)
            to_stack.append(x_context)

        x_all = torch.cat(to_stack, dim=1) if len(to_stack) > 1 else to_stack[0]

        mean = x_all.mean(dim=1, keepdim=True)
        var = x_all.var(dim=1, keepdim=True, unbiased=False)
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
        x_target: torch.Tensor,
        y_target: torch.Tensor,
        t: torch.Tensor,
        mask_target: torch.Tensor | None = None,
        x_context: torch.Tensor | None = None,
        y_context: torch.Tensor | None = None,
        mask_context: torch.Tensor | None = None,
        *,
        key: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Conditional forward: predict target logits conditioned on context.

        Args:
            x_target: [B, N_t, F] or [N_t, F]
            y_target: [B, N_t, V] (noisy tokens) or [N_t, V]
            t: [B] (or scalar) timesteps
            mask_target: Optional [B, N_t] indicator (1 = mask)
            x_context/y_context: optional context tensors
            mask_context: Optional [B, N_c] indicator (1 = mask)
        """
        added_batch_dim = False
        if x_target.ndim == 2:
            x_target = x_target.unsqueeze(0)
            added_batch_dim = True
        if y_target.ndim == 2:
            y_target = y_target.unsqueeze(0)
        if t.ndim == 0:
            t = t.unsqueeze(0)

        if x_context is not None and x_context.ndim == 2:
            x_context = x_context.unsqueeze(0)
        if y_context is not None and y_context.ndim == 2:
            y_context = y_context.unsqueeze(0)

        mean, std = self._compute_normalization_stats(x_target, x_context)
        x_target_norm = self._normalize(x_target, mean, std)
        x_context_norm = self._normalize(x_context, mean, std) if x_context is not None else None

        mask_target_tensor = None
        if mask_target is not None:
            mask_target_tensor = mask_target.to(device=x_target.device, dtype=torch.float32)
            if mask_target_tensor.ndim == 1:
                mask_target_tensor = mask_target_tensor.unsqueeze(0)
        mask_context_tensor = None
        if mask_context is not None:
            mask_context_tensor = mask_context.to(device=x_target.device, dtype=torch.float32)
            if mask_context_tensor.ndim == 1:
                mask_context_tensor = mask_context_tensor.unsqueeze(0)

        out = self.core(
            x_target_norm,
            y_target,
            t.to(x_target.device, dtype=torch.float32),
            mask_target_tensor,
            x_context_norm,
            y_context,
            mask_context_tensor,
        )

        if added_batch_dim:
            out = out.squeeze(0)
        return out


def get_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_one_hot_labels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert scalar class labels to one-hot vectors, leaving already-encoded
    tensors unchanged.
    """
    if y.ndim > 0 and y.size(-1) == num_classes:
        return y.to(dtype=torch.float32)
    y_idx = y.to(dtype=torch.long)
    if y_idx.ndim > 0 and y_idx.size(-1) == 1:
        y_idx = y_idx.squeeze(-1)
    return F.one_hot(y_idx, num_classes=num_classes).to(dtype=torch.float32)


def compute_lr(config: Config, step: int, *, steps_per_epoch: int) -> float:
    """
    Cosine schedule with linear warmup, mirroring the regression config.
    """
    warmup_steps = steps_per_epoch * config.optim.num_warmup_epochs
    decay_steps = steps_per_epoch * config.optim.num_decay_epochs
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
    process: D3PM,
    *,
    num_timesteps: int,
    config: Config,
    device: torch.device | None = None,
    loss_type: str = "l1",
    steps_per_epoch: int,
    checkpoint_dir: Path,
    checkpoint_interval: int = 10_000,
) -> Tuple[NanoTabPFNModel, Dict[str, List[float]]]:
    """
    Diffusion-based training loop over the prior dump.
    """
    device = device or get_default_device()
    model.to(device)
    model.train()

    diffusion_key = torch.Generator(device=device)
    diffusion_key.manual_seed(config.training.seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.init_lr,
        weight_decay=config.optim.weight_decay,
    )
    losses: List[float] = []
    grad_norms: List[float] = []
    lrs: List[float] = []

    total_steps = steps_per_epoch * config.training.num_epochs
    if total_steps <= 0:
        raise ValueError("Total training steps must be positive.")

    prior.num_steps = total_steps

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(step_idx: int) -> None:
        ckpt_path = checkpoint_dir / f"model_step_{step_idx}.pt"
        torch.save(
            {
                "config": asdict(config),
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step_idx,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")

    progress = tqdm(enumerate(prior, start=1), total=total_steps, desc="training")

    for step, batch in progress:
        optimizer.zero_grad(set_to_none=True)

        # Move to device and one-hot encode labels.
        x_tgt = torch.as_tensor(batch.x_target, device=device, dtype=torch.float32)
        x_ctx = torch.as_tensor(batch.x_context, device=device, dtype=torch.float32)

        y_tgt_raw = torch.as_tensor(batch.y_target, device=device)
        y_ctx_raw = torch.as_tensor(batch.y_context, device=device)
        # Pad one-hot to vocab_size (adds absorbing channel if enabled)
        y_tgt = to_one_hot_labels(y_tgt_raw, model.num_classes)
        y_ctx = to_one_hot_labels(y_ctx_raw, model.num_classes)

        batch_t = Batch(
            x_target=x_tgt,
            y_target=y_tgt,
            x_context=x_ctx,
            y_context=y_ctx,
            mask_target=None,
            mask_context=None,
        )

        lr = compute_lr(config, step, steps_per_epoch=steps_per_epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_out = diffusion_loss(
            process,
            model,
            batch_t,
            diffusion_key,
            num_timesteps=num_timesteps,
            loss_type=loss_type,
        )
        # diffusion_loss may return just total loss or (total, kl, ce) when hybrid
        if isinstance(loss_out, tuple):
            loss = loss_out[0]
            loss_kl = loss_out[1] if len(loss_out) > 1 else None
            loss_ce = loss_out[2] if len(loss_out) > 2 else None
        else:
            loss = loss_out
            loss_kl = None
            loss_ce = None

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
            log_dict = {
                "step": step,
                "train/loss": loss_val,
                "train/lr": lr_val,
                "train/grad_norm": grad_norm_val,
            }
            if loss_kl is not None:
                log_dict["train/loss_kl"] = float(loss_kl.item())
            if loss_ce is not None:
                log_dict["train/loss_ce"] = float(loss_ce.item())
            wandb.log(log_dict)
        except Exception:
            # Keep training even if wandb is not initialized or fails.
            pass

        if checkpoint_interval > 0 and step % checkpoint_interval == 0:
            save_checkpoint(step)

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
    
    corrupted_tasks = find_corrupted_tasks(h5_path)
    
    if corrupted_tasks:
        print(f"Found {len(corrupted_tasks)} corrupted tasks that will be skipped:")
        print(f"  Task indices: {sorted(corrupted_tasks)}")
    else:
        print("No corrupted tasks found.")
    
    vocab_size = config.model.num_outputs + (1 if config.diffusion.use_absorbing else 0)

    model = NanoTabPFNModel(
        num_features=config.model.num_features,
        num_outputs=vocab_size,
        embedding_size=config.model.embedding_size,
        num_attention_heads=config.model.num_attention_heads,
        num_layers=config.model.num_layers,
    )

    # Build discrete diffusion process on the same device, honoring config.
    transition_type = "absorbing" if config.diffusion.use_absorbing else getattr(config.diffusion, "transition_mat_type", "uniform")
    beta_type = "jsd" if config.diffusion.use_absorbing else config.diffusion.schedule

    schedule = D3PMSchedule.make_uniform(
        T=config.diffusion.timesteps,
        vocab_size=vocab_size,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_type=beta_type,
        transition_mat_type=transition_type,
        device=device,
        dtype=torch.float32,
    )
    process = D3PM(
        model,
        schedule,
        transition_mat_type=transition_type,
    )

    prior = PriorDumpDataLoader(
        h5_path,
        num_steps=1,
        batch_size=config.training.batch_size,
        p_ctx=config.training.p_ctx,
        device=device,
        seed=config.training.seed,
        skip_tasks=corrupted_tasks,
    )

    steps_per_epoch = prior.steps_per_epoch
    if steps_per_epoch == 0:
        raise RuntimeError("No valid tasks found for training.")
    total_steps = steps_per_epoch * config.training.num_epochs
    prior.num_steps = total_steps
    print(f"Training with {steps_per_epoch} steps/epoch ({total_steps} total steps).")

    model, logs = train(
        model,
        prior,
        process,
        num_timesteps=config.diffusion.timesteps,
        config=config,
        device=device,
        loss_type="kl",
        steps_per_epoch=steps_per_epoch,
        checkpoint_dir=run_dir,
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
