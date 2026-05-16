"""TabICL-style training loop for regression NDPs with Gaussian diffusion."""

from __future__ import annotations

import copy
import math
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_diffusion_processes.process import (
    GaussianDiffusion,
    cosine_schedule,
    loss as diffusion_loss,
)
from neural_diffusion_processes.types import Batch

from tabicl_style.batching import (
    align_micro_batch,
    pad_nested_batch,
    split_micro_batches,
    validate_micro_batch,
)
from tabicl_style.config import Config
from tabicl_style.data import build_dataloader, build_dataset
from tabicl_style.model import NDPRegressor
from tabicl_style.utils import normalize_y, set_seed, split_context_target


# ---------------------------------------------------------------------------
# Model + process construction
# ---------------------------------------------------------------------------

def build_model_and_process(config: Config, device: torch.device):
    model = NDPRegressor(
        embedding_size=config.model.embedding_size,
        num_attention_heads=config.model.num_attention_heads,
        num_layers=config.model.num_layers,
    ).to(device)

    betas = cosine_schedule(
        config.diffusion.beta_start,
        config.diffusion.beta_end,
        config.diffusion.timesteps,
    ).to(device)
    process = GaussianDiffusion(betas)

    return model, process


# ---------------------------------------------------------------------------
# EMA helpers
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def compute_lr(config: Config, step: int, *, total_steps: int) -> float:
    if total_steps <= 0:
        return config.optimizer.end_lr

    warmup_steps = max(1, int(round(total_steps * 0.10)))
    warmup_steps = min(warmup_steps, total_steps)
    decay_steps = max(1, int(round(total_steps * 0.8)))
    decay_steps = max(0, min(decay_steps, total_steps - warmup_steps))

    init_lr = config.optimizer.init_lr
    peak_lr = config.optimizer.peak_lr
    end_lr = config.optimizer.end_lr

    if step <= warmup_steps:
        if warmup_steps == 0:
            return peak_lr
        alpha = step / float(warmup_steps)
        return init_lr + (peak_lr - init_lr) * alpha

    if decay_steps <= 0:
        return end_lr

    t = min(step - warmup_steps, decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t / float(decay_steps)))
    return end_lr + (peak_lr - end_lr) * cosine


def compute_grad_norm(parameters) -> torch.Tensor:
    if not parameters:
        return torch.tensor(0.0)
    total = torch.zeros((), device=parameters[0].device)
    for p in parameters:
        if p.grad is None:
            continue
        total = total + p.grad.data.norm(2).pow(2)
    return total.sqrt()


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def infer_steps_per_epoch(config: Config) -> int:
    prior_dir = config.training.prior_dir
    if prior_dir:
        p = Path(prior_dir)
        if p.is_dir():
            count = len(list(p.glob("batch_*.pt")))
            if count > 0:
                return count
    spe = config.training.samples_per_epoch
    if spe > 0:
        return max(1, int(spe // config.training.batch_size))
    return 1


def should_save_checkpoint(step: int, total_steps: int, save_every: int) -> bool:
    del total_steps
    if save_every <= 0:
        return False
    return step % save_every == 0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        tc = config.training
        self.device = torch.device(tc.device)

        set_seed(tc.np_seed, tc.torch_seed)

        self.model, self.process = build_model_and_process(config, self.device)
        self.ema = EMA(self.model, decay=config.optimizer.ema_rate)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.init_lr,
            weight_decay=config.optimizer.weight_decay,
        )

        self.amp = bool(tc.amp and "cuda" in tc.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        if self.amp:
            dtype = torch.float16 if tc.dtype == "float16" else torch.float32
            self.amp_ctx = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            self.amp_ctx = nullcontext()

        self.reset_data_iter()

        self.curr_step = 0
        self.diffusion_key = torch.Generator(device=self.device)
        self.diffusion_key.manual_seed(tc.torch_seed)

        self.steps_per_epoch = infer_steps_per_epoch(config)
        self.total_steps = int(self.steps_per_epoch * tc.num_epochs)
        if self.total_steps <= 0:
            raise ValueError("Total training steps must be positive.")

        self.wandb_run = None
        self._wandb_log_failed = False
        if tc.wandb_log:
            if wandb is None:
                print("wandb is not available; skipping logging.")
            elif tc.wandb_mode == "disabled":
                pass
            else:
                self.wandb_run = wandb.init(
                    dir=tc.wandb_dir,
                    project=tc.wandb_project,
                    name=tc.wandb_name,
                    id=tc.wandb_id,
                    config=asdict(config),
                    resume="allow",
                    mode=tc.wandb_mode,
                )

        if tc.checkpoint_path:
            self.load_checkpoint(tc.checkpoint_path)
        elif tc.checkpoint_dir:
            latest = self.get_latest_checkpoint(Path(tc.checkpoint_dir))
            if latest:
                self.load_checkpoint(latest)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def reset_data_iter(self) -> None:
        tc = self.config.training
        dataset = build_dataset(tc)
        self.dataloader = build_dataloader(tc, dataset)
        self.data_iter = iter(self.dataloader)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def get_latest_checkpoint(self, ckpt_dir: Path) -> Path | None:
        if not ckpt_dir.is_dir():
            return None
        ckpts = sorted(ckpt_dir.glob("step-*.pt"))
        return ckpts[-1] if ckpts else None

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
            if "ema_state_dict" in ckpt:
                self.ema.load_state_dict(ckpt["ema_state_dict"])
            if "optimizer_state" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.curr_step = int(ckpt.get("curr_step", 0))
        else:
            self.model.load_state_dict(ckpt)
        print(f"Loaded checkpoint from {path}")

    def save_checkpoint(self, step: int) -> None:
        tc = self.config.training
        if not tc.checkpoint_dir:
            return
        ckpt_dir = Path(tc.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"step-{step}.pt"
        payload = {
            "state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "curr_step": step,
            "config": asdict(self.config),
        }
        torch.save(payload, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _assert_finite(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"non-finite {name}")

    def run_micro_batch(
        self, micro_batch, micro_batch_idx: int, num_micro_batches: int
    ) -> Dict[str, float]:
        del micro_batch_idx
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        micro_X = micro_X.to(self.device, dtype=torch.float32)
        micro_y = micro_y.to(self.device, dtype=torch.float32)
        self._assert_finite("micro_X", micro_X)
        self._assert_finite("micro_y", micro_y)

        x_context, y_context, x_target, y_target = split_context_target(
            micro_X, micro_y, train_size
        )
        tc = self.config.training

        # y normalization: z-score using context stats
        y_context_norm, y_target_norm, _mean, _std = normalize_y(y_context, y_target)
        # y_context_norm / y_target_norm are [B, N, 1]

        y_context_stats = y_context.unsqueeze(-1) if y_context.ndim == 2 else y_context
        raw_std = y_context_stats.std(dim=1, keepdim=True, unbiased=False)  # [B,1,1]

        tiny_std_task = raw_std.squeeze(-1).squeeze(-1) < float(tc.std_min)  # [B]
        hard_outlier_frac = (
            (y_target_norm.abs() > float(tc.hard_z_threshold))
            .to(torch.float32)
            .mean(dim=(1, 2))
        )  # [B]
        extreme_task = hard_outlier_frac > float(tc.max_hard_z_frac)
        finite_task = (
            torch.isfinite(y_context_norm).all(dim=(1, 2))
            & torch.isfinite(y_target_norm).all(dim=(1, 2))
        )
        valid_task = (~tiny_std_task) & (~extreme_task) & finite_task

        clipped_target_values = int(
            (y_target_norm.abs() > float(tc.z_max)).sum().item()
        )
        y_context_norm = y_context_norm.clamp(
            min=-float(tc.z_max), max=float(tc.z_max)
        )
        y_target_norm = y_target_norm.clamp(
            min=-float(tc.z_max), max=float(tc.z_max)
        )

        total_tasks = int(micro_X.shape[0])
        valid_tasks = int(valid_task.sum().item())
        stats = {
            "total_tasks": total_tasks,
            "valid_tasks": valid_tasks,
            "skipped_tiny_std_tasks": int(tiny_std_task.sum().item()),
            "skipped_extreme_tasks": int(extreme_task.sum().item()),
            "skipped_nonfinite_tasks": int((~finite_task).sum().item()),
            "clipped_target_values": clipped_target_values,
        }

        if valid_tasks == 0:
            return {"loss": float("nan"), "skipped_micro": True, **stats}

        x_context = x_context[valid_task]
        x_target = x_target[valid_task]
        y_context_norm = y_context_norm[valid_task]
        y_target_norm = y_target_norm[valid_task]

        mask_context = torch.zeros(
            x_context.shape[:2], device=self.device, dtype=torch.float32
        )
        mask_target = torch.zeros(
            x_target.shape[:2], device=self.device, dtype=torch.float32
        )

        batch = Batch(
            x_target=x_target,
            y_target=y_target_norm,
            x_context=x_context,
            y_context=y_context_norm,
            mask_target=mask_target,
            mask_context=mask_context,
        )

        with self.amp_ctx:
            loss = diffusion_loss(
                self.process,
                self.model,
                batch,
                self.diffusion_key,
                num_timesteps=self.config.diffusion.timesteps,
                loss_type=self.config.training.loss_type,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")

        scaled_loss = loss / float(num_micro_batches)
        if self.amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return {"loss": float(loss.item()), "skipped_micro": False, **stats}

    def run_batch(self, batch, *, step: int) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch = pad_nested_batch(batch)
        micro_batches, num_micro_batches = split_micro_batches(
            batch, self.config.training.micro_batch_size
        )

        results: Dict[str, float] = {
            "loss": 0.0,
            "total_tasks": 0,
            "valid_tasks": 0,
            "skipped_tiny_std_tasks": 0,
            "skipped_extreme_tasks": 0,
            "skipped_nonfinite_tasks": 0,
            "clipped_target_values": 0,
            "skipped_micro_batches": 0,
        }
        failed_batches = 0
        processed = 0
        skip_update = False
        skip_reason = ""

        iterable = micro_batches
        if self.config.training.micro_progress:
            iterable = tqdm(
                micro_batches,
                total=num_micro_batches,
                desc=f"micro {step}",
                leave=False,
            )

        for idx, micro_batch in enumerate(iterable):
            try:
                micro_results = self.run_micro_batch(
                    micro_batch, idx, num_micro_batches
                )
                for key in (
                    "total_tasks",
                    "valid_tasks",
                    "skipped_tiny_std_tasks",
                    "skipped_extreme_tasks",
                    "skipped_nonfinite_tasks",
                    "clipped_target_values",
                ):
                    results[key] += int(micro_results.get(key, 0))
                if micro_results.get("skipped_micro", False):
                    results["skipped_micro_batches"] += 1
                    continue
                results["loss"] += micro_results["loss"]
                processed += 1
            except torch.cuda.OutOfMemoryError:
                print(
                    f"Warning: OOM in micro-batch {idx+1}/{num_micro_batches}, skipping."
                )
                torch.cuda.empty_cache()
                failed_batches += 1
            except FloatingPointError as exc:
                print(
                    f"Warning: non-finite values in micro-batch {idx+1}/{num_micro_batches}, "
                    "skipping update."
                )
                skip_update = True
                skip_reason = str(exc)
                break

        if failed_batches / max(1, num_micro_batches) > 0.1:
            raise RuntimeError("Too many micro-batches failed due to OOM.")

        if skip_update:
            self.optimizer.zero_grad(set_to_none=True)
            lr = compute_lr(self.config, step, total_steps=self.total_steps)
            return {
                "loss": float("nan"),
                "skipped": True,
                "skip_reason": skip_reason,
                "lr": float(lr),
                "step": step,
                "total_tasks": int(results["total_tasks"]),
                "valid_tasks": int(results["valid_tasks"]),
                "skipped_tiny_std_tasks": int(results["skipped_tiny_std_tasks"]),
                "skipped_extreme_tasks": int(results["skipped_extreme_tasks"]),
                "skipped_nonfinite_tasks": int(results["skipped_nonfinite_tasks"]),
                "clipped_target_values": int(results["clipped_target_values"]),
                "skipped_micro_batches": int(results["skipped_micro_batches"]),
            }

        if processed == 0:
            self.optimizer.zero_grad(set_to_none=True)
            lr = compute_lr(self.config, step, total_steps=self.total_steps)
            return {
                "loss": float("nan"),
                "skipped": True,
                "skip_reason": "no valid micro-batches",
                "lr": float(lr),
                "step": step,
                "total_tasks": int(results["total_tasks"]),
                "valid_tasks": int(results["valid_tasks"]),
                "skipped_tiny_std_tasks": int(results["skipped_tiny_std_tasks"]),
                "skipped_extreme_tasks": int(results["skipped_extreme_tasks"]),
                "skipped_nonfinite_tasks": int(results["skipped_nonfinite_tasks"]),
                "clipped_target_values": int(results["clipped_target_values"]),
                "skipped_micro_batches": int(results["skipped_micro_batches"]),
            }

        # Gradient unscale + clip
        if self.amp:
            self.scaler.unscale_(self.optimizer)
        if self.config.training.gradient_clipping > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clipping
            )
        else:
            grad_norm = compute_grad_norm(list(self.model.parameters()))

        # LR schedule
        lr = compute_lr(self.config, step, total_steps=self.total_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        # Optimizer step
        if self.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # EMA update
        self.ema.update(self.model)

        if processed > 0:
            results["loss"] /= processed
        results["grad_norm"] = float(grad_norm)
        results["lr"] = float(lr)
        results["step"] = step
        return results

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        tc = self.config.training
        progress = tqdm(
            total=self.total_steps,
            initial=self.curr_step,
            desc="training",
        )
        start_epoch = self.curr_step // self.steps_per_epoch

        for epoch in range(start_epoch, tc.num_epochs):
            self.reset_data_iter()
            for step_idx in range(self.steps_per_epoch):
                batch = next(self.data_iter)
                step = epoch * self.steps_per_epoch + step_idx + 1
                if step <= self.curr_step:
                    continue

                results = self.run_batch(batch, step=step)
                self.curr_step = step
                progress.update(1)

                if results.get("skipped"):
                    progress.set_description(
                        f"step {step}/{self.total_steps} skipped"
                    )
                    postfix = {
                        "lr": f"{results.get('lr', 0.0):.6f}",
                        "skip": "1",
                        "valid": (
                            f"{int(results.get('valid_tasks', 0))}/"
                            f"{int(results.get('total_tasks', 0))}"
                        ),
                    }
                else:
                    progress.set_description(
                        f"step {step}/{self.total_steps} loss={results['loss']:.4f}"
                    )
                    postfix = {
                        "lr": f"{results.get('lr', 0.0):.6f}",
                        "grad": f"{results.get('grad_norm', 0.0):.3f}",
                        "valid": (
                            f"{int(results.get('valid_tasks', 0))}/"
                            f"{int(results.get('total_tasks', 0))}"
                        ),
                    }
                progress.set_postfix(postfix)

                if should_save_checkpoint(
                    step, self.total_steps, tc.save_every
                ):
                    self.save_checkpoint(step)

                if self.wandb_run is not None:
                    if results.get("skipped"):
                        log_dict = {
                            "step": step,
                            "train/skipped": 1,
                            "train/lr": float(results.get("lr", 0.0)),
                            "train/valid_tasks": int(results.get("valid_tasks", 0)),
                            "train/total_tasks": int(results.get("total_tasks", 0)),
                            "train/skipped_tiny_std_tasks": int(
                                results.get("skipped_tiny_std_tasks", 0)
                            ),
                            "train/skipped_extreme_tasks": int(
                                results.get("skipped_extreme_tasks", 0)
                            ),
                            "train/skipped_nonfinite_tasks": int(
                                results.get("skipped_nonfinite_tasks", 0)
                            ),
                            "train/clipped_target_values": int(
                                results.get("clipped_target_values", 0)
                            ),
                            "train/skipped_micro_batches": int(
                                results.get("skipped_micro_batches", 0)
                            ),
                        }
                    else:
                        log_dict = {
                            "step": step,
                            "train/loss": float(results["loss"]),
                            "train/lr": float(results.get("lr", 0.0)),
                            "train/grad_norm": float(
                                results.get("grad_norm", 0.0)
                            ),
                            "train/valid_tasks": int(results.get("valid_tasks", 0)),
                            "train/total_tasks": int(results.get("total_tasks", 0)),
                            "train/skipped_tiny_std_tasks": int(
                                results.get("skipped_tiny_std_tasks", 0)
                            ),
                            "train/skipped_extreme_tasks": int(
                                results.get("skipped_extreme_tasks", 0)
                            ),
                            "train/skipped_nonfinite_tasks": int(
                                results.get("skipped_nonfinite_tasks", 0)
                            ),
                            "train/clipped_target_values": int(
                                results.get("clipped_target_values", 0)
                            ),
                            "train/skipped_micro_batches": int(
                                results.get("skipped_micro_batches", 0)
                            ),
                        }
                    try:
                        self.wandb_run.log(log_dict, step=step)
                    except Exception as exc:
                        if not self._wandb_log_failed:
                            print(f"wandb.log failed: {exc}")
                            self._wandb_log_failed = True

                if self.curr_step >= self.total_steps:
                    break
            if self.curr_step >= self.total_steps:
                break


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="TabICL_regression03 NDP training")

    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--loss-type", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--amp", type=str, default="False")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--samples-per-epoch", type=int, default=2**14)
    parser.add_argument(
        "--std-min",
        type=float,
        default=1e-3,
        help="Skip task when context target std is below this threshold.",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=10.0,
        help="Clamp normalized y targets to [-z_max, z_max].",
    )
    parser.add_argument(
        "--hard-z-threshold",
        type=float,
        default=30.0,
        help="Hard outlier threshold on normalized targets for task filtering.",
    )
    parser.add_argument(
        "--max-hard-z-frac",
        type=float,
        default=0.05,
        help="Drop task when fraction(|z| > hard_z_threshold) exceeds this value.",
    )

    # Model
    parser.add_argument("--embedding-size", type=int, default=96)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=5)

    # Diffusion
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--beta-start", type=float, default=3e-4)
    parser.add_argument("--beta-end", type=float, default=0.5)

    # Optimizer
    parser.add_argument("--peak-lr", type=float, default=1e-2)
    parser.add_argument("--init-lr", type=float, default=2e-4)
    parser.add_argument("--end-lr", type=float, default=1e-4)
    parser.add_argument("--ema-rate", type=float, default=0.995)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)

    # Data
    parser.add_argument("--prior-dir", type=str, default=None)
    parser.add_argument("--tabicl-repo", type=str, default=None)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--min-train-size", type=float, default=0.1)
    parser.add_argument("--max-train-size", type=float, default=0.9)

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--save-every",
        type=int,
        default=10000,
        help="Save one checkpoint every N training steps; set <= 0 to disable.",
    )

    # Wandb
    parser.add_argument("--wandb-log", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb_log", action="store_false")
    parser.add_argument("--wandb-project", type=str, default="TabICL-regression03")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="offline")
    parser.add_argument("--wandb-dir", type=str, default=None)

    # Seeds
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    from tabicl_style.config import (
        Config,
        DiffusionConfig,
        ModelConfig,
        OptimizerConfig,
        TrainingConfig,
    )

    config = Config(
        model=ModelConfig(
            embedding_size=args.embedding_size,
            num_attention_heads=args.num_attention_heads,
            num_layers=args.num_layers,
        ),
        diffusion=DiffusionConfig(
            timesteps=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
        ),
        optimizer=OptimizerConfig(
            peak_lr=args.peak_lr,
            init_lr=args.init_lr,
            end_lr=args.end_lr,
            ema_rate=args.ema_rate,
            weight_decay=args.weight_decay,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            num_epochs=args.num_epochs,
            samples_per_epoch=args.samples_per_epoch,
            loss_type=args.loss_type,
            std_min=args.std_min,
            z_max=args.z_max,
            hard_z_threshold=args.hard_z_threshold,
            max_hard_z_frac=args.max_hard_z_frac,
            gradient_clipping=args.gradient_clipping,
            device=args.device,
            amp=args.amp.lower() == "true",
            np_seed=args.seed,
            torch_seed=args.seed,
            num_workers=args.num_workers,
            prior_dir=args.prior_dir,
            tabicl_repo=args.tabicl_repo,
            max_features=args.max_features,
            max_seq_len=args.max_seq_len,
            min_train_size=args.min_train_size,
            max_train_size=args.max_train_size,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_path=args.checkpoint_path,
            save_every=args.save_every,
            wandb_log=args.wandb_log,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            wandb_mode=args.wandb_mode,
            wandb_dir=args.wandb_dir,
        ),
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
