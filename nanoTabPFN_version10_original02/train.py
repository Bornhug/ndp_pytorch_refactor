from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bar_distribution import (
    DEFAULT_BORDERS_PATH,
    DEFAULT_NUM_BARS,
    FullSupportBarDistribution,
    border_asset_metadata,
    build_distribution,
)
from model import NanoTabPFNModel

try:
    import schedulefree
except Exception:
    schedulefree = None

try:
    import wandb
except Exception:
    wandb = None


def set_randomness_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def _save_checkpoint(
    model: NanoTabPFNModel,
    checkpoint_dir: Path,
    step: int,
    model_config: Dict[str, object],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"model_step_{step:07d}.pt"
    payload = {
        "state_dict": model.state_dict(),
        "config": {"model": model_config},
        "step": int(step),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _save_final_checkpoint(
    model: NanoTabPFNModel,
    checkpoint_dir: Path,
    step: int,
    model_config: Dict[str, object],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model_final.pt"
    payload = {
        "state_dict": model.state_dict(),
        "config": {"model": model_config},
        "step": int(step),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _compute_grad_norm(parameters) -> torch.Tensor:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    total = torch.zeros((), device=params[0].grad.device)
    for param in params:
        total = total + param.grad.data.norm(2).pow(2)
    return total.sqrt()


def _num_micro_batches(num_items: int, micro_batch_size: int) -> int:
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be > 0, got {micro_batch_size}")
    if num_items <= 0:
        return 0
    return (int(num_items) + int(micro_batch_size) - 1) // int(micro_batch_size)


def _iter_micro_slices(num_items: int, micro_batch_size: int):
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be > 0, got {micro_batch_size}")
    for start in range(0, int(num_items), int(micro_batch_size)):
        yield slice(start, min(start + int(micro_batch_size), int(num_items)))


def _scaled_micro_loss(loss_mat: torch.Tensor, total_target_elements: int) -> torch.Tensor:
    if total_target_elements <= 0:
        raise ValueError(
            f"total_target_elements must be > 0, got {total_target_elements}"
        )
    return loss_mat.sum() / float(total_target_elements)


def _validate_and_prepare_tasks(
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    train_test_split_index: int,
    *,
    std_min: float,
    z_max: float,
    hard_z_threshold: float,
    max_hard_z_frac: float,
) -> Dict[str, object]:
    batch_size = int(y_all.shape[0])
    result: Dict[str, object] = {
        "total_tasks": batch_size,
        "valid_tasks": 0,
        "filtered_tiny_std_tasks": 0,
        "filtered_extreme_tasks": 0,
        "filtered_nonfinite_tasks": 0,
        "filtered_invalid_split_tasks": 0,
        "clipped_target_values": 0,
        "x_valid": None,
        "y_train_norm_valid": None,
        "y_target_norm_valid": None,
    }

    seq_len = int(y_all.shape[1])
    if train_test_split_index <= 0 or train_test_split_index >= seq_len:
        result["filtered_invalid_split_tasks"] = batch_size
        return result

    y_train = y_all[:, :train_test_split_index]
    y_target = y_all[:, train_test_split_index:]

    y_mean = torch.mean(y_train, dim=1, keepdim=True)
    y_std = torch.std(y_train, dim=1, keepdim=True, unbiased=False)
    y_std_safe = y_std.clamp_min(float(std_min))
    y_train_norm = (y_train - y_mean) / y_std_safe
    y_target_norm_raw = (y_target - y_mean) / y_std_safe

    tiny_std_mask = y_std.squeeze(-1) < float(std_min)
    hard_outlier_frac = (
        (y_target_norm_raw.abs() > float(hard_z_threshold))
        .to(torch.float32)
        .mean(dim=1)
    )
    extreme_mask = hard_outlier_frac > float(max_hard_z_frac)
    finite_mask = (
        torch.isfinite(x_all).all(dim=(1, 2))
        & torch.isfinite(y_train_norm).all(dim=1)
        & torch.isfinite(y_target_norm_raw).all(dim=1)
    )
    valid_task_mask = (~tiny_std_mask) & (~extreme_mask) & finite_mask

    result["valid_tasks"] = int(valid_task_mask.sum().item())
    result["filtered_tiny_std_tasks"] = int(tiny_std_mask.sum().item())
    result["filtered_extreme_tasks"] = int(extreme_mask.sum().item())
    result["filtered_nonfinite_tasks"] = int((~finite_mask).sum().item())
    result["clipped_target_values"] = int(
        (y_target_norm_raw.abs() > float(z_max)).sum().item()
    )
    y_train_norm = y_train_norm.clamp(min=-float(z_max), max=float(z_max))
    y_target_norm = y_target_norm_raw.clamp(min=-float(z_max), max=float(z_max))
    result["x_valid"] = x_all[valid_task_mask]
    result["y_train_norm_valid"] = y_train_norm[valid_task_mask]
    result["y_target_norm_valid"] = y_target_norm[valid_task_mask]
    return result


def train(
    model: NanoTabPFNModel,
    prior: DataLoader,
    lr: float = 1e-4,
    device: torch.device | None = None,
    std_min: float = 5e-2,
    z_max: float = 5.0,
    hard_z_threshold: float = 5.0,
    max_hard_z_frac: float = 0.01,
    bar_distribution: FullSupportBarDistribution | None = None,
    checkpoint_interval: int = 300,
    checkpoint_dir: Path | None = None,
    model_config: Optional[Dict[str, object]] = None,
    log_every: int = 10,
    micro_batch_size: int = 4,
    wandb_run=None,
):
    """Train nanoTabPFN for regression on a prior loader."""
    if not device:
        device = torch.device(get_default_device())
    micro_batch_size = int(micro_batch_size)
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be > 0, got {micro_batch_size}")
    model.to(device)

    if schedulefree is not None:
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(), lr=lr, weight_decay=0.0
        )
        has_schedulefree = True
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        has_schedulefree = False

    criterion = (
        bar_distribution
        if bar_distribution is not None
        else build_distribution(expected_num_bars=model.num_bars)
    ).to(device)

    model.train()
    if has_schedulefree:
        optimizer.train()

    train_time = 0.0
    last_step = 0
    total_tasks = 0
    total_valid_tasks = 0
    filtered_tiny_std_tasks = 0
    filtered_extreme_tasks = 0
    filtered_nonfinite_tasks = 0
    filtered_invalid_split_tasks = 0
    clipped_target_values = 0
    skipped_update_steps = 0

    try:
        for step, full_data in enumerate(prior):
            step_num = step + 1
            last_step = step_num
            step_start_time = time.time()
            train_test_split_index = int(full_data["train_test_split_index"])

            x_all = full_data["x"].to(device).to(torch.float32)
            y_all = full_data["y"].to(device).to(torch.float32)
            optimizer.zero_grad(set_to_none=True)

            task_batch = _validate_and_prepare_tasks(
                x_all,
                y_all,
                train_test_split_index,
                std_min=float(std_min),
                z_max=float(z_max),
                hard_z_threshold=float(hard_z_threshold),
                max_hard_z_frac=float(max_hard_z_frac),
            )

            batch_total_tasks = int(task_batch["total_tasks"])
            batch_valid_tasks = int(task_batch["valid_tasks"])
            batch_filtered_tiny_std = int(task_batch["filtered_tiny_std_tasks"])
            batch_filtered_extreme = int(task_batch["filtered_extreme_tasks"])
            batch_filtered_nonfinite = int(task_batch["filtered_nonfinite_tasks"])
            batch_filtered_invalid_split = int(
                task_batch["filtered_invalid_split_tasks"]
            )
            batch_clipped_values = int(task_batch["clipped_target_values"])

            total_tasks += batch_total_tasks
            total_valid_tasks += batch_valid_tasks
            filtered_tiny_std_tasks += batch_filtered_tiny_std
            filtered_extreme_tasks += batch_filtered_extreme
            filtered_nonfinite_tasks += batch_filtered_nonfinite
            filtered_invalid_split_tasks += batch_filtered_invalid_split
            clipped_target_values += batch_clipped_values

            did_update = False
            total_loss = float("nan")
            grad_norm = float("nan")
            grad_norm_postclip = float("nan")
            current_lr = float(optimizer.param_groups[0]["lr"])
            batch_num_micro_batches = 0
            if batch_valid_tasks > 0:
                x_valid = task_batch["x_valid"]
                y_train_norm_valid = task_batch["y_train_norm_valid"]
                y_target_norm_valid = task_batch["y_target_norm_valid"]
                total_target_elements = int(y_target_norm_valid.numel())
                batch_num_micro_batches = _num_micro_batches(
                    batch_valid_tasks,
                    micro_batch_size,
                )

                loss_sum = 0.0
                skip_update = False
                for micro_slice in _iter_micro_slices(
                    batch_valid_tasks,
                    micro_batch_size,
                ):
                    x_micro = x_valid[micro_slice]
                    y_train_micro = y_train_norm_valid[micro_slice]
                    y_target_micro = y_target_norm_valid[micro_slice]

                    output_logits = model(
                        (x_micro, y_train_micro),
                        train_test_split_index=train_test_split_index,
                    )

                    if not torch.isfinite(output_logits).all():
                        skip_update = True
                        break

                    loss_mat = criterion(output_logits, y_target_micro)
                    loss = _scaled_micro_loss(loss_mat, total_target_elements)
                    if not torch.isfinite(loss):
                        skip_update = True
                        break

                    loss_sum += float(loss_mat.detach().sum().cpu().item())
                    loss.backward()

                if skip_update:
                    skipped_update_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                else:
                    grad_norm = float(
                        _compute_grad_norm(list(model.parameters()))
                        .detach()
                        .cpu()
                        .item()
                    )
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grad_norm_postclip = float(
                        _compute_grad_norm(list(model.parameters()))
                        .detach()
                        .cpu()
                        .item()
                    )
                    optimizer.step()
                    total_loss = float(loss_sum / float(total_target_elements))
                    did_update = True
            else:
                skipped_update_steps += 1

            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": total_loss,
                        "train/lr": current_lr,
                        "train/grad_norm": grad_norm,
                        "train/grad_norm_postclip": grad_norm_postclip,
                        "train/step_time_sec": float(step_train_duration),
                        "train/elapsed_sec": float(train_time),
                        "train/step": step_num,
                        "train/valid_tasks": batch_valid_tasks,
                        "train/total_tasks": batch_total_tasks,
                        "train/filtered_tiny_std_tasks": batch_filtered_tiny_std,
                        "train/filtered_extreme_tasks": batch_filtered_extreme,
                        "train/filtered_nonfinite_tasks": batch_filtered_nonfinite,
                        "train/filtered_invalid_split_tasks": batch_filtered_invalid_split,
                        "train/clipped_target_values": batch_clipped_values,
                        "train/skipped_update_step": 0 if did_update else 1,
                        "train/micro_batch_size": int(micro_batch_size),
                        "train/num_micro_batches": int(batch_num_micro_batches),
                    },
                    step=step_num,
                )

            if log_every > 0 and step_num % log_every == 0:
                print(
                    f"step {step_num:6d} | time {train_time:8.1f}s | "
                    f"loss {total_loss:10.6f} | lr {current_lr:9.2e} | grad {grad_norm:8.3f} | "
                    f"valid {batch_valid_tasks:3d}/{batch_total_tasks:3d} | "
                    f"micro {batch_num_micro_batches:2d}x{micro_batch_size:<2d} | "
                    f"tiny_std {batch_filtered_tiny_std:2d} | extreme {batch_filtered_extreme:2d} | "
                    f"nonfinite {batch_filtered_nonfinite:2d} | invalid_split {batch_filtered_invalid_split:2d} | "
                    f"clipped {batch_clipped_values:4d}",
                    flush=True,
                )

            if (
                checkpoint_interval > 0
                and checkpoint_dir is not None
                and model_config is not None
                and step_num % checkpoint_interval == 0
            ):
                ckpt_path = _save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=step_num,
                    model_config=model_config,
                )
                print(f"Saved checkpoint: {ckpt_path}", flush=True)

    except KeyboardInterrupt:
        print("Training interrupted by user.", flush=True)

    print(
        "Training stability summary | "
        f"tasks_seen={total_tasks} | valid_tasks={total_valid_tasks} | "
        f"filtered_tiny_std={filtered_tiny_std_tasks} | "
        f"filtered_extreme={filtered_extreme_tasks} | "
        f"filtered_nonfinite={filtered_nonfinite_tasks} | "
        f"filtered_invalid_split={filtered_invalid_split_tasks} | "
        f"clipped_targets={clipped_target_values} | "
        f"skipped_update_steps={skipped_update_steps}",
        flush=True,
    )
    if wandb_run is not None:
        wandb_run.summary["train/tasks_seen"] = int(total_tasks)
        wandb_run.summary["train/valid_tasks"] = int(total_valid_tasks)
        wandb_run.summary["train/filtered_tiny_std_tasks"] = int(filtered_tiny_std_tasks)
        wandb_run.summary["train/filtered_extreme_tasks"] = int(filtered_extreme_tasks)
        wandb_run.summary["train/filtered_nonfinite_tasks"] = int(filtered_nonfinite_tasks)
        wandb_run.summary["train/filtered_invalid_split_tasks"] = int(filtered_invalid_split_tasks)
        wandb_run.summary["train/clipped_target_values"] = int(clipped_target_values)
        wandb_run.summary["train/skipped_update_steps"] = int(skipped_update_steps)

    return model, last_step


class PriorDumpDataLoader(DataLoader):
    """Loads TabICL pre-generated prior files (batch_*.pt)."""

    def __init__(
        self,
        prior_dir,
        num_steps,
        batch_size,
        device=None,
        shuffle_files: bool = False,
        seed: int = 0,
        skip_nonfinite: bool = True,
    ):
        self.prior_dir = Path(prior_dir)
        self.num_steps = int(num_steps)
        self.batch_size = int(batch_size)
        self.device = device if device is not None else get_default_device()
        self.shuffle_files = bool(shuffle_files)
        self.skip_nonfinite = bool(skip_nonfinite)
        self.pointer = 0
        self.skipped_nonfinite = 0

        if not self.prior_dir.is_dir():
            raise FileNotFoundError(f"Prior directory not found: {self.prior_dir}")

        self.files = sorted(self.prior_dir.glob("batch_*.pt"))
        if not self.files:
            raise RuntimeError(f"No batch_*.pt files found in {self.prior_dir}")

        self.rng = np.random.RandomState(seed)
        self.order = np.arange(len(self.files), dtype=np.int64)
        if self.shuffle_files and len(self.order) > 1:
            self.rng.shuffle(self.order)

    def _next_file(self) -> Path:
        if self.pointer >= len(self.order):
            self.pointer = 0
            if self.shuffle_files and len(self.order) > 1:
                self.rng.shuffle(self.order)
        idx = int(self.order[self.pointer])
        self.pointer += 1
        return self.files[idx]

    def _load_batch_from_file(self, batch_path: Path):
        payload = torch.load(batch_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload in {batch_path}, got {type(payload)}")

        x_flat = torch.as_tensor(payload["X"], dtype=torch.float32).reshape(-1)
        y = torch.as_tensor(payload["y"], dtype=torch.float32)
        d = torch.as_tensor(payload["d"], dtype=torch.long).reshape(-1)
        seq_lens = torch.as_tensor(payload["seq_lens"], dtype=torch.long).reshape(-1)
        train_sizes = torch.as_tensor(payload["train_sizes"], dtype=torch.long).reshape(-1)

        if y.ndim != 2:
            raise ValueError(f"Expected y shape [B,N], got {tuple(y.shape)} in {batch_path}")

        payload_bsz = int(y.shape[0])
        if payload_bsz < self.batch_size:
            raise ValueError(
                f"Unexpected batch size in {batch_path}: got {payload_bsz}, "
                f"expected at least {self.batch_size}"
            )
        if payload_bsz > self.batch_size:
            y = y[: self.batch_size]
            d = d[: self.batch_size]
            seq_lens = seq_lens[: self.batch_size]
            train_sizes = train_sizes[: self.batch_size]
        bsz = int(y.shape[0])

        unique_seq = torch.unique(seq_lens)
        unique_split = torch.unique(train_sizes)
        if unique_seq.numel() != 1:
            raise ValueError(
                f"Expected fixed seq_len per file in {batch_path}, got {unique_seq.tolist()}"
            )
        if unique_split.numel() != 1:
            raise ValueError(
                f"Expected fixed train_size per file in {batch_path}, got {unique_split.tolist()}"
            )

        seq_len = int(unique_seq[0].item())
        split_idx = int(unique_split[0].item())
        if split_idx <= 0 or split_idx >= seq_len:
            raise ValueError(
                f"Invalid train_test_split_index={split_idx} for seq_len={seq_len} in {batch_path}"
            )

        max_d = int(d.max().item())
        x = torch.zeros((bsz, seq_len, max_d), dtype=torch.float32)

        expected_numel = int((seq_lens * d).sum().item())
        if expected_numel > int(x_flat.numel()):
            raise ValueError(
                f"Flattened X mismatch in {batch_path}: "
                f"expected at least {expected_numel}, got {x_flat.numel()}"
            )

        offset = 0
        for i in range(bsz):
            n_i = int(seq_lens[i].item())
            d_i = int(d[i].item())
            count = n_i * d_i
            xi = x_flat[offset : offset + count].view(n_i, d_i)
            x[i, :n_i, :d_i] = xi
            offset += count

        y = y[:, :seq_len]
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f"Non-finite values found in {batch_path}")

        return {
            "x": x.to(self.device),
            "y": y.to(self.device),
            "train_test_split_index": split_idx,
        }

    def __iter__(self):
        for _ in range(self.num_steps):
            attempts = 0
            while attempts < len(self.files):
                batch_path = self._next_file()
                try:
                    yield self._load_batch_from_file(batch_path)
                    break
                except ValueError as exc:
                    if self.skip_nonfinite and "Non-finite values found" in str(exc):
                        self.skipped_nonfinite += 1
                        attempts += 1
                        print(
                            f"[PriorDumpDataLoader] Skipping non-finite batch file: {batch_path}",
                            flush=True,
                        )
                        continue
                    raise
            else:
                raise RuntimeError(
                    "Unable to find a valid finite batch after scanning all files once."
                )

    def __len__(self):
        return self.num_steps


def parse_args():
    parser = argparse.ArgumentParser(description="Train nanoTabPFN regression (version10_original02).")
    parser.add_argument(
        "--prior-dir",
        type=str,
        default="/mnt/c/Users/apex/Code/Python/tabicl/data/stage1_continuous_y_seed123",
        help="Directory containing TabICL pre-generated batch_*.pt files.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of training updates. If unset, uses num_epochs * number_of_prior_files.",
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=4,
        help=(
            "Number of valid tasks processed per forward/backward pass. "
            "Gradients accumulate so the effective optimizer batch remains --batch-size."
        ),
    )
    parser.add_argument("--shuffle-files", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=get_default_device())
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--std-min",
        type=float,
        default=5e-2,
        help="Skip task when train-context target std is below this threshold.",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=5.0,
        help="Clamp normalized context/target y values to [-z_max, z_max] before loss.",
    )
    parser.add_argument(
        "--hard-z-threshold",
        type=float,
        default=5.0,
        help="Hard outlier threshold on normalized y used for task filtering.",
    )
    parser.add_argument(
        "--max-hard-z-frac",
        type=float,
        default=0.01,
        help="Drop task when fraction(|z| > hard_z_threshold) exceeds this.",
    )
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--embedding-size", type=int, default=96)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--num-bars", type=int, default=DEFAULT_NUM_BARS)
    parser.add_argument(
        "--borders-path",
        type=str,
        default=str(DEFAULT_BORDERS_PATH),
        help="Path to the cached TabPFN v2 normalized regressor bar borders.",
    )

    parser.add_argument("--checkpoint-interval", type=int, default=300)
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="nanoTabPFN_version10_original02")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_randomness_seed(args.seed)

    device = torch.device(args.device)
    border_meta = border_asset_metadata(
        args.borders_path,
        expected_num_bars=int(args.num_bars),
    )
    model_init_config = {
        "embedding_size": int(args.embedding_size),
        "num_attention_heads": int(args.num_attention_heads),
        "mlp_hidden_size": int(args.mlp_hidden_size),
        "num_layers": int(args.num_layers),
        "num_bars": int(args.num_bars),
    }
    model_config: Dict[str, object] = {
        **model_init_config,
        "target_distribution": "FullSupportBarDistribution",
        "z_max": float(args.z_max),
        **border_meta,
    }
    model = NanoTabPFNModel(**model_init_config)
    bar_distribution = build_distribution(
        args.borders_path,
        expected_num_bars=int(args.num_bars),
    )

    prior_dir = Path(args.prior_dir).resolve()
    all_files = sorted(prior_dir.glob("batch_*.pt"))
    if not all_files:
        raise RuntimeError(f"No batch_*.pt files found in {prior_dir}")
    if int(args.num_epochs) <= 0:
        raise ValueError(f"num_epochs must be > 0, got {args.num_epochs}")
    if int(args.batch_size) <= 0:
        raise ValueError(f"batch_size must be > 0, got {args.batch_size}")
    if int(args.micro_batch_size) <= 0:
        raise ValueError(
            f"micro_batch_size must be > 0, got {args.micro_batch_size}"
        )
    steps_per_epoch = len(all_files)
    num_steps = (
        int(args.num_steps)
        if args.num_steps is not None
        else steps_per_epoch * int(args.num_epochs)
    )
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")

    print(f"Prior directory: {prior_dir}", flush=True)
    print(
        f"Found {len(all_files)} prior files | steps/epoch {steps_per_epoch} | "
        f"epochs {int(args.num_epochs)} | training updates {num_steps}",
        flush=True,
    )
    print(
        f"Effective batch_size {int(args.batch_size)} | "
        f"micro_batch_size {int(args.micro_batch_size)}",
        flush=True,
    )
    prior = PriorDumpDataLoader(
        str(prior_dir),
        num_steps=num_steps,
        batch_size=int(args.batch_size),
        device=device,
        shuffle_files=bool(args.shuffle_files),
        seed=int(args.seed),
        skip_nonfinite=True,
    )

    wandb_run = None
    if not args.disable_wandb:
        if wandb is None:
            raise RuntimeError(
                "wandb is not installed. Install it (`pip install wandb`) or run with --disable-wandb."
            )
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config={
                    "seed": int(args.seed),
                    "device": args.device,
                    "prior_dir": str(prior_dir),
                    "num_epochs": int(args.num_epochs),
                    "steps_per_epoch": int(steps_per_epoch),
                    "num_steps": int(num_steps),
                    "batch_size": int(args.batch_size),
                    "micro_batch_size": int(args.micro_batch_size),
                    "shuffle_files": bool(args.shuffle_files),
                    "lr": float(args.lr),
                    "std_min": float(args.std_min),
                    "z_max": float(args.z_max),
                    "hard_z_threshold": float(args.hard_z_threshold),
                    "max_hard_z_frac": float(args.max_hard_z_frac),
                    "log_every": int(args.log_every),
                    "checkpoint_interval": int(args.checkpoint_interval),
                    "model": model_config,
                },
            )

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).resolve()
    elif wandb_run is not None:
        checkpoint_dir = Path(wandb_run.dir).resolve() / "checkpoints"
    else:
        checkpoint_dir = HERE / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}", flush=True)

    model, last_step = train(
        model=model,
        prior=prior,
        lr=float(args.lr),
        device=device,
        std_min=float(args.std_min),
        z_max=float(args.z_max),
        hard_z_threshold=float(args.hard_z_threshold),
        max_hard_z_frac=float(args.max_hard_z_frac),
        bar_distribution=bar_distribution,
        checkpoint_interval=int(args.checkpoint_interval),
        checkpoint_dir=checkpoint_dir,
        model_config=model_config,
        log_every=int(args.log_every),
        micro_batch_size=int(args.micro_batch_size),
        wandb_run=wandb_run,
    )

    final_ckpt_path = _save_final_checkpoint(
        model=model,
        checkpoint_dir=checkpoint_dir,
        step=last_step,
        model_config=model_config,
    )
    print(f"Saved final checkpoint: {final_ckpt_path}", flush=True)

    if prior.skipped_nonfinite > 0:
        print(
            f"Skipped non-finite prior files during training: {prior.skipped_nonfinite}",
            flush=True,
        )

    if wandb_run is not None:
        wandb_run.summary["final_checkpoint"] = str(final_ckpt_path)
        wandb_run.summary["skipped_nonfinite_batches"] = int(prior.skipped_nonfinite)
        wandb_run.finish()


if __name__ == "__main__":
    main()
