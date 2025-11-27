from __future__ import annotations

import datetime
import math
import pathlib
import random
import string
from dataclasses import asdict
import argparse
import sys

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

# Ensure project root (one level up from regression/) is on PYTHONPATH for local imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import Config
from neural_diffusion_processes.model import BiDimensionalAttentionModel
from neural_diffusion_processes.process import GaussianDiffusion, cosine_schedule, loss as diffusion_loss
from neural_diffusion_processes.types import Batch


EXPERIMENT = "regression-pytorch"
EXPERIMENT_NAME: str | None = None
DATETIME = datetime.datetime.now().strftime("%b%d_%H%M%S")
HERE = pathlib.Path(__file__).parent
LOG_DIR = "logs"


def get_experiment_name(config: Config) -> str:
    del config
    global EXPERIMENT_NAME

    if EXPERIMENT_NAME is None:
        letters = string.ascii_lowercase
        run_id = "".join(random.choice(letters) for _ in range(4))
        EXPERIMENT_NAME = f"{DATETIME}_{run_id}"

    return EXPERIMENT_NAME


def get_experiment_dir(config: Config, exist_ok: bool = True) -> pathlib.Path:
    experiment_name = get_experiment_name(config)
    root = HERE / LOG_DIR / EXPERIMENT / experiment_name
    root.mkdir(parents=True, exist_ok=exist_ok)
    return root


def set_seed(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return g


def make_model(config: Config, device: torch.device) -> BiDimensionalAttentionModel:
    model = BiDimensionalAttentionModel(
        n_layers=config.network.n_layers,
        hidden_dim=config.network.hidden_dim,
        num_heads=config.network.num_heads,
    )
    return model.to(device)


def make_process(config: Config, device: torch.device) -> GaussianDiffusion:
    betas = cosine_schedule(
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        timesteps=config.diffusion.timesteps,
    ).to(device)
    return GaussianDiffusion(betas)


def compute_lr(config: Config, step: int) -> float:
    warmup_steps = config.steps_per_epoch * config.optimizer.num_warmup_epochs
    decay_steps = config.steps_per_epoch * config.optimizer.num_decay_epochs
    init_lr = config.optimizer.init_lr
    peak_lr = config.optimizer.peak_lr
    end_lr = config.optimizer.end_lr

    if step < warmup_steps:
        if warmup_steps == 0:
            return peak_lr
        alpha = step / float(warmup_steps)
        return init_lr + (peak_lr - init_lr) * alpha

    t = min(step - warmup_steps, decay_steps)
    if decay_steps <= 0:
        return end_lr

    cosine = 0.5 * (1.0 + math.cos(math.pi * t / float(decay_steps)))
    return end_lr + (peak_lr - end_lr) * cosine


def get_data(
    dataset: str,
    input_dim: int = 1,
    train: bool = True,
    batch_size: int = 1024,
    num_epochs: int = 1,
):
    task = "training" if train else "interpolation"
    data_path = HERE / "data" / f"{dataset}_{input_dim}_{task}.npz"
    data = np.load(data_path)

    x_target = torch.from_numpy(data["x_target"].astype(np.float32))
    y_target = torch.from_numpy(data["y_target"].astype(np.float32))
    x_context = torch.from_numpy(data["x_context"].astype(np.float32))
    y_context = torch.from_numpy(data["y_context"].astype(np.float32))
    mask_target = torch.from_numpy(data["mask_target"].astype(np.float32))
    mask_context = torch.from_numpy(data["mask_context"].astype(np.float32))

    base_dataset = TensorDataset(
        x_target, y_target, x_context, y_context, mask_target, mask_context
    )

    loader = DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
    )

    def generator():
        for _ in range(num_epochs):
            for xb, yb, xcb, ycb, mtb, mcb in loader:
                yield Batch(
                    x_target=xb,
                    y_target=yb,
                    x_context=xcb,
                    y_context=ycb,
                    mask_target=mtb,
                    mask_context=mcb,
                )

    return generator()


def batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        x_target=torch.as_tensor(batch.x_target, device=device, dtype=torch.float32),
        y_target=torch.as_tensor(batch.y_target, device=device, dtype=torch.float32),
        x_context=torch.as_tensor(batch.x_context, device=device, dtype=torch.float32)
        if batch.x_context is not None
        else None,
        y_context=torch.as_tensor(batch.y_context, device=device, dtype=torch.float32)
        if batch.y_context is not None
        else None,
        mask_target=torch.as_tensor(batch.mask_target, device=device, dtype=torch.float32)
        if batch.mask_target is not None
        else None,
        mask_context=torch.as_tensor(batch.mask_context, device=device, dtype=torch.float32)
        if batch.mask_context is not None
        else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NDP regression model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g., matern, se). Defaults to Config.dataset.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Input dimensionality. Defaults to Config.input_dim.",
    )
    args = parser.parse_args()

    config = Config()
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.input_dim is not None:
        config.input_dim = args.input_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = get_experiment_dir(config)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    wandb.init(
        project="ndp-regression",
        name=str(exp_dir.name),
        config=asdict(config),
        dir=str(exp_dir),
    )

    # Seed global RNGs (PyTorch, NumPy, Python)
    set_seed(config.seed)

    model = make_model(config, device)
    process = make_process(config, device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.optimizer.init_lr,
        weight_decay=config.optimizer.weight_decay,
    )

    wandb.watch(model, log="gradients", log_freq=100)

    ema_model = make_model(config, device)
    ema_model.load_state_dict(model.state_dict())
    ema_decay = config.optimizer.ema_rate

    total_steps = config.total_steps

    ds_train = get_data(
        dataset=config.dataset,
        input_dim=config.input_dim,
        train=True,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
    )

    step = 0
    progress = tqdm(total=total_steps)
    steps_log: list[int] = []
    losses_log: list[float] = []
    lrs_log: list[float] = []
    for batch_raw in ds_train:
        step += 1
        if step > total_steps:
            break

        batch = batch_to_device(batch_raw, device)

        lr = compute_lr(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        loss = diffusion_loss(
            process,
            model,
            batch,
            None,
            num_timesteps=config.diffusion.timesteps,
            loss_type=config.loss_type,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)

        steps_log.append(step)
        losses_log.append(float(loss.item()))
        lrs_log.append(float(lr))
        wandb.log(
            {
                "step": step,
                "train/loss": float(loss.item()),
                "train/lr": float(lr),
            }
        )

        progress.update(1)
        if step % 10 == 0 or step == 1:
            progress.set_description(f"loss={loss.item():.4f}, lr={lr:.2e}")

        if step % (config.steps_per_epoch * max(1, config.num_epochs // 10)) == 0 or step == total_steps:
            ckpt_path = exp_dir / "checkpoints" / f"model_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "config": asdict(config),
                    "model_state": model.state_dict(),
                    "ema_state": ema_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                ckpt_path,
            )

    metrics_path = exp_dir / "training_metrics.npz"
    np.savez(
        metrics_path,
        step=np.array(steps_log, dtype=np.int64),
        loss=np.array(losses_log, dtype=np.float32),
        lr=np.array(lrs_log, dtype=np.float32),
    )

    wandb.finish()


if __name__ == "__main__":
    main()
