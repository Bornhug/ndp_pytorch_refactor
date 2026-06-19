"""TabICL-style training loop for regression NDPs with Gaussian diffusion."""

from __future__ import annotations

import copy
import shutil
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
    denoising_prediction_loss,
    prepare_denoising_targets,
)
from neural_diffusion_processes.regressor import NDPRegressor
from neural_diffusion_processes.types import Batch

from tabicl_style.batching import (
    align_micro_batch,
    pad_nested_batch,
    split_micro_batches,
    validate_micro_batch,
)
from tabicl_style.config import Config
from tabicl_style.data import build_dataloader, build_dataset
from tabicl_style.lora import apply_lora, is_lora_adapter_key, trainable_parameters
from tabicl_style.utils import (
    checkpoint_curr_step,
    compute_lr_from_schedule_steps,
    get_latest_checkpoint,
    infer_latest_run_dir,
    infer_next_checkpoint_dir,
    infer_lr_schedule_steps,
    infer_steps_per_epoch,
    normalize_y,
    prior_start_for_step,
    amp_dtype_name,
    resolve_amp_settings,
    set_seed,
    split_context_target,
)


# ---------------------------------------------------------------------------
# Model + process construction
# ---------------------------------------------------------------------------

def configure_torch_runtime(config: Config) -> None:
    """Apply PyTorch process-wide performance settings.

    This currently controls float32 matrix multiplication precision. It is run
    before model construction so linear layers and attention use the configured
    matmul behavior for the whole training process.
    """
    precision = getattr(config.training, "float32_matmul_precision", None)
    if precision:
        torch.set_float32_matmul_precision(str(precision))


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    """Return the original eager model behind a ``torch.compile`` wrapper.

    ``torch.compile`` wraps the module and stores the original model in
    ``_orig_mod``. Checkpointing, gradient clipping, and EMA should operate on
    the real parameters, so callers use this helper before touching weights.
    """
    return getattr(model, "_orig_mod", model)


def model_state_dict(model: nn.Module):
    """Return checkpoint weights with the same keys for eager/compiled models.

    Without unwrapping, a compiled model would save keys prefixed by the
    wrapper. Saving the eager module state keeps checkpoints loadable regardless
    of whether training used ``torch.compile``.
    """
    return unwrap_compiled_model(model).state_dict()


def maybe_compile_model(model: nn.Module, config: Config) -> nn.Module:
    """Compile the model only when ``training.torch_compile`` is enabled.

    Compilation is optional because it can improve steady-state throughput but
    adds startup cost and may be harder to debug. The returned object should be
    used for forward/backward, while helper functions unwrap it when raw
    parameter access is needed.
    """
    if not bool(getattr(config.training, "torch_compile", False)):
        return model
    mode = str(getattr(config.training, "torch_compile_mode", "default"))
    device = str(getattr(config.training, "device", ""))
    if "cuda" in device:
        missing = [tool for tool in ("nvcc", "ptxas") if shutil.which(tool) is None]
        if missing:
            raise RuntimeError(
                "torch_compile=True on CUDA needs CUDA compiler tools on PATH; "
                f"missing: {', '.join(missing)}. In the WSL ndp_sim environment run: "
                "conda install -y -c nvidia cuda-nvcc=12.9.86. "
                "Set training.torch_compile=False to run without compilation."
            )
    try:
        return torch.compile(model, mode=mode)
    except RuntimeError as exc:
        if sys.version_info >= (3, 12) and "Dynamo is not supported" in str(exc):
            raise RuntimeError(
                "torch_compile=True requires a PyTorch/Python combination with "
                "Dynamo support. Use the WSL ndp_sim Python 3.11 environment, "
                "or set training.torch_compile=False."
            ) from exc
        raise


def build_diffusion_betas(config: Config) -> torch.Tensor:
    """Create the fixed diffusion noise schedule from config.

    The rest of training consumes the resulting beta tensor through
    ``GaussianDiffusion``. Only the ``cosine`` schedule is currently supported,
    so unsupported names fail early instead of silently changing behavior.
    """
    schedule = str(config.diffusion.schedule).strip().lower()
    if schedule == "cosine":
        return cosine_schedule(
            config.diffusion.beta_start,
            config.diffusion.beta_end,
            config.diffusion.timesteps,
        )
    raise ValueError(
        f"Unsupported diffusion schedule {config.diffusion.schedule!r}. "
        "Supported schedules: 'cosine'."
    )


def build_model_and_process(config: Config, device: torch.device):
    """Construct the denoising model and diffusion process for training.

    The model predicts epsilon noise for normalized scalar targets. The process
    owns the fixed beta/alpha schedule used by both training loss and sampling.
    Both objects are moved to the requested training device before returning.
    """
    configure_torch_runtime(config)
    model = NDPRegressor(
        embedding_size=config.model.embedding_size,
        num_attention_heads=config.model.num_attention_heads,
        num_layers=config.model.num_layers,
        num_timesteps=config.diffusion.timesteps,
    ).to(device)
    model = apply_lora(model, config).to(device)

    betas = build_diffusion_betas(config).to(device)
    process = GaussianDiffusion(betas)

    return model, process


# ---------------------------------------------------------------------------
# EMA helpers
# ---------------------------------------------------------------------------

class EMA:
    """Maintain a smoothed copy of model weights for evaluation/checkpoints.

    Training updates the live model every optimizer step. EMA keeps a frozen
    shadow model whose weights change more slowly, which usually gives more
    stable evaluation samples for diffusion-style models.
    """

    def __init__(self, model: nn.Module, decay: float = 0.995):
        """Create the shadow model and disable gradients on it.

        The shadow starts as an exact deep copy. It is never used for backward;
        ``update`` moves it toward the live model after optimizer steps.
        """
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend live weights into the shadow weights after one optimizer step.

        For each parameter, the update is:
        ``shadow = decay * shadow + (1 - decay) * live``.
        """
        source_model = unwrap_compiled_model(model)
        for s_param, m_param in zip(self.shadow.parameters(), source_model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        """Return EMA weights so checkpoints can save the smoothed model."""
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict, *, strict: bool = True):
        """Restore EMA weights when resuming training from a checkpoint."""
        return self.shadow.load_state_dict(state_dict, strict=strict)


def compute_grad_norm(parameters) -> torch.Tensor:
    """Compute the global L2 norm over all available gradients.

    Parameters without gradients are ignored. This is used when gradient
    clipping is disabled so logs still report the true accumulated grad norm.
    """
    if not parameters:
        return torch.tensor(0.0)
    total = torch.zeros((), device=parameters[0].device)
    for p in parameters:
        if p.grad is None:
            continue
        total = total + p.grad.data.norm(2).pow(2)
    return total.sqrt()


def optimizer_step_with_scaler_stats(optimizer, scaler, *, amp: bool) -> Dict[str, float]:
    """Run the optimizer step and report whether GradScaler skipped it.

    ``GradScaler.step`` skips the wrapped optimizer step when it detects
    non-finite fp16 gradients. PyTorch does not expose that as a direct flag, so
    we use the standard signal: the scale decreases after ``update``.
    """
    scaler_enabled = bool(amp and scaler is not None and scaler.is_enabled())
    scale_before = float(scaler.get_scale()) if scaler_enabled else 0.0
    scaler_step_skipped = 0

    if amp:
        scaler.step(optimizer)
        scaler.update()
        if scaler_enabled:
            scale_after = float(scaler.get_scale())
            scaler_step_skipped = int(scale_after < scale_before)
    else:
        optimizer.step()

    return {"scaler_step_skipped": int(scaler_step_skipped)}


def skipped_optimizer_step_stats(scaler, *, amp: bool) -> Dict[str, float]:
    """Return optimizer-step log fields when the trainer skips before stepping."""
    del scaler, amp
    return {"scaler_step_skipped": 0}


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def configure_checkpoint_dir(
    config: Config,
    *,
    total_steps: int,
    runs_root: str | Path | None = None,
) -> Path | None:
    """Choose the run directory and return the checkpoint to load, if any."""
    tc = config.training
    root = Path(runs_root) if runs_root is not None else ROOT / "runs"

    if tc.checkpoint_path:
        checkpoint_path = Path(tc.checkpoint_path).expanduser()
        tc.checkpoint_dir = str(checkpoint_path.resolve().parent)
        print(f"checkpoint_dir inferred from checkpoint_path: {tc.checkpoint_dir}")
        return checkpoint_path

    if tc.checkpoint_dir:
        latest = get_latest_checkpoint(Path(tc.checkpoint_dir))
        if latest:
            print(f"checkpoint_dir explicit; resuming latest checkpoint: {latest}")
        return latest

    if bool(getattr(tc, "auto_resume_latest", True)):
        latest_run = infer_latest_run_dir(root)
        if latest_run is not None:
            latest_checkpoint = get_latest_checkpoint(latest_run)
            if latest_checkpoint is not None:
                latest_step = checkpoint_curr_step(latest_checkpoint)
                if latest_step < total_steps:
                    tc.checkpoint_dir = str(latest_run)
                    print(
                        "checkpoint_dir auto-resumed: "
                        f"{tc.checkpoint_dir} at step {latest_step}"
                    )
                    return latest_checkpoint

    tc.checkpoint_dir = str(infer_next_checkpoint_dir(root))
    Path(tc.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"checkpoint_dir inferred: {tc.checkpoint_dir}")
    return None


def should_save_checkpoint(step: int, total_steps: int, save_every: int) -> bool:
    """Return whether the current global step should write a checkpoint.

    ``save_every <= 0`` disables periodic checkpointing. ``total_steps`` is
    accepted for a stable call shape but is not needed by the current policy.
    """
    del total_steps
    if save_every <= 0:
        return False
    return step % save_every == 0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Stateful training driver for one NDP regression run.

    ``Trainer`` owns everything that changes during training: model weights,
    EMA weights, optimizer state, data iterator position, diffusion RNG, current
    global step, checkpointing, and optional wandb logging.
    """

    def __init__(self, config: Config):
        """Initialize model, optimizer, data, schedules, logging, and checkpoint state.

        The constructor is intentionally heavy: after it returns, ``train`` can
        start immediately. If ``checkpoint_path`` or ``checkpoint_dir`` is set,
        model/optimizer/EMA state is restored before optional compilation.
        """
        self.config = config
        tc = config.training
        self.device = torch.device(tc.device)

        set_seed(tc.np_seed, tc.torch_seed)

        self.curr_step = 0
        self.steps_per_epoch = infer_steps_per_epoch(config)
        self.total_steps = int(self.steps_per_epoch * tc.num_epochs)
        if self.total_steps <= 0:
            raise ValueError("Total training steps must be positive.")
        self.lr_warmup_steps, self.lr_decay_steps = infer_lr_schedule_steps(
            config,
            self.total_steps,
        )
        checkpoint_to_load = configure_checkpoint_dir(
            config,
            total_steps=self.total_steps,
        )

        self.model, self.process = build_model_and_process(config, self.device)
        self.ema = EMA(self.model, decay=config.optimizer.ema_rate)

        params_to_optimize = trainable_parameters(self.model)
        if not params_to_optimize:
            raise ValueError("No trainable parameters available for the optimizer.")
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=config.optimizer.init_lr,
            weight_decay=config.optimizer.weight_decay,
        )

        self.amp, self.amp_dtype = resolve_amp_settings(
            bool(tc.amp),
            tc.dtype,
            device=self.device,
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp and self.amp_dtype in (torch.float16, torch.bfloat16)
        )
        if self.amp:
            self.amp_context = torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
            )
            print(f"Training AMP: enabled (dtype={amp_dtype_name(self.amp_dtype)})")
        else:
            self.amp_context = nullcontext()
            print("Training AMP: not used")

        self.diffusion_key = torch.Generator(device=self.device)
        self.diffusion_key.manual_seed(tc.torch_seed)

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
                    resume=tc.wandb_resume,
                    mode=tc.wandb_mode,
                )

        if checkpoint_to_load is not None:
            if tc.delete_after_load:
                raise RuntimeError(
                    "Cannot safely resume from checkpoint when "
                    "training.delete_after_load=True because prior batch files "
                    "needed for resume may have been deleted."
                )
            self.load_checkpoint(checkpoint_to_load)
        self.model = maybe_compile_model(self.model, config)
        self.reset_data_iter(
            start_from=self.current_prior_start(),
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def current_prior_start(self) -> int:
        """Return the prior file index matching the next training step."""
        return prior_start_for_step(
            self.config.training.load_prior_start,
            self.curr_step,
            self.steps_per_epoch,
        )

    def reset_data_iter(
        self,
        *,
        start_from: int | None = None,
        max_batches: int | None = None,
    ) -> None:
        """Recreate the dataset, DataLoader, and iterator.

        This is called at construction and at every epoch boundary. Rebuilding
        the iterator lets generated or disk-backed prior batches restart at the
        correct file for fresh epochs and checkpoint resumes.
        """
        tc = self.config.training
        dataset = build_dataset(tc, start_from=start_from, max_batches=max_batches)
        self.dataloader = build_dataloader(tc, dataset)
        self.data_iter = iter(self.dataloader)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def get_latest_checkpoint(self, ckpt_dir: Path) -> Path | None:
        """Find the latest checkpoint file in a checkpoint directory.

        Returns ``None`` when the directory does not exist or contains no
        ``step-*.pt`` files. Numeric step parsing is used so ``step-100.pt``
        correctly sorts after ``step-20.pt``.
        """
        return get_latest_checkpoint(ckpt_dir)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore training state from a checkpoint.

        New-format checkpoints contain model weights, EMA weights, optimizer
        state, and ``curr_step``. A plain state dict is also accepted for older
        checkpoints, but that path restores only model weights.
        """
        ckpt = torch.load(path, map_location=self.device)
        if "state_dict" in ckpt:
            strict = not bool(getattr(getattr(self.config, "lora", None), "enabled", False))
            load_result = self.model.load_state_dict(ckpt["state_dict"], strict=strict)
            loaded_base_into_lora = False
            if not strict:
                unexpected = list(load_result.unexpected_keys)
                missing = list(load_result.missing_keys)
                loaded_base_into_lora = bool(missing)
                if unexpected or any(not is_lora_adapter_key(key) for key in missing):
                    raise RuntimeError(
                        "Base checkpoint did not load cleanly into LoRA model: "
                        f"missing={missing}, unexpected={unexpected}"
                    )
            if "ema_state_dict" in ckpt:
                if strict:
                    self.ema.load_state_dict(ckpt["ema_state_dict"])
                else:
                    load_result = self.ema.load_state_dict(
                        ckpt["ema_state_dict"],
                        strict=False,
                    )
                    unexpected = list(load_result.unexpected_keys)
                    missing = list(load_result.missing_keys)
                    if unexpected or any(not is_lora_adapter_key(key) for key in missing):
                        raise RuntimeError(
                            "EMA checkpoint did not load cleanly into LoRA model: "
                            f"missing={missing}, unexpected={unexpected}"
                        )
            if "optimizer_state" in ckpt and not loaded_base_into_lora:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            if not loaded_base_into_lora:
                self.curr_step = int(ckpt.get("curr_step", 0))
        else:
            self.model.load_state_dict(ckpt)
        print(f"Loaded checkpoint from {path}")

    def save_checkpoint(self, step: int) -> None:
        """Write model, EMA, optimizer, step, and config to disk.

        The model state is unwrapped before saving so checkpoints remain
        compatible whether the live model is eager or compiled.
        """
        tc = self.config.training
        if not tc.checkpoint_dir:
            return
        ckpt_dir = Path(tc.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"step-{step}.pt"
        payload = {
            "state_dict": model_state_dict(self.model),
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
        """Reject NaN/Inf tensors before they can corrupt an optimizer update."""
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"non-finite {name}")

    def run_micro_batch(
        self, micro_batch, micro_batch_idx: int, num_micro_batches: int
    ) -> Dict[str, float]:
        """Run one gradient-accumulation slice of a training batch.

        A DataLoader batch can contain many independent tabular tasks. To keep
        memory bounded, ``run_batch`` splits it into smaller micro-batches and
        calls this method for each slice.

        This method does four things:
        1. moves one slice to the training device and splits it into
           context/target points;
        2. normalizes y values using context statistics and filters unstable
           tasks;
        3. computes the diffusion denoising loss on the remaining tasks;
        4. calls backward on a scaled loss so gradients accumulate correctly.

        It does not call ``optimizer.step``. It returns loss/filtering stats so
        ``run_batch`` can decide whether the full update is safe to apply.
        """
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

        # Normalize both context and target y with context statistics. The
        # diffusion model trains on normalized y, while evaluation later
        # denormalizes predictions back to the original target scale.
        y_context_norm, y_target_norm, _mean, _std = normalize_y(y_context, y_target)

        y_context_stats = y_context.unsqueeze(-1) if y_context.ndim == 2 else y_context
        raw_std = y_context_stats.std(dim=1, keepdim=True, unbiased=False)  # [B,1,1]

        # Drop pathological tasks before backward. Tiny context variance makes
        # z-score normalization unstable, and extreme normalized targets can
        # dominate a whole update.
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

        # Clamp remaining normalized values to bound the loss scale without
        # dropping otherwise usable tasks.
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

        batch = Batch(
            x_target=x_target,
            y_target=y_target_norm,
            x_context=x_context,
            y_context=y_context_norm,
        )

        t, yt, noise_true = prepare_denoising_targets(
            self.process,
            batch,
            self.diffusion_key,
            num_timesteps=self.config.diffusion.timesteps,
        )

        with self.amp_context:
            noise_hat = self.model(
                x_target=batch.x_target,
                y_target=yt,
                t=t.to(batch.x_target.device),
                x_context=batch.x_context,
                y_context=batch.y_context,
            )

        loss = denoising_prediction_loss(
            noise_true,
            noise_hat,
            loss_type=self.config.training.loss_type,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")

        # Divide by the number of micro-batches so accumulated gradients match
        # the scale of a single full-batch backward pass.
        scaled_loss = loss / float(num_micro_batches)
        if self.amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return {"loss": float(loss.item()), "skipped_micro": False, **stats}

    def run_batch(
        self,
        batch,
        *,
        step: int,
        lr_step: int | None = None,
    ) -> Dict[str, float]:
        """Run one global training step and maybe update model weights.

        The incoming batch is first padded and split into micro-batches. Each
        valid micro-batch backpropagates into the same parameter gradients.
        After all slices are processed, this method performs the operations
        that should happen exactly once per global step:

        1. discard accumulated gradients if a numerical failure was detected;
        2. unscale AMP gradients when needed;
        3. clip or measure the accumulated gradient norm;
        4. set the learning rate for this step;
        5. step the optimizer and update EMA weights.

        The returned dictionary is the single source for progress-bar and wandb
        training metrics.
        """
        lr_step_for_schedule = step if lr_step is None else int(lr_step)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # TabICL batches can contain nested tensors with different shapes. Pad
        # once at the full-batch level, then split by task count for memory.
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

        # Each successful micro-batch contributes gradients. Skipped
        # micro-batches contribute stats only, so a few bad tasks do not waste
        # the whole DataLoader batch.
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

        # A numerical failure means some gradients may already be contaminated,
        # so discard the whole accumulated update.
        if skip_update:
            self.optimizer.zero_grad(set_to_none=True)
            lr = compute_lr_from_schedule_steps(
                self.config,
                lr_step_for_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
            )
            return {
                "loss": float("nan"),
                "skipped": True,
                "skip_reason": skip_reason,
                "lr": float(lr),
                "step": step,
                "lr_step": lr_step_for_schedule,
                "total_tasks": int(results["total_tasks"]),
                "valid_tasks": int(results["valid_tasks"]),
                "skipped_tiny_std_tasks": int(results["skipped_tiny_std_tasks"]),
                "skipped_extreme_tasks": int(results["skipped_extreme_tasks"]),
                "skipped_nonfinite_tasks": int(results["skipped_nonfinite_tasks"]),
                "clipped_target_values": int(results["clipped_target_values"]),
                "skipped_micro_batches": int(results["skipped_micro_batches"]),
                **skipped_optimizer_step_stats(self.scaler, amp=self.amp),
            }

        # If every micro-batch was filtered or failed safely, leave weights
        # unchanged and report the skip instead of stepping on empty gradients.
        if processed == 0:
            self.optimizer.zero_grad(set_to_none=True)
            lr = compute_lr_from_schedule_steps(
                self.config,
                lr_step_for_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
            )
            return {
                "loss": float("nan"),
                "skipped": True,
                "skip_reason": "no valid micro-batches",
                "lr": float(lr),
                "step": step,
                "lr_step": lr_step_for_schedule,
                "total_tasks": int(results["total_tasks"]),
                "valid_tasks": int(results["valid_tasks"]),
                "skipped_tiny_std_tasks": int(results["skipped_tiny_std_tasks"]),
                "skipped_extreme_tasks": int(results["skipped_extreme_tasks"]),
                "skipped_nonfinite_tasks": int(results["skipped_nonfinite_tasks"]),
                "clipped_target_values": int(results["clipped_target_values"]),
                "skipped_micro_batches": int(results["skipped_micro_batches"]),
                **skipped_optimizer_step_stats(self.scaler, amp=self.amp),
            }

        # AMP gradients must be unscaled before clipping; otherwise the clip
        # threshold would be applied to scaled values rather than true grads.
        if self.amp:
            self.scaler.unscale_(self.optimizer)
        trainable_model = unwrap_compiled_model(self.model)
        if self.config.training.gradient_clipping > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                trainable_model.parameters(),
                self.config.training.gradient_clipping,
            )
        else:
            grad_norm = compute_grad_norm(list(trainable_model.parameters()))

        # The LR is computed once per optimizer update, not per micro-batch.
        lr = compute_lr_from_schedule_steps(
            self.config,
            lr_step_for_schedule,
            warmup_steps=self.lr_warmup_steps,
            decay_steps=self.lr_decay_steps,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        # Optimizer step. With fp16 AMP, GradScaler may skip the wrapped
        # optimizer update when it detects non-finite gradients.
        results.update(
            optimizer_step_with_scaler_stats(
                self.optimizer,
                self.scaler,
                amp=self.amp,
            )
        )

        # EMA tracks the post-update weights and is saved/evaluated separately.
        self.ema.update(self.model)

        if processed > 0:
            results["loss"] /= processed
        results["skipped"] = False
        results["grad_norm"] = float(grad_norm)
        results["lr"] = float(lr)
        results["step"] = step
        results["lr_step"] = lr_step_for_schedule
        return results

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run epochs until ``total_steps`` is reached.

        Each loop iteration pulls one DataLoader batch, delegates the actual
        training step to ``run_batch``, updates progress output, writes periodic
        checkpoints, and mirrors the same metrics to wandb when enabled.
        """
        tc = self.config.training
        progress = tqdm(
            total=self.total_steps,
            initial=self.curr_step,
            desc="training",
        )
        start_epoch = self.curr_step // self.steps_per_epoch
        resume_offset = self.curr_step % self.steps_per_epoch

        for epoch in range(start_epoch, tc.num_epochs):
            start_step_idx = resume_offset if epoch == start_epoch else 0
            self.reset_data_iter(
                start_from=int(tc.load_prior_start) + int(start_step_idx)
            )
            for step_idx in range(start_step_idx, self.steps_per_epoch):
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
                    log_dict = {
                        "step": step,
                        "train/skipped": int(bool(results.get("skipped", False))),
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
                        "train/scaler_step_skipped": int(
                            results.get("scaler_step_skipped", 0)
                        ),
                    }
                    if not results.get("skipped"):
                        log_dict.update(
                            {
                                "train/loss": float(results["loss"]),
                                "train/grad_norm": float(
                                    results.get("grad_norm", 0.0)
                                ),
                            }
                        )
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

        if tc.save_every > 0 and tc.checkpoint_dir and self.curr_step > 0:
            final_checkpoint = Path(tc.checkpoint_dir) / f"step-{self.curr_step}.pt"
            if not final_checkpoint.exists():
                self.save_checkpoint(self.curr_step)


def main() -> None:
    """Entry point used when running this file as a script.

    Training currently starts from the dataclass defaults in ``Config``. For
    custom settings, construct ``Config`` in Python and pass it to ``Trainer``.
    """
    config = Config()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
