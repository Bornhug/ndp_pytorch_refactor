"""Configuration dataclasses for TabICL_regression03 NDP training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    embedding_size: int = 96
    num_attention_heads: int = 8
    num_layers: int = 5


@dataclass
class DiffusionConfig:
    schedule: str = "cosine"
    beta_start: float = 3e-4
    beta_end: float = 0.5
    timesteps: int = 500


@dataclass
class OptimizerConfig:
    warmup_fraction: float = 0.10
    decay_fraction: float = 0.80
    init_lr: float = 2e-4
    peak_lr: float = 1e-2
    end_lr: float = 1e-4
    ema_rate: float = 0.995
    weight_decay: float = 0.0


@dataclass
class TrainingConfig:
    # Batch sizing and training length.
    batch_size: int = 32
    micro_batch_size: int = 8
    num_epochs: int = 100

    # Loss choice and target-normalization safety filters.
    loss_type: str = "l1"
    std_min: float = 1e-3
    z_max: float = 10.0
    hard_z_threshold: float = 30.0
    max_hard_z_frac: float = 0.05
    gradient_clipping: float = 1.0

    # Device, precision, and PyTorch runtime behavior.
    device: str = "cuda"
    dtype: str = "float32"
    amp: bool = False
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    float32_matmul_precision: str = "high"

    # Reproducibility and DataLoader/runtime progress settings.
    np_seed: int = 42
    torch_seed: int = 42
    num_workers: int = 1
    micro_progress: bool = False

    # External TabICL prior source and saved-prior loading controls.
    tabicl_repo: str | None = None
    prior_dir: str | None = None
    load_prior_start: int = 0
    delete_after_load: bool = False
    prior_device: str = "cpu"

    # Checkpoint resume/save paths and save cadence.
    checkpoint_dir: str | None = None
    checkpoint_path: str | None = None
    save_every: int = 100000

    # Weights & Biases logging configuration.
    wandb_log: bool = True
    wandb_project: str = "TabICL-regression03"
    wandb_name: str | None = None
    wandb_id: str | None = None
    wandb_dir: str | None = None
    wandb_mode: str = "offline"

    # Reserved online-evaluation settings; not used by the current train loop.
    eval_every: int = 0
    eval_batches: int = 10
    eval_sampling_steps: int = 50
    eval_sampling_method: str = "ddpm"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
