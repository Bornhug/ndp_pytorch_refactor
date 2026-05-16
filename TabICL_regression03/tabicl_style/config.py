"""Configuration dataclasses for TabICL_regression03 NDP training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    embedding_size: int = 96
    num_attention_heads: int = 8
    num_layers: int = 5
    output_dim: int = 1  # continuous scalar


@dataclass
class DiffusionConfig:
    schedule: str = "cosine"
    beta_start: float = 3e-4
    beta_end: float = 0.5
    timesteps: int = 500


@dataclass
class OptimizerConfig:
    num_warmup_epochs: int = 1
    warmup_steps: int = 0  # unused; warmup is auto 10% of total steps
    num_decay_epochs: int = 80
    init_lr: float = 2e-4
    peak_lr: float = 1e-2
    end_lr: float = 1e-4
    ema_rate: float = 0.995
    weight_decay: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 32
    micro_batch_size: int = 8
    num_epochs: int = 100
    samples_per_epoch: int = 2**14
    loss_type: str = "l1"
    std_min: float = 1e-3
    z_max: float = 10.0
    hard_z_threshold: float = 30.0
    max_hard_z_frac: float = 0.05
    gradient_clipping: float = 1.0
    device: str = "cuda"
    dtype: str = "float32"
    amp: bool = False
    np_seed: int = 42
    torch_seed: int = 42
    num_workers: int = 1
    micro_progress: bool = False
    # data
    tabicl_repo: str | None = None
    prior_dir: str | None = None
    load_prior_start: int = 0
    delete_after_load: bool = False
    batch_size_per_gp: int = 4
    min_features: int = 2
    max_features: int = 100
    min_seq_len: int | None = None
    max_seq_len: int = 1024
    log_seq_len: bool = False
    seq_len_per_gp: bool = False
    min_train_size: float = 0.1
    max_train_size: float = 0.9
    replay_small: bool = False
    prior_type: str = "mix_scm"
    prior_device: str = "cpu"
    # checkpointing
    checkpoint_dir: str | None = None
    checkpoint_path: str | None = None
    save_every: int = 10000
    # wandb
    wandb_log: bool = True
    wandb_project: str = "TabICL-regression03"
    wandb_name: str | None = None
    wandb_id: str | None = None
    wandb_dir: str | None = None
    wandb_mode: str = "offline"
    # evaluation
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
