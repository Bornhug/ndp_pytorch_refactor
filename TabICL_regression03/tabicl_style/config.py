"""Configuration dataclasses for TabICL_regression03 NDP training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    embedding_size: int = 128
    num_attention_heads: int = 8
    num_layers: int = 12


@dataclass
class DiffusionConfig:
    schedule: str = "cosine"
    beta_start: float = 3e-4
    beta_end: float = 0.5
    timesteps: int = 500


@dataclass
class OptimizerConfig:
    warmup_fraction: float = 0.02
    decay_fraction: float = 0.80
    lr_schedule: str = "cosine"
    polynomial_power: float = 1.0
    init_lr: float = 2e-5
    peak_lr: float = 2e-4
    end_lr: float = 1e-5
    ema_rate: float = 0.995
    weight_decay: float = 0.0


@dataclass
class LoraConfig:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    train_layer_norm: bool = True
    train_output_head: bool = True


@dataclass
class TrainingConfig:
    # Batch sizing and training length.
    batch_size: int = 32
    micro_batch_size: int = 4
    num_epochs: int = 10

    # Loss choice and target-normalization safety filters.
    loss_type: str = "l1"
    # Skip tasks whose context y standard deviation is too small for stable z-scoring.
    std_min: float = 0.05
    # Clamp normalized context/target y values to this absolute value after filtering.
    z_max: float = 5.0
    # Mark normalized target values beyond this absolute value as hard outliers.
    hard_z_threshold: float = 5.0
    # Skip a task if more than this fraction of target points are hard outliers.
    max_hard_z_frac: float = 0.01
    # Clip global gradient norm to this value; set <= 0 to disable clipping.
    gradient_clipping: float = 1.0

    # Device, precision, and PyTorch runtime behavior.
    # PyTorch device used for model/loss computation; WSL ndp normally uses "cuda".
    device: str = "cuda"
    # Autocast dtype used only when amp=True; "auto" prefers bf16 on supported CUDA GPUs.
    dtype: str = "bf16"
    # Enable CUDA mixed precision to reduce memory use and usually speed training.
    amp: bool = False
    # Optional torch.compile graph optimization. Use Python 3.11 + torch 2.8/cu129
    # with CUDA compiler tools installed; Python 3.12 + torch 2.3 cannot compile.
    torch_compile: bool = False
    # Compilation mode passed to torch.compile when torch_compile=True. "default"
    # avoids CUDA graph capture issues from repeated gradient-accumulation forwards;
    # "reduce-overhead" can be faster only for stricter static-shape loops.
    torch_compile_mode: str = "default"
    # Process-wide float32 matmul precision policy: "highest", "high", or "medium".
    float32_matmul_precision: str = "high"

    # Reproducibility and DataLoader/runtime progress settings.
    np_seed: int = 42
    torch_seed: int = 42
    # DataLoader worker count; keep conservative because LoadPriorDataset is iterable.
    num_workers: int = 2
    # Show a nested progress bar for micro-batches; useful for debugging but noisy.
    micro_progress: bool = False

    # External TabICL prior source and saved-prior loading controls.
    tabicl_repo: str | None = "/mnt/c/Users/apex/Code/Python/tabicl"
    prior_dir: str | None = "/mnt/c/Users/apex/Code/Python/tabicl/data_regression/stage1"
    load_prior_start: int = 0
    delete_after_load: bool = False
    prior_device: str = "cpu"

    # Checkpoint resume/save paths and save cadence. If checkpoint_dir is None,
    # train.py automatically uses the next TabICL_regression03/runs/runXX folder.
    checkpoint_dir: str | None = None
    checkpoint_path: str | None = None
    auto_resume_latest: bool = True
    save_every: int = 3000

    # Weights & Biases logging configuration.
    wandb_log: bool = True
    wandb_project: str = "TabICL-regression03"
    wandb_name: str | None = "preln12-emb128-stage1-gdrive"
    wandb_id: str | None = None
    wandb_dir: str | None = None
    wandb_mode: str = "online"
    wandb_resume: str = "allow"

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
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
