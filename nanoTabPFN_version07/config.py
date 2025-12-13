from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Hyperparameters for the NanoTabPFN-style diffusion model.
    """

    # Dimensionalities inspired by the nanoTabPFN README example.
    embedding_size: int = 96
    num_attention_heads: int = 8
    num_layers: int = 5

    # Dataset-specific structure for this prior dump.
    num_features: int = 5
    num_outputs: int = 2  # binary classification (two classes)


@dataclass
class DiffusionConfig:
    schedule: str = "cosine"
    beta_start: float = 3e-4
    beta_end: float = 0.5
    timesteps: int = 500
    transition_mat_type: str = "uniform"  # "uniform" or "absorbing"
    use_absorbing: bool = False
    transition_mat_type: str = "uniform"


@dataclass
class OptimizerConfig:
    num_warmup_epochs: int = 1
    num_decay_epochs: int = 80
    init_lr: float = 2e-5
    peak_lr: float = 3e-4
    end_lr: float = 1e-5
    weight_decay: float = 0.0


@dataclass
class TrainingConfig:
    """
    Configuration for pretraining on the nanoTabPFN prior dump.
    """

    # Relative path to the prior data dump (300k tasks).
    h5_path: str = "300k_150x5_2.h5"

    # Meta-training schedule
    batch_size: int = 32
    num_epochs: int = 100
    # Number of meta-tasks per epoch (matches regression defaults).
    samples_per_epoch: int = int(2**14)

    # Fixed context ratio per task (fraction of points used as context).
    p_ctx: float = 0.25

    seed: int = 0
    
    # Evaluation settings
    eval_every: int = 500  # Run evaluation every N steps (0 = disabled)
    eval_sampling_method: str = "ddim"  # "ddpm" or "ddim"
    eval_sampling_steps: int = 50  # Number of steps for DDIM sampling

    def resolved_h5_path(self) -> Path:
        """
        Resolve the H5 path relative to this file's directory.
        """
        here = Path(__file__).resolve().parent
        return here / self.h5_path

    @property
    def steps_per_epoch(self) -> int:
        return self.samples_per_epoch // self.batch_size

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.num_epochs


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
