# Training Config Field Audit

This note tracks config fields that currently do not affect
`tabicl_style/train.py` training behavior.

`tabicl_style/train.py` now starts from `Config()` directly. Config defaults
live in `tabicl_style/config.py`; runtime overrides should be supplied by
constructing `Config` in Python and passing it to `Trainer`.

| Field | Why unused |
|---|---|
| `TrainingConfig.eval_every` | Defined in `tabicl_style/config.py`, but not used by the training loop. |
| `TrainingConfig.eval_batches` | Defined in `tabicl_style/config.py`, but not used by the training loop. |
| `TrainingConfig.eval_sampling_steps` | Defined in `tabicl_style/config.py`, but not used by the training loop. |
| `TrainingConfig.eval_sampling_method` | Defined in `tabicl_style/config.py`, but not used by the training loop. |

## No Longer Unused

| Field | Current status |
|---|---|
| `ModelConfig.output_dim` | Removed from `tabicl_style/config.py`; regression output remains scalar in the model/process code. |
| `DiffusionConfig.schedule` | Now read by `build_diffusion_betas()` in `tabicl_style/train.py`. Only `"cosine"` is supported right now. |
| `OptimizerConfig.num_warmup_epochs` | Removed from `tabicl_style/config.py`; replaced by `OptimizerConfig.warmup_fraction`. |
| `OptimizerConfig.warmup_steps` | Removed from `tabicl_style/config.py`; replaced by `OptimizerConfig.warmup_fraction`. |
| `OptimizerConfig.warmup_fraction` | Now read by `compute_lr()` in `tabicl_style/utils.py`. Default is `0.10`, meaning 10% of total training epochs/steps. |
| `OptimizerConfig.num_decay_epochs` | Removed from `tabicl_style/config.py`; replaced by `OptimizerConfig.decay_fraction`. |
| `OptimizerConfig.decay_fraction` | Now read by `compute_lr()` in `tabicl_style/utils.py`. Default is `0.80`, meaning 80% of total training epochs/steps. |
| `TrainingConfig.num_workers` | Now read by `build_dataloader()` in `tabicl_style/data.py` and passed to PyTorch `DataLoader`. |
