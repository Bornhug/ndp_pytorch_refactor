# nanoTabPFN_version10_original02

Regression-oriented nanoTabPFN based on `nanoTabPFN_version10_original`.

Main differences from the original classification setup:
- Trains on TabICL pre-generated prior files (`batch_*.pt`) instead of H5.
- Decoder outputs TabPFN-style regression bar logits: `[B, num_targets, 5000]`.
- Uses `FullSupportBarDistribution` negative log-density over normalized targets.
- Clamps normalized context/target y values to `[-z_max, z_max]` before the training loss.
- Uses context-based y normalization during training and denormalization at inference.
- Training skips non-finite prior files and continues.
- Training logs to Weights & Biases (if installed) and saves checkpoints every 300 updates.
- Training does not run OpenML evaluation; use the standalone evaluation scripts after checkpoints are saved.

## Files
- `model.py`: architecture and sklearn-like `NanoTabPFNRegressor` interface.
- `bar_distribution.py`: local FullSupport bar distribution and border loading.
- `scripts/extract_tabpfn_v2_borders.py`: one-time helper to cache official TabPFN v2 borders.
- `train.py`: training loop + TabICL pre-generated prior loader + W&B + checkpointing.
- `evaluation.py`: standalone OpenML regression CV evaluator with R2/RMSE/MAE/QICE/sharpness JSON.
- `tabarena_eval/evaluation.py`: OpenML regression CV evaluation backend.
- `tabarena_eval/run_sweep.py`: checkpoint sweep for regression metrics.

## Border Asset

Training and evaluation require the cached TabPFN v2 regressor borders:

```bash
python nanoTabPFN_version10_original02/scripts/extract_tabpfn_v2_borders.py \
  --version v2 \
  --device cpu \
  --num-bars 5000 \
  --output nanoTabPFN_version10_original02/assets/tabpfn_v2_regressor_borders_5000.pt
```

## Train

```bash
python nanoTabPFN_version10_original02/train.py
```

By default, training loads prior data from:
`/mnt/c/Users/apex/Code/Python/tabicl/data/stage1_continuous_y_seed123`

Useful options:

```bash
python nanoTabPFN_version10_original02/train.py \
  --prior-dir /mnt/c/Users/apex/Code/Python/tabicl/data/stage1_continuous_y_seed123 \
  --num-layers 5 \
  --num-bars 5000 \
  --z-max 5 \
  --checkpoint-interval 300 \
  --wandb-project nanoTabPFN_version10_original02
```

Disable W&B:

```bash
python nanoTabPFN_version10_original02/train.py --disable-wandb
```

## Standalone evaluation

```bash
python nanoTabPFN_version10_original02/evaluation.py \
  --checkpoint <checkpoint.pt> \
  --n-splits 5 --random-state 0 \
  --num-bins 10 --sharpness-coverage 0.90 \
  --output-json <evaluation.json>
```

## Sweep checkpoints

```bash
python nanoTabPFN_version10_original02/tabarena_eval/run_sweep.py --checkpoints-dir <dir> --plot-path auto --plot-metric R2
```
