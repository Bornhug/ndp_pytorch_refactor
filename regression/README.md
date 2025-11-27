# 1. Introduction
This experiment is to train a NDP model on Synthetic GP regression dataset. Please follow the following instructions to 
complete the whole experiment.

# 2. Data

From the repo root (`ndp_pytorch_refactor`), generate the datasets:

```
python regression/generate_data.py
```

Results:
- NPZ files saved to `regression/data/` with pattern `<dataset>_<input_dim>_<task>.npz` (e.g., `matern_1_training.npz`, `matern_1_interpolation.npz`).
- A histogram figure of context/target counts is saved as `regression/num_data.png` (since plotting is enabled by default).

## 2.1 Visualization
From repo root you can inspect generated `.npz` files:

- Convert to Excel (defaults to saving in `regression/visualization/`):
  ```
  python regression/visualization/npz_to_excel.py regression/data/matern_1_training.npz --out regression/visualization/matern_1_training.xlsx
  ```
- Plot batches (saves under `regression/visualization/`):
  ```
  python regression/visualization/plot_npz_samples.py regression/data/matern_1_interpolation.npz --max-batches 16 --out regression/visualization/matern_1_interpolation_plot.png
  ```


# 3. Training

Run the PyTorch training loop (writes to `regression/logs/regression-pytorch/...`). Choose dataset/input dimension via flags; defaults are in `config.py` (dataset defaults to `matern`, input_dim to `1`):

```
python regression/train.py --dataset matern --input-dim 1
```

You may sign up a free account at [wandb.ai](https://wandb.ai) and set your API key in `~/.netrc` to log the run:

Here we also provide a pre-trained checkpoint:
`regression\logs\regression-pytorch\Nov27_114624_cspc\checkpoints\model_step_128000.pt`

# 4. Interpolation / Evaluation

Evaluate a trained checkpoint. Replace `<run>` and the checkpoint filename as needed:

```
python regression/interpolation.py --ckpt regression/logs/regression-pytorch/<run>/checkpoints/model_step_XXXXXX.pt --dataset matern --input-dim 1 --batch-size 4 --num-samples 8 --seed 42 --npz-path regression/data/matern_1_interpolation.npz
```

For example, use the provided pre-trained checkpoint:

```
python regression/interpolation.py --ckpt regression/logs/regression-pytorch/Nov27_114624_cspc/checkpoints/model_step_128000.pt --dataset matern --input-dim 1 --batch-size 4 --num-samples 8 --seed 42 --npz-path regression/data/matern_1_interpolation.npz
```

The script writes evaluation outputs next to the run directory you point `--ckpt` to:
- Metrics JSON: `regression/logs/regression-pytorch/<run>/interpolation_metrics_<dataset>_<input_dim>.json`
- Comparison plot: `regression/logs/regression-pytorch/<run>/interpolation_plot_<dataset>_<input_dim>.png`

## 4.1 Metrics JSON
Example (`matern`, 1D, 8 samples) from `regression/logs/regression-pytorch/Nov27_114624_cspc/interpolation_metrics_matern_1.json`:
- Baseline GP log-likelihood mean ≈ -2.01 (high variance; some outliers strongly negative).
- Method 1 (NDP-based) log-likelihood mean ≈ 0.45 with low variance (std ≈ 0.12), indicating the NDP sampler improves log-likelihood over the baseline GP on this split.
- Method 2 results are extremely negative (The number of samples (8) is extremely less than the number of evaluated target points (50), causing rank inefficient problem); prefer Method 1.

## 4.2 Comparison plot
Example plot (`regression/logs/regression-pytorch/Nov27_114624_cspc/interpolation_plot_matern_1.png`):
- Blue: NDP mean with 95% CI from sampled trajectories.
- Red: Baseline GP mean with 95% CI.
- Black dots: observed context points.
- In this example batch, the NDP mean tracks the GP closely while offering calibrated uncertainty around the context-driven fit.
