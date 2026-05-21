# TabICL_regression03 Commands

Run these from the repository root:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ndp_sim
cd /mnt/c/Users/apex/Code/Python/ndp_pytorch_refactor
```

## Environment Setup

Create the WSL conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n ndp_sim python=3.11 -y
conda activate ndp_sim
cd /mnt/c/Users/apex/Code/Python/ndp_pytorch_refactor
```

Install CUDA compiler tools for `torch.compile`:

```bash
conda install -y -c nvidia cuda-nvcc=12.9.86
```

Install Python dependencies. This uses the Tsinghua PyPI mirror for normal
packages; `requirements.txt` also adds the official PyTorch CUDA 12.9 wheel
index for `torch==2.8.0+cu129`.

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  -r TabICL_regression03/requirements.txt
```

Verify CUDA and PyTorch:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Verify `torch.compile`:

```bash
python - <<'PY'
import torch

torch.set_float32_matmul_precision("high")
model = torch.compile(torch.nn.Linear(4, 4).cuda(), mode="default")
y = model(torch.randn(2, 4, device="cuda"))
torch.cuda.synchronize()
print(torch.__version__, torch.version.cuda, tuple(y.shape))
PY
```

## Train

`train.py` uses the default dataclass settings from
`TabICL_regression03/tabicl_style/config.py`.

```bash
python TabICL_regression03/tabicl_style/train.py
```

If `checkpoint_dir` is unset in `config.py`, `Trainer` first checks the latest
`TabICL_regression03/runs/runXX` directory. If it contains an unfinished
`step-*.pt`, training resumes from that checkpoint and starts from the next
needed prior batch file. Otherwise, it creates the next `runXX` directory.

To force a fresh run instead of auto-resuming, set this in `config.py`:

```python
auto_resume_latest = False
```

If `checkpoint_dir` is explicitly set, `Trainer` auto-loads the latest
`step-*.pt` from that directory.

## Evaluate One Checkpoint With DDPM

```bash
python TabICL_regression03/tabicl_style/evaluation.py \
  --checkpoint TabICL_regression03/runs/run01/step-50000.pt \
  --device cuda \
  --amp \
  --amp-dtype auto \
  --num-sampling-steps 500 \
  --sampling-method ddpm \
  --max-features-eval 32 \
  --max-rows-eval 1000 \
  --new-instances-eval 0 \
  --n-splits 10 \
  --n-repeats 20 \
  --random-state 0 \
  --output-json TabICL_regression03/runs/run01/step-50000_ddpm500.json
```

To evaluate only selected datasets:

```bash
python TabICL_regression03/tabicl_style/evaluation.py \
  --checkpoint TabICL_regression03/runs/run01/step-50000.pt \
  --device cuda \
  --amp \
  --amp-dtype auto \
  --num-sampling-steps 500 \
  --sampling-method ddpm \
  --datasets abalone,boston \
  --output-json TabICL_regression03/runs/run01/step-50000_ddpm500_subset.json
```

## Evaluate One Checkpoint With DDIM

```bash
python TabICL_regression03/tabicl_style/evaluation.py \
  --checkpoint TabICL_regression03/runs/run_1000t_20260520_122626/step-30000.pt \
  --device cuda \
  --amp \
  --amp-dtype auto \
  --num-sampling-steps 1000 \
  --sampling-method ddpm \
  --ddim-eta 0.0 \
  --max-features-eval 32 \
  --max-rows-eval 0 \
  --new-instances-eval 200 \
  --n-splits 5 \
  --n-repeats 20 \
  --random-state 0 \
  --output-json TabICL_regression03/runs/run_1000t_20260520_122626/step-30000_ddpm1000.json
```

To force full precision evaluation even when a checkpoint config used AMP, add:

```bash
--no-amp
```

## Evaluate Saved Prediction Uncertainty With QICE

Run this after `evaluation.py --output-json`; it reuses the saved repeated
predictions and does not reload the checkpoint:

```bash
python TabICL_regression03/tabicl_style/evaluation_uncertainty.py \
  --input-json TabICL_regression03/runs/run_1000t_20260520_122626/step-30000_ddpm1000.json \
  --num-bins 10 \
  --output-json TabICL_regression03/runs/run_1000t_20260520_122626/step-30000_ddpm1000_uncertainty.json
```

## Sweep Saved Checkpoints

```bash
python TabICL_regression03/tabicl_style/run_checkpoint_sweep.py \
  --checkpoints-dir TabICL_regression03/runs/run01 \
  --device cuda \
  --num-sampling-steps 500 \
  --sampling-method ddpm \
  --max-features-eval 32 \
  --max-rows-eval 1000 \
  --new-instances-eval 0 \
  --n-splits 10 \
  --random-state 0 \
  --output-json TabICL_regression03/runs/run01/checkpoint_sweep_ddpm500.json \
  --plot-dir TabICL_regression03/runs/run01/evaluation_plots
```

## Sweep DDIM Sampling Steps

```bash
python TabICL_regression03/tabicl_style/run_ddim_eval_sweep.py \
  --checkpoint TabICL_regression03/runs/run01/step-50000.pt \
  --output-json TabICL_regression03/runs/run01/step-50000_ddim_sweep.json \
  --device cuda \
  --sampling-steps 250,125,50,10,5 \
  --ddim-eta 0.0 \
  --max-features-eval 32 \
  --max-rows-eval 1000 \
  --new-instances-eval 0 \
  --n-splits 10 \
  --n-repeats 20 \
  --random-state 0
```

## CPU Smoke Checks

```bash
python -m pytest TabICL_regression03/tests -q
python TabICL_regression03/tabicl_style/evaluation.py --help
```
