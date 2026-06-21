#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/apex/Code/Python/ndp_pytorch_refactor

RUN_DIR="nanoTabPFN_version10_original02/checkpoints_concurrent_seed123_l5_bars5000"
SESSION="nanotabpfn_v10_l5_bars5000_concurrent"

mkdir -p "$RUN_DIR"

if find "$RUN_DIR" -maxdepth 1 -name 'model_step_*.pt' | grep -q .; then
  echo "run dir already has checkpoints: $RUN_DIR"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd /mnt/c/Users/apex/Code/Python/ndp_pytorch_refactor && \
   /home/apex/miniconda3/bin/conda run --no-capture-output -n ndp_sim \
     python nanoTabPFN_version10_original02/train.py \
       --device cuda \
       --checkpoint-dir $RUN_DIR \
       --disable-wandb \
       --log-every 10 \
       --checkpoint-interval 300 \
     2>&1 | tee -a $RUN_DIR/train.log"

tmux ls
