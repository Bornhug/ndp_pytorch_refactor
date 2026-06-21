#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

NT10_ENV_FILE="${NT10_ENV_FILE:-$HOME/.nt10_vast_env}"
NT10_ENV_OVERRIDE_NAMES=(
  NT10_REPO
  NT10_BRANCH
  NT10_WORKDIR
  NT10_ENV
  NT10_CONDA_DIR
  NT10_REMOTE
  NT10_SYNC_INTERVAL
  NT10_TMUX_SESSION
  NT10_SYNC_TMUX_SESSION
  NT10_DEVICE
  NT10_WANDB_API_KEY
  NT10_WANDB_MODE
  NT10_DISABLE_WANDB
  NT10_WANDB_PROJECT
  NT10_WANDB_ENTITY
  NT10_WANDB_RUN_NAME
  NT10_WANDB_DIR
  NT10_PRIOR_NAME
  NT10_PRIOR_DIR
  NT10_PRIOR_TMP_DIR
  NT10_PRIOR_ARCHIVE_REMOTE
  NT10_PRIOR_EXPECTED_BATCHES
  NT10_PRIOR_OVERWRITE_TMP
  NT10_RUN_NAME
  NT10_CHECKPOINT_DIR
  NT10_ALLOW_NONEMPTY_CHECKPOINT_DIR
  NT10_BORDERS_PATH
  NT10_TABPFN_VERSION
  NT10_NUM_BARS
  NT10_NUM_EPOCHS
  NT10_NUM_STEPS
  NT10_BATCH_SIZE
  NT10_MICRO_BATCH_SIZE
  NT10_SHUFFLE_FILES
  NT10_SEED
  NT10_LR
  NT10_STD_MIN
  NT10_Z_MAX
  NT10_HARD_Z_THRESHOLD
  NT10_MAX_HARD_Z_FRAC
  NT10_LOG_EVERY
  NT10_CHECKPOINT_INTERVAL
  NT10_EMBEDDING_SIZE
  NT10_NUM_ATTENTION_HEADS
  NT10_MLP_HIDDEN_SIZE
  NT10_NUM_LAYERS
  NT10_PIP_INDEX_URL
  NT10_PIP_TRUSTED_HOST
  NT10_CONDA_FORGE_CHANNEL
  NT10_CUDA_CHANNEL
  NT10_INSTALL_CUDA_NVCC
  NT10_HF_HOME
  NT10_EVAL_CHECKPOINT
  NT10_EVAL_MAX_FEATURES
  NT10_EVAL_NEW_INSTANCES
  NT10_EVAL_SPLITS
  NT10_EVAL_RANDOM_STATE
  NT10_EVAL_NUM_BINS
  NT10_EVAL_SHARPNESS_COVERAGE
  NT10_EVAL_OUTPUT
  NT10_EVAL_VERBOSE
  NT10_SWEEP_CHECKPOINTS_DIR
  NT10_SWEEP_OUTPUT
  NT10_SWEEP_PLOT_PATH
  NT10_SWEEP_PLOT_METRIC
)
for name in "${NT10_ENV_OVERRIDE_NAMES[@]}"; do
  if [[ -v $name ]]; then
    printf -v "NT10_OVERRIDE_$name" '%s' "${!name}"
  fi
done
if [[ -f "$NT10_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$NT10_ENV_FILE"
fi
for name in "${NT10_ENV_OVERRIDE_NAMES[@]}"; do
  override="NT10_OVERRIDE_$name"
  if [[ -v $override ]]; then
    printf -v "$name" '%s' "${!override}"
    export "$name"
  fi
done
unset NT10_ENV_OVERRIDE_NAMES name override

NT10_REPO="${NT10_REPO:-https://github.com/Bornhug/ndp_pytorch_refactor.git}"
NT10_BRANCH="${NT10_BRANCH:-branch01}"
NT10_WORKDIR="${NT10_WORKDIR:-/workspace/ndp_pytorch_refactor}"
NT10_ENV="${NT10_ENV:-ndp_sim}"
NT10_CONDA_DIR="${NT10_CONDA_DIR:-$HOME/miniconda3}"
NT10_REMOTE="${NT10_REMOTE:-}"
NT10_SYNC_INTERVAL="${NT10_SYNC_INTERVAL:-1800}"
NT10_TMUX_SESSION="${NT10_TMUX_SESSION:-nt10-train}"
NT10_SYNC_TMUX_SESSION="${NT10_SYNC_TMUX_SESSION:-nt10-sync}"
NT10_DEVICE="${NT10_DEVICE:-cuda}"
NT10_WANDB_API_KEY="${NT10_WANDB_API_KEY:-${WANDB_API_KEY:-}}"
NT10_DISABLE_WANDB="${NT10_DISABLE_WANDB:-1}"
NT10_WANDB_PROJECT="${NT10_WANDB_PROJECT:-nanoTabPFN_version10_original02}"
NT10_WANDB_ENTITY="${NT10_WANDB_ENTITY:-}"
NT10_WANDB_RUN_NAME="${NT10_WANDB_RUN_NAME:-}"
NT10_WANDB_MODE="${NT10_WANDB_MODE:-}"
NT10_PRIOR_NAME="${NT10_PRIOR_NAME:-stage1_continuous_y_seed123}"
NT10_PRIOR_DIR="${NT10_PRIOR_DIR:-/workspace/prior/$NT10_PRIOR_NAME}"
NT10_PRIOR_TMP_DIR="${NT10_PRIOR_TMP_DIR:-${NT10_PRIOR_DIR}.tmp}"
NT10_PRIOR_EXPECTED_BATCHES="${NT10_PRIOR_EXPECTED_BATCHES:-15000}"
NT10_PRIOR_OVERWRITE_TMP="${NT10_PRIOR_OVERWRITE_TMP:-0}"
if [[ -z "${NT10_PRIOR_ARCHIVE_REMOTE:-}" && -n "$NT10_REMOTE" ]]; then
  NT10_PRIOR_ARCHIVE_REMOTE="$NT10_REMOTE/runs/prior/${NT10_PRIOR_NAME}.zip"
else
  NT10_PRIOR_ARCHIVE_REMOTE="${NT10_PRIOR_ARCHIVE_REMOTE:-}"
fi
NT10_RUN_NAME="${NT10_RUN_NAME:-vast_l5_bars5000_seed123}"
NT10_ROOT="$NT10_WORKDIR/nanoTabPFN_version10_original02"
NT10_CHECKPOINT_DIR="${NT10_CHECKPOINT_DIR:-$NT10_ROOT/runs/$NT10_RUN_NAME}"
NT10_ALLOW_NONEMPTY_CHECKPOINT_DIR="${NT10_ALLOW_NONEMPTY_CHECKPOINT_DIR:-0}"
NT10_BORDERS_PATH="${NT10_BORDERS_PATH:-$NT10_ROOT/assets/tabpfn_v2_regressor_borders_5000.pt}"
NT10_TABPFN_VERSION="${NT10_TABPFN_VERSION:-v2}"
NT10_NUM_BARS="${NT10_NUM_BARS:-5000}"
NT10_NUM_EPOCHS="${NT10_NUM_EPOCHS:-10}"
NT10_NUM_STEPS="${NT10_NUM_STEPS:-}"
NT10_BATCH_SIZE="${NT10_BATCH_SIZE:-32}"
NT10_MICRO_BATCH_SIZE="${NT10_MICRO_BATCH_SIZE:-4}"
NT10_SHUFFLE_FILES="${NT10_SHUFFLE_FILES:-0}"
NT10_SEED="${NT10_SEED:-0}"
NT10_LR="${NT10_LR:-2e-4}"
NT10_STD_MIN="${NT10_STD_MIN:-5e-2}"
NT10_Z_MAX="${NT10_Z_MAX:-5}"
NT10_HARD_Z_THRESHOLD="${NT10_HARD_Z_THRESHOLD:-5}"
NT10_MAX_HARD_Z_FRAC="${NT10_MAX_HARD_Z_FRAC:-0.01}"
NT10_LOG_EVERY="${NT10_LOG_EVERY:-10}"
NT10_CHECKPOINT_INTERVAL="${NT10_CHECKPOINT_INTERVAL:-300}"
NT10_EMBEDDING_SIZE="${NT10_EMBEDDING_SIZE:-96}"
NT10_NUM_ATTENTION_HEADS="${NT10_NUM_ATTENTION_HEADS:-4}"
NT10_MLP_HIDDEN_SIZE="${NT10_MLP_HIDDEN_SIZE:-192}"
NT10_NUM_LAYERS="${NT10_NUM_LAYERS:-5}"
NT10_CONDA_FORGE_CHANNEL="${NT10_CONDA_FORGE_CHANNEL:-conda-forge}"
NT10_CUDA_CHANNEL="${NT10_CUDA_CHANNEL:-nvidia}"
NT10_INSTALL_CUDA_NVCC="${NT10_INSTALL_CUDA_NVCC:-0}"
NT10_HF_HOME="${NT10_HF_HOME:-/workspace/hf_home}"
NT10_SKLEARN_DATA="$NT10_ROOT/.sklearn_data"
NT10_WANDB_DIR="${NT10_WANDB_DIR:-$NT10_ROOT/wandb}"
NT10_LOG_DIR="$NT10_ROOT/runs/vast_logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

is_truthy() {
  local value
  value="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif have sudo; then
    sudo "$@"
  else
    die "Need root privileges or sudo to run: $*"
  fi
}

repo_script_path() {
  local path="$NT10_ROOT/scripts/$(basename "$SCRIPT_PATH")"
  if [[ -f "$path" ]]; then
    printf '%s' "$path"
  else
    printf '%s' "$SCRIPT_PATH"
  fi
}

usage() {
  cat <<'EOF'
Usage:
  bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh <command>

Required for cloud sync:
  NT10_REMOTE="gdrive:ndp-nanotabpfn-v10-original02"

Repository/runtime:
  NT10_REPO="https://github.com/Bornhug/ndp_pytorch_refactor.git"
  NT10_BRANCH="branch01"
  NT10_WORKDIR="/workspace/ndp_pytorch_refactor"
  NT10_ENV="ndp_sim"
  NT10_ENV_FILE="$HOME/.nt10_vast_env"

Prior data:
  NT10_PRIOR_DIR="/workspace/prior/stage1_continuous_y_seed123"
  NT10_PRIOR_ARCHIVE_REMOTE="$NT10_REMOTE/runs/prior/stage1_continuous_y_seed123.zip"
  NT10_PRIOR_TMP_DIR="$NT10_PRIOR_DIR.tmp"
  NT10_PRIOR_EXPECTED_BATCHES="15000"
  NT10_PRIOR_OVERWRITE_TMP="0"

Training:
  NT10_RUN_NAME="vast_l5_bars5000_seed123"
  NT10_CHECKPOINT_DIR="$NT10_ROOT/runs/$NT10_RUN_NAME"
  NT10_ALLOW_NONEMPTY_CHECKPOINT_DIR="0"
  NT10_DISABLE_WANDB="1"
  NT10_DEVICE="cuda"
  NT10_NUM_EPOCHS="10"
  NT10_NUM_STEPS=""       Optional exact update count; overrides epochs when set.
  NT10_BATCH_SIZE="32"
  NT10_MICRO_BATCH_SIZE="4"
  NT10_SEED="0"
  NT10_LR="2e-4"
  NT10_NUM_LAYERS="5"
  NT10_NUM_BARS="5000"
  NT10_BORDERS_PATH="$NT10_ROOT/assets/tabpfn_v2_regressor_borders_5000.pt"

Commands:
  setup          Install tools/env, sparse-checkout nanoTabPFN, sync cloud state, run doctor.
  pull-code      Clone or update branch01 with sparse checkout for nanoTabPFN_version10_original02.
  install-env    Install Miniconda if needed and install nanoTabPFN requirements.
  sync-pull      Copy runs/assets/sklearn cache/wandb from NT10_REMOTE to this instance.
  sync-push      Copy local runs/assets/sklearn cache/wandb back to NT10_REMOTE.
  sync-loop      Run sync-push every NT10_SYNC_INTERVAL seconds.
  prior-status   Show prior archive, disk, temp/final counts, and samples.
  prior-extract  Stream prior ZIP from rclone into NT10_PRIOR_TMP_DIR only.
  prior-activate Verify temp prior data, then move it to NT10_PRIOR_DIR.
  borders        Verify existing TabPFN v2 border asset, or generate it if missing.
  doctor         Check GPU, conda, PyTorch CUDA, repo, rclone, borders, and prior data.
  train          Run nanoTabPFN training. Refuses existing checkpoints unless explicitly allowed.
  train-tmux     Start train inside tmux session nt10-train.
  eval-latest    Evaluate the latest checkpoint under NT10_CHECKPOINT_DIR.
  sweep          Sweep checkpoints under NT10_CHECKPOINT_DIR.
  status         Show checkpoints/results, branch, cache sizes, and recent logs.
  shell          Open an activated conda shell in the repo.
  menu           Interactive command menu.
  help           Show this message.
EOF
}

ensure_nt10_dirs() {
  mkdir -p "$NT10_ROOT/runs" "$NT10_CHECKPOINT_DIR" "$NT10_SKLEARN_DATA" \
    "$NT10_WANDB_DIR" "$NT10_LOG_DIR" "$(dirname "$NT10_BORDERS_PATH")" \
    "$NT10_HF_HOME"
}

export_runtime_env() {
  export NT10_ROOT
  export NT10_PRIOR_DIR
  export NT10_BORDERS_PATH
  export SCIKIT_LEARN_DATA="$NT10_SKLEARN_DATA"
  export WANDB_DIR="$NT10_WANDB_DIR"
  export HF_HOME="$NT10_HF_HOME"
  if [[ -n "$NT10_WANDB_MODE" ]]; then
    export WANDB_MODE="$NT10_WANDB_MODE"
  fi
  if [[ -n "$NT10_WANDB_API_KEY" && "$NT10_WANDB_API_KEY" != \<*\> ]]; then
    export WANDB_API_KEY="$NT10_WANDB_API_KEY"
  fi
  ensure_nt10_dirs
}

ensure_rclone_config_dir() {
  mkdir -p "$HOME/.config/rclone"
}

write_env_file() {
  local target
  target="$NT10_ENV_FILE"
  mkdir -p "$(dirname "$target")"
  umask 077
  {
    printf '# nanoTabPFN_version10_original02 Vast environment. Generated by setup.\n'
    printf 'export NT10_REPO=%q\n' "$NT10_REPO"
    printf 'export NT10_BRANCH=%q\n' "$NT10_BRANCH"
    printf 'export NT10_WORKDIR=%q\n' "$NT10_WORKDIR"
    printf 'export NT10_ENV=%q\n' "$NT10_ENV"
    printf 'export NT10_CONDA_DIR=%q\n' "$NT10_CONDA_DIR"
    printf 'export NT10_REMOTE=%q\n' "$NT10_REMOTE"
    printf 'export NT10_SYNC_INTERVAL=%q\n' "$NT10_SYNC_INTERVAL"
    printf 'export NT10_TMUX_SESSION=%q\n' "$NT10_TMUX_SESSION"
    printf 'export NT10_SYNC_TMUX_SESSION=%q\n' "$NT10_SYNC_TMUX_SESSION"
    printf 'export NT10_DEVICE=%q\n' "$NT10_DEVICE"
    printf 'export NT10_PRIOR_NAME=%q\n' "$NT10_PRIOR_NAME"
    printf 'export NT10_PRIOR_DIR=%q\n' "$NT10_PRIOR_DIR"
    printf 'export NT10_PRIOR_TMP_DIR=%q\n' "$NT10_PRIOR_TMP_DIR"
    printf 'export NT10_PRIOR_ARCHIVE_REMOTE=%q\n' "$NT10_PRIOR_ARCHIVE_REMOTE"
    printf 'export NT10_PRIOR_EXPECTED_BATCHES=%q\n' "$NT10_PRIOR_EXPECTED_BATCHES"
    printf 'export NT10_PRIOR_OVERWRITE_TMP=%q\n' "$NT10_PRIOR_OVERWRITE_TMP"
    printf 'export NT10_RUN_NAME=%q\n' "$NT10_RUN_NAME"
    printf 'export NT10_CHECKPOINT_DIR=%q\n' "$NT10_CHECKPOINT_DIR"
    printf 'export NT10_ALLOW_NONEMPTY_CHECKPOINT_DIR=%q\n' "$NT10_ALLOW_NONEMPTY_CHECKPOINT_DIR"
    printf 'export NT10_BORDERS_PATH=%q\n' "$NT10_BORDERS_PATH"
    printf 'export NT10_TABPFN_VERSION=%q\n' "$NT10_TABPFN_VERSION"
    printf 'export NT10_NUM_BARS=%q\n' "$NT10_NUM_BARS"
    printf 'export NT10_NUM_EPOCHS=%q\n' "$NT10_NUM_EPOCHS"
    printf 'export NT10_NUM_STEPS=%q\n' "$NT10_NUM_STEPS"
    printf 'export NT10_BATCH_SIZE=%q\n' "$NT10_BATCH_SIZE"
    printf 'export NT10_MICRO_BATCH_SIZE=%q\n' "$NT10_MICRO_BATCH_SIZE"
    printf 'export NT10_SHUFFLE_FILES=%q\n' "$NT10_SHUFFLE_FILES"
    printf 'export NT10_SEED=%q\n' "$NT10_SEED"
    printf 'export NT10_LR=%q\n' "$NT10_LR"
    printf 'export NT10_STD_MIN=%q\n' "$NT10_STD_MIN"
    printf 'export NT10_Z_MAX=%q\n' "$NT10_Z_MAX"
    printf 'export NT10_HARD_Z_THRESHOLD=%q\n' "$NT10_HARD_Z_THRESHOLD"
    printf 'export NT10_MAX_HARD_Z_FRAC=%q\n' "$NT10_MAX_HARD_Z_FRAC"
    printf 'export NT10_LOG_EVERY=%q\n' "$NT10_LOG_EVERY"
    printf 'export NT10_CHECKPOINT_INTERVAL=%q\n' "$NT10_CHECKPOINT_INTERVAL"
    printf 'export NT10_EMBEDDING_SIZE=%q\n' "$NT10_EMBEDDING_SIZE"
    printf 'export NT10_NUM_ATTENTION_HEADS=%q\n' "$NT10_NUM_ATTENTION_HEADS"
    printf 'export NT10_MLP_HIDDEN_SIZE=%q\n' "$NT10_MLP_HIDDEN_SIZE"
    printf 'export NT10_NUM_LAYERS=%q\n' "$NT10_NUM_LAYERS"
    printf 'export NT10_DISABLE_WANDB=%q\n' "$NT10_DISABLE_WANDB"
    printf 'export NT10_WANDB_PROJECT=%q\n' "$NT10_WANDB_PROJECT"
    printf 'export NT10_WANDB_ENTITY=%q\n' "$NT10_WANDB_ENTITY"
    printf 'export NT10_WANDB_RUN_NAME=%q\n' "$NT10_WANDB_RUN_NAME"
    printf 'export NT10_WANDB_MODE=%q\n' "$NT10_WANDB_MODE"
    printf 'export NT10_WANDB_DIR=%q\n' "$NT10_WANDB_DIR"
    printf 'export NT10_CONDA_FORGE_CHANNEL=%q\n' "$NT10_CONDA_FORGE_CHANNEL"
    printf 'export NT10_CUDA_CHANNEL=%q\n' "$NT10_CUDA_CHANNEL"
    printf 'export NT10_INSTALL_CUDA_NVCC=%q\n' "$NT10_INSTALL_CUDA_NVCC"
    printf 'export NT10_HF_HOME=%q\n' "$NT10_HF_HOME"
    if [[ -n "$NT10_WANDB_API_KEY" && "$NT10_WANDB_API_KEY" != \<*\> ]]; then
      printf 'export NT10_WANDB_API_KEY=%q\n' "$NT10_WANDB_API_KEY"
    fi
  } >"$target"
  chmod 600 "$target"
  log "Saved nanoTabPFN Vast environment to $target."
}

source_conda() {
  if have conda; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
  elif [[ -f "$NT10_CONDA_DIR/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$NT10_CONDA_DIR/etc/profile.d/conda.sh"
  else
    die "conda is not available. Run install-env or setup first."
  fi
}

conda_env_exists() {
  source_conda
  conda env list | awk '{print $1}' | grep -qx "$NT10_ENV"
}

run_in_env() {
  source_conda
  export_runtime_env
  cd "$NT10_WORKDIR"
  conda run --no-capture-output -n "$NT10_ENV" "$@"
}

install_system_tools() {
  log "Installing system tools."
  if have apt-get; then
    as_root apt-get update
    as_root apt-get install -y git curl ca-certificates tmux rclone bzip2 unzip nano
    as_root apt-get install -y libarchive-tools
  else
    for tool in git curl tmux rclone bsdtar; do
      have "$tool" || die "Missing $tool and apt-get is unavailable."
    done
  fi
}

install_miniconda_if_needed() {
  if have conda || [[ -f "$NT10_CONDA_DIR/etc/profile.d/conda.sh" ]]; then
    log "Conda already available."
    return
  fi

  local arch installer tmp_dir
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) installer="Miniconda3-latest-Linux-x86_64.sh" ;;
    aarch64|arm64) installer="Miniconda3-latest-Linux-aarch64.sh" ;;
    *) die "Unsupported architecture for Miniconda: $arch" ;;
  esac

  tmp_dir="$(mktemp -d)"
  log "Installing Miniconda to $NT10_CONDA_DIR."
  curl -fsSL "https://repo.anaconda.com/miniconda/$installer" -o "$tmp_dir/miniconda.sh"
  bash "$tmp_dir/miniconda.sh" -b -p "$NT10_CONDA_DIR"
  rm -rf "$tmp_dir"
}

pull_code() {
  log "Preparing sparse checkout at $NT10_WORKDIR."
  if [[ ! -d "$NT10_WORKDIR/.git" ]]; then
    mkdir -p "$(dirname "$NT10_WORKDIR")"
    git clone --filter=blob:none --no-checkout --branch "$NT10_BRANCH" \
      "$NT10_REPO" "$NT10_WORKDIR"
    git -C "$NT10_WORKDIR" sparse-checkout init --cone
    git -C "$NT10_WORKDIR" sparse-checkout set nanoTabPFN_version10_original02
    git -C "$NT10_WORKDIR" checkout "$NT10_BRANCH"
  else
    git -C "$NT10_WORKDIR" fetch origin "$NT10_BRANCH"
    git -C "$NT10_WORKDIR" checkout "$NT10_BRANCH"
    git -C "$NT10_WORKDIR" sparse-checkout init --cone || true
    git -C "$NT10_WORKDIR" sparse-checkout set nanoTabPFN_version10_original02
    git -C "$NT10_WORKDIR" pull --ff-only origin "$NT10_BRANCH"
  fi

  [[ -d "$NT10_ROOT" ]] || die "Sparse checkout did not create $NT10_ROOT"
  ensure_nt10_dirs
}

install_env() {
  [[ -f "$NT10_ROOT/requirements.txt" ]] || pull_code
  install_miniconda_if_needed
  source_conda

  if conda_env_exists; then
    log "Conda env $NT10_ENV already exists."
  else
    log "Creating conda env $NT10_ENV with Python 3.11."
    conda create -n "$NT10_ENV" -y --override-channels \
      -c "$NT10_CONDA_FORGE_CHANNEL" python=3.11 pip
  fi

  log "Ensuring pip is installed in $NT10_ENV."
  conda install -n "$NT10_ENV" -y --override-channels \
    -c "$NT10_CONDA_FORGE_CHANNEL" pip

  if is_truthy "$NT10_INSTALL_CUDA_NVCC"; then
    log "Installing CUDA compiler tools into $NT10_ENV."
    conda install -n "$NT10_ENV" -y --override-channels \
      -c "$NT10_CUDA_CHANNEL" -c "$NT10_CONDA_FORGE_CHANNEL" cuda-nvcc=12.9.86
  fi

  log "Installing nanoTabPFN Python requirements."
  conda run --no-capture-output -n "$NT10_ENV" python -m pip install --upgrade pip
  conda run --no-capture-output -n "$NT10_ENV" python -m pip install \
    -i "${NT10_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    --trusted-host "${NT10_PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}" \
    -r "$NT10_ROOT/requirements.txt"
}

configure_wandb() {
  export_runtime_env
  if is_truthy "$NT10_DISABLE_WANDB"; then
    log "W&B is disabled by NT10_DISABLE_WANDB=1."
    return
  fi
  case "$NT10_WANDB_MODE" in
    disabled|offline)
      log "W&B mode is $NT10_WANDB_MODE; skipping online login."
      return
      ;;
  esac

  if [[ -z "$NT10_WANDB_API_KEY" || "$NT10_WANDB_API_KEY" == \<*\> ]]; then
    log "W&B API key not set. Training may fail if wandb is enabled and no login exists."
    return
  fi

  conda_env_exists || die "Conda env $NT10_ENV does not exist. Run install-env first."
  log "Configuring W&B login from NT10_WANDB_API_KEY/WANDB_API_KEY."
  run_in_env python - <<'PY'
import os
import wandb

key = os.environ.get("WANDB_API_KEY")
if not key:
    raise SystemExit("WANDB_API_KEY is missing")
wandb.login(key=key, relogin=True)
print("W&B login configured.")
PY
}

require_remote() {
  [[ -n "$NT10_REMOTE" ]] || die "NT10_REMOTE is required for sync commands."
  have rclone || die "rclone is not installed. Run setup or install rclone first."
}

remote_path() {
  printf '%s/%s' "$NT10_REMOTE" "$1"
}

copy_remote_dir_if_present() {
  local name local_dir remote_dir
  name="$1"
  local_dir="$2"
  shift 2
  remote_dir="$(remote_path "$name")"

  mkdir -p "$local_dir"
  if rclone lsf "$remote_dir" >/dev/null 2>&1; then
    log "Pulling $remote_dir -> $local_dir"
    rclone copy "$remote_dir" "$local_dir" --progress "$@"
  else
    log "Remote path missing or empty; skipping pull: $remote_dir"
  fi
}

copy_local_dir() {
  local name local_dir remote_dir
  name="$1"
  local_dir="$2"
  shift 2
  remote_dir="$(remote_path "$name")"

  mkdir -p "$local_dir"
  log "Pushing $local_dir -> $remote_dir"
  rclone copy "$local_dir" "$remote_dir" --progress "$@"
}

sync_pull() {
  require_remote
  ensure_nt10_dirs
  copy_remote_dir_if_present "runs" "$NT10_ROOT/runs" --exclude "prior/**"
  copy_remote_dir_if_present "assets" "$NT10_ROOT/assets"
  copy_remote_dir_if_present "sklearn_data" "$NT10_SKLEARN_DATA"
  copy_remote_dir_if_present "wandb" "$NT10_WANDB_DIR"
}

sync_push() {
  require_remote
  ensure_nt10_dirs
  copy_local_dir "runs" "$NT10_ROOT/runs" --exclude "prior/**"
  copy_local_dir "assets" "$NT10_ROOT/assets"
  copy_local_dir "sklearn_data" "$NT10_SKLEARN_DATA"
  copy_local_dir "wandb" "$NT10_WANDB_DIR"
}

sync_loop() {
  require_remote
  mkdir -p "$NT10_LOG_DIR"
  log "Starting sync loop every $NT10_SYNC_INTERVAL seconds."
  while true; do
    sync_push 2>&1 | tee -a "$NT10_LOG_DIR/sync-loop.log" || true
    sleep "$NT10_SYNC_INTERVAL"
  done
}

require_prior_archive_remote() {
  [[ -n "$NT10_PRIOR_ARCHIVE_REMOTE" ]] || die \
    "NT10_PRIOR_ARCHIVE_REMOTE is required. Set NT10_REMOTE or NT10_PRIOR_ARCHIVE_REMOTE."
  have rclone || die "rclone is not installed. Run setup or install rclone first."
}

batch_count() {
  local dir
  dir="$1"
  if [[ -d "$dir" ]]; then
    find "$dir" -maxdepth 1 -type f -name 'batch_*.pt' 2>/dev/null | wc -l | tr -d '[:space:]'
  else
    printf '0'
  fi
}

sample_batches() {
  local dir
  dir="$1"
  if [[ ! -d "$dir" ]]; then
    printf 'missing\t%s\n' "$dir"
    return
  fi

  printf 'first files:\n'
  find "$dir" -maxdepth 1 -type f -name 'batch_*.pt' 2>/dev/null | sort | sed -n '1,3p'
  printf 'last files:\n'
  find "$dir" -maxdepth 1 -type f -name 'batch_*.pt' 2>/dev/null | sort | tail -n 3
}

du_if_present() {
  local path
  path="$1"
  if [[ -e "$path" ]]; then
    du -sh "$path" 2>/dev/null || true
  else
    printf 'missing\t%s\n' "$path"
  fi
}

df_for_path() {
  local path probe
  path="$1"
  probe="$path"
  while [[ ! -e "$probe" && "$probe" != "/" ]]; do
    probe="$(dirname "$probe")"
  done
  df -h "$probe" 2>/dev/null || df -h / || true
}

prior_status() {
  log "Prior: remote archive"
  if [[ -n "$NT10_PRIOR_ARCHIVE_REMOTE" ]] && have rclone; then
    rclone lsl "$NT10_PRIOR_ARCHIVE_REMOTE" || true
    rclone md5sum "$NT10_PRIOR_ARCHIVE_REMOTE" || true
  elif [[ -z "$NT10_PRIOR_ARCHIVE_REMOTE" ]]; then
    log "WARNING: NT10_PRIOR_ARCHIVE_REMOTE is not set."
  else
    log "WARNING: rclone is not installed."
  fi

  log "Prior: disk"
  df_for_path "$NT10_PRIOR_DIR"

  log "Prior: final directory"
  du_if_present "$NT10_PRIOR_DIR"
  printf 'batch count: %s\n' "$(batch_count "$NT10_PRIOR_DIR")"
  sample_batches "$NT10_PRIOR_DIR"

  log "Prior: temp directory"
  du_if_present "$NT10_PRIOR_TMP_DIR"
  printf 'batch count: %s\n' "$(batch_count "$NT10_PRIOR_TMP_DIR")"
  sample_batches "$NT10_PRIOR_TMP_DIR"
}

prior_extract() {
  require_prior_archive_remote
  have bsdtar || die "bsdtar is missing. Run setup to install libarchive-tools."

  [[ ! -e "$NT10_PRIOR_DIR" ]] || die \
    "Final prior path already exists: $NT10_PRIOR_DIR. Refusing to create a second copy."

  if [[ -e "$NT10_PRIOR_TMP_DIR" ]]; then
    if ! is_truthy "$NT10_PRIOR_OVERWRITE_TMP"; then
      die "Temp prior path already exists: $NT10_PRIOR_TMP_DIR. Set NT10_PRIOR_OVERWRITE_TMP=1 to replace it."
    fi
    [[ -n "$NT10_PRIOR_TMP_DIR" && "$NT10_PRIOR_TMP_DIR" != "/" && "$NT10_PRIOR_TMP_DIR" == *.tmp ]] || \
      die "Refusing to remove unsafe temp path: $NT10_PRIOR_TMP_DIR"
    log "Removing existing temp prior path because NT10_PRIOR_OVERWRITE_TMP=1: $NT10_PRIOR_TMP_DIR"
    rm -rf -- "$NT10_PRIOR_TMP_DIR"
  fi

  mkdir -p "$NT10_PRIOR_TMP_DIR"
  log "Streaming prior archive without storing the ZIP locally:"
  log "  $NT10_PRIOR_ARCHIVE_REMOTE -> $NT10_PRIOR_TMP_DIR"
  rclone cat "$NT10_PRIOR_ARCHIVE_REMOTE" | bsdtar -xf - -C "$NT10_PRIOR_TMP_DIR"

  local count
  count="$(batch_count "$NT10_PRIOR_TMP_DIR")"
  log "Extracted $count batch files into temp prior path."
  if [[ "$NT10_PRIOR_EXPECTED_BATCHES" != "0" && "$count" != "$NT10_PRIOR_EXPECTED_BATCHES" ]]; then
    die "Expected $NT10_PRIOR_EXPECTED_BATCHES batch files, found $count. Data remains in $NT10_PRIOR_TMP_DIR for inspection."
  fi
  log "Prior extraction complete. Inspect $NT10_PRIOR_TMP_DIR, then run prior-activate when ready."
}

prior_activate() {
  [[ -d "$NT10_PRIOR_TMP_DIR" ]] || die "Temp prior path missing: $NT10_PRIOR_TMP_DIR"
  [[ ! -e "$NT10_PRIOR_DIR" ]] || die "Final prior path already exists: $NT10_PRIOR_DIR"

  local count
  count="$(batch_count "$NT10_PRIOR_TMP_DIR")"
  if [[ "$NT10_PRIOR_EXPECTED_BATCHES" != "0" && "$count" != "$NT10_PRIOR_EXPECTED_BATCHES" ]]; then
    die "Expected $NT10_PRIOR_EXPECTED_BATCHES batch files in $NT10_PRIOR_TMP_DIR, found $count."
  fi

  mkdir -p "$(dirname "$NT10_PRIOR_DIR")"
  mv "$NT10_PRIOR_TMP_DIR" "$NT10_PRIOR_DIR"
  log "Activated prior data: $NT10_PRIOR_DIR"
  prior_status
}

verify_borders() {
  conda_env_exists || die "Conda env $NT10_ENV does not exist. Run install-env first."
  run_in_env python - <<'PY'
import os
import sys

root = os.environ["NT10_ROOT"]
sys.path.insert(0, root)
from bar_distribution import border_asset_metadata

meta = border_asset_metadata(
    os.environ["NT10_BORDERS_PATH"],
    expected_num_bars=int(os.environ.get("NT10_NUM_BARS", "5000")),
)
print("border asset:", meta)
PY
}

ensure_borders() {
  [[ -d "$NT10_ROOT" ]] || die "$NT10_ROOT not found. Run pull-code first."
  conda_env_exists || die "Conda env $NT10_ENV does not exist. Run install-env first."
  export_runtime_env

  if [[ -f "$NT10_BORDERS_PATH" ]]; then
    log "Border asset exists; verifying: $NT10_BORDERS_PATH"
    verify_borders
    return
  fi

  log "Border asset missing; extracting TabPFN $NT10_TABPFN_VERSION regressor borders."
  run_in_env python nanoTabPFN_version10_original02/scripts/extract_tabpfn_v2_borders.py \
    --version "$NT10_TABPFN_VERSION" \
    --device cpu \
    --num-bars "$NT10_NUM_BARS" \
    --output "$NT10_BORDERS_PATH"
  verify_borders
}

latest_checkpoint() {
  [[ -d "$NT10_CHECKPOINT_DIR" ]] || return 0
  find "$NT10_CHECKPOINT_DIR" -type f \( -name 'model_step_*.pt' -o -name 'model_final.pt' \) \
    2>/dev/null | sort -V | tail -n 1
}

latest_eval_json() {
  [[ -d "$NT10_ROOT/runs" ]] || return 0
  find "$NT10_ROOT/runs" -type f -name '*eval*.json' 2>/dev/null | sort -V | tail -n 1
}

check_training_inputs() {
  [[ -d "$NT10_ROOT" ]] || die "$NT10_ROOT not found. Run pull-code first."
  [[ -f "$NT10_BORDERS_PATH" ]] || die \
    "Border asset missing: $NT10_BORDERS_PATH. Run the borders command first."
  [[ -d "$NT10_PRIOR_DIR" ]] || die \
    "Prior directory missing: $NT10_PRIOR_DIR. Run prior-extract and prior-activate first."
  compgen -G "$NT10_PRIOR_DIR/batch_*.pt" >/dev/null || die \
    "No batch_*.pt files found in $NT10_PRIOR_DIR."

  local count
  count="$(batch_count "$NT10_PRIOR_DIR")"
  if [[ "$NT10_PRIOR_EXPECTED_BATCHES" != "0" && "$count" != "$NT10_PRIOR_EXPECTED_BATCHES" ]]; then
    die "Expected $NT10_PRIOR_EXPECTED_BATCHES batch_*.pt files in $NT10_PRIOR_DIR, found $count."
  fi
}

warn_training_inputs() {
  local ok=1 count
  [[ -d "$NT10_ROOT" ]] || {
    log "WARNING: repository folder missing: $NT10_ROOT."
    ok=0
  }
  [[ -f "$NT10_BORDERS_PATH" ]] || {
    log "WARNING: border asset missing: $NT10_BORDERS_PATH."
    ok=0
  }
  [[ -d "$NT10_PRIOR_DIR" ]] || {
    log "WARNING: prior directory missing: $NT10_PRIOR_DIR."
    ok=0
  }
  if [[ -d "$NT10_PRIOR_DIR" ]] && ! compgen -G "$NT10_PRIOR_DIR/batch_*.pt" >/dev/null; then
    log "WARNING: no batch_*.pt files found in $NT10_PRIOR_DIR."
    ok=0
  fi
  if [[ -d "$NT10_PRIOR_DIR" ]] && compgen -G "$NT10_PRIOR_DIR/batch_*.pt" >/dev/null; then
    count="$(batch_count "$NT10_PRIOR_DIR")"
    if [[ "$NT10_PRIOR_EXPECTED_BATCHES" != "0" && "$count" != "$NT10_PRIOR_EXPECTED_BATCHES" ]]; then
      log "WARNING: expected $NT10_PRIOR_EXPECTED_BATCHES batch_*.pt files in $NT10_PRIOR_DIR, found $count."
      ok=0
    fi
  fi
  if [[ "$ok" -eq 0 ]]; then
    log "Training will fail until the warnings above are fixed."
  fi
}

check_checkpoint_dir_safety() {
  mkdir -p "$NT10_CHECKPOINT_DIR"
  if is_truthy "$NT10_ALLOW_NONEMPTY_CHECKPOINT_DIR"; then
    return
  fi
  if find "$NT10_CHECKPOINT_DIR" -maxdepth 1 -type f \
    \( -name 'model_step_*.pt' -o -name 'model_final.pt' \) | grep -q .; then
    die "Checkpoint directory already contains model checkpoints: $NT10_CHECKPOINT_DIR. This train.py does not resume. Set a new NT10_RUN_NAME/NT10_CHECKPOINT_DIR, or set NT10_ALLOW_NONEMPTY_CHECKPOINT_DIR=1 if overwriting is intentional."
  fi
}

doctor() {
  log "Doctor: repository"
  if [[ -d "$NT10_WORKDIR/.git" ]]; then
    git -C "$NT10_WORKDIR" status --short --branch || true
    git -C "$NT10_WORKDIR" sparse-checkout list || true
  else
    log "Repository missing at $NT10_WORKDIR."
  fi

  log "Doctor: GPU"
  if have nvidia-smi; then
    nvidia-smi
  else
    log "WARNING: nvidia-smi not found."
  fi

  log "Doctor: conda/PyTorch"
  if have conda || [[ -f "$NT10_CONDA_DIR/etc/profile.d/conda.sh" ]]; then
    source_conda
    if conda_env_exists; then
      run_in_env python - <<'PY' || true
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
    else
      log "WARNING: conda env $NT10_ENV does not exist."
    fi
  else
    log "WARNING: conda not installed."
  fi

  log "Doctor: rclone"
  if [[ -n "$NT10_REMOTE" ]] && have rclone; then
    rclone lsd "$NT10_REMOTE" || true
  elif [[ -z "$NT10_REMOTE" ]]; then
    log "WARNING: NT10_REMOTE is not set."
  else
    log "WARNING: rclone is not installed."
  fi

  log "Doctor: nanoTabPFN directories"
  du_if_present "$NT10_ROOT/runs"
  du_if_present "$NT10_ROOT/assets"
  du_if_present "$NT10_SKLEARN_DATA"
  du_if_present "$NT10_WANDB_DIR"

  log "Doctor: border asset"
  if [[ -f "$NT10_BORDERS_PATH" ]] && (have conda || [[ -f "$NT10_CONDA_DIR/etc/profile.d/conda.sh" ]]); then
    if conda_env_exists; then
      verify_borders || true
    else
      log "WARNING: conda env $NT10_ENV does not exist; cannot verify border asset."
      du_if_present "$NT10_BORDERS_PATH"
    fi
  else
    du_if_present "$NT10_BORDERS_PATH"
  fi

  log "Doctor: prior data"
  warn_training_inputs
}

train() {
  [[ -d "$NT10_ROOT" ]] || die "$NT10_ROOT not found. Run pull-code first."
  conda_env_exists || die "Conda env $NT10_ENV does not exist. Run install-env first."
  configure_wandb
  check_training_inputs
  check_checkpoint_dir_safety
  export_runtime_env

  local cmd=(
    python nanoTabPFN_version10_original02/train.py
    --device "$NT10_DEVICE"
    --prior-dir "$NT10_PRIOR_DIR"
    --batch-size "$NT10_BATCH_SIZE"
    --micro-batch-size "$NT10_MICRO_BATCH_SIZE"
    --seed "$NT10_SEED"
    --lr "$NT10_LR"
    --std-min "$NT10_STD_MIN"
    --z-max "$NT10_Z_MAX"
    --hard-z-threshold "$NT10_HARD_Z_THRESHOLD"
    --max-hard-z-frac "$NT10_MAX_HARD_Z_FRAC"
    --log-every "$NT10_LOG_EVERY"
    --embedding-size "$NT10_EMBEDDING_SIZE"
    --num-attention-heads "$NT10_NUM_ATTENTION_HEADS"
    --mlp-hidden-size "$NT10_MLP_HIDDEN_SIZE"
    --num-layers "$NT10_NUM_LAYERS"
    --num-bars "$NT10_NUM_BARS"
    --borders-path "$NT10_BORDERS_PATH"
    --checkpoint-interval "$NT10_CHECKPOINT_INTERVAL"
    --checkpoint-dir "$NT10_CHECKPOINT_DIR"
  )
  if [[ -n "$NT10_NUM_STEPS" ]]; then
    cmd+=(--num-steps "$NT10_NUM_STEPS")
  else
    cmd+=(--num-epochs "$NT10_NUM_EPOCHS")
  fi
  if is_truthy "$NT10_SHUFFLE_FILES"; then
    cmd+=(--shuffle-files)
  fi
  if is_truthy "$NT10_DISABLE_WANDB"; then
    cmd+=(--disable-wandb)
  else
    cmd+=(--wandb-project "$NT10_WANDB_PROJECT")
    if [[ -n "$NT10_WANDB_ENTITY" ]]; then
      cmd+=(--wandb-entity "$NT10_WANDB_ENTITY")
    fi
    if [[ -n "$NT10_WANDB_RUN_NAME" ]]; then
      cmd+=(--wandb-run-name "$NT10_WANDB_RUN_NAME")
    fi
  fi

  log "Starting nanoTabPFN training."
  log "Checkpoint directory: $NT10_CHECKPOINT_DIR"
  mkdir -p "$NT10_CHECKPOINT_DIR"
  run_in_env "${cmd[@]}" 2>&1 | tee -a "$NT10_CHECKPOINT_DIR/train.log"
}

train_tmux() {
  have tmux || die "tmux is not installed. Run setup first."
  if tmux has-session -t "$NT10_TMUX_SESSION" 2>/dev/null; then
    log "tmux session already exists: $NT10_TMUX_SESSION"
    log "Attach with: tmux attach -t $NT10_TMUX_SESSION"
    return
  fi
  tmux new-session -d -s "$NT10_TMUX_SESSION" \
    "cd '$NT10_WORKDIR' && bash '$(repo_script_path)' train"
  log "Started training tmux session: $NT10_TMUX_SESSION"
  log "Attach with: tmux attach -t $NT10_TMUX_SESSION"
}

start_sync_tmux() {
  have tmux || die "tmux is not installed. Run setup first."
  if tmux has-session -t "$NT10_SYNC_TMUX_SESSION" 2>/dev/null; then
    log "tmux session already exists: $NT10_SYNC_TMUX_SESSION"
    return
  fi
  tmux new-session -d -s "$NT10_SYNC_TMUX_SESSION" \
    "cd '$NT10_WORKDIR' && bash '$(repo_script_path)' sync-loop"
  log "Started sync tmux session: $NT10_SYNC_TMUX_SESSION"
}

eval_latest() {
  conda_env_exists || die "Conda env $NT10_ENV does not exist. Run install-env first."
  local checkpoint output
  checkpoint="${NT10_EVAL_CHECKPOINT:-$(latest_checkpoint)}"
  [[ -n "$checkpoint" ]] || die "No model_step_*.pt or model_final.pt checkpoint found under $NT10_CHECKPOINT_DIR."
  output="${NT10_EVAL_OUTPUT:-${checkpoint%.pt}_rows${NT10_EVAL_NEW_INSTANCES:-1000}_eval.json}"

  local cmd=(
    python nanoTabPFN_version10_original02/evaluation.py
    --checkpoint "$checkpoint"
    --device "$NT10_DEVICE"
    --max-features-eval "${NT10_EVAL_MAX_FEATURES:-32}"
    --new-instances-eval "${NT10_EVAL_NEW_INSTANCES:-1000}"
    --n-splits "${NT10_EVAL_SPLITS:-5}"
    --random-state "${NT10_EVAL_RANDOM_STATE:-0}"
    --num-bins "${NT10_EVAL_NUM_BINS:-10}"
    --sharpness-coverage "${NT10_EVAL_SHARPNESS_COVERAGE:-0.90}"
    --borders-path "$NT10_BORDERS_PATH"
    --output-json "$output"
  )
  if is_truthy "${NT10_EVAL_VERBOSE:-0}"; then
    cmd+=(--verbose)
  fi

  log "Evaluating checkpoint: $checkpoint"
  run_in_env "${cmd[@]}"
}

sweep() {
  conda_env_exists || die "Conda env $NT10_ENV does not exist. Run install-env first."
  local checkpoints_dir output plot_path
  checkpoints_dir="${NT10_SWEEP_CHECKPOINTS_DIR:-$NT10_CHECKPOINT_DIR}"
  output="${NT10_SWEEP_OUTPUT:-$checkpoints_dir/evaluation/sweep_results.json}"
  plot_path="${NT10_SWEEP_PLOT_PATH:-auto}"
  run_in_env python nanoTabPFN_version10_original02/tabarena_eval/run_sweep.py \
    --checkpoints-dir "$checkpoints_dir" \
    --device "$NT10_DEVICE" \
    --n-splits "${NT10_EVAL_SPLITS:-5}" \
    --random-state "${NT10_EVAL_RANDOM_STATE:-0}" \
    --max-features-eval "${NT10_EVAL_MAX_FEATURES:-32}" \
    --new-instances-eval "${NT10_EVAL_NEW_INSTANCES:-1000}" \
    --output-json "$output" \
    --plot-path "$plot_path" \
    --plot-metric "${NT10_SWEEP_PLOT_METRIC:-R2}"
}

status() {
  log "Status"
  if [[ -d "$NT10_WORKDIR/.git" ]]; then
    git -C "$NT10_WORKDIR" status --short --branch || true
    git -C "$NT10_WORKDIR" log -1 --oneline --decorate || true
  fi
  printf 'Prior directory: %s\n' "$NT10_PRIOR_DIR"
  printf 'Prior batch count: %s\n' "$(batch_count "$NT10_PRIOR_DIR")"
  printf 'Checkpoint directory: %s\n' "$NT10_CHECKPOINT_DIR"
  printf 'Latest checkpoint: %s\n' "$(latest_checkpoint)"
  printf 'Latest eval JSON: %s\n' "$(latest_eval_json)"
  du_if_present "$NT10_ROOT/runs"
  du_if_present "$NT10_ROOT/assets"
  du_if_present "$NT10_SKLEARN_DATA"
  du_if_present "$NT10_WANDB_DIR"
  if have tmux; then
    tmux ls 2>/dev/null || true
  fi
  if [[ -f "$NT10_LOG_DIR/sync-loop.log" ]]; then
    log "Recent sync-loop log"
    tail -n 20 "$NT10_LOG_DIR/sync-loop.log"
  fi
  if [[ -f "$NT10_CHECKPOINT_DIR/train.log" ]]; then
    log "Recent training log"
    tail -n 20 "$NT10_CHECKPOINT_DIR/train.log"
  fi
}

open_shell() {
  source_conda
  export_runtime_env
  cd "$NT10_WORKDIR"
  conda activate "$NT10_ENV"
  log "Activated $NT10_ENV in $NT10_WORKDIR."
  exec bash -i
}

menu() {
  while true; do
    cat <<'EOF'

nanoTabPFN v10 original02 Vast menu
  1) status
  2) doctor
  3) sync-pull
  4) sync-push
  5) prior-status
  6) prior-extract
  7) prior-activate
  8) borders
  9) train-tmux
  10) start-sync-tmux
  11) eval-latest
  12) sweep
  13) shell
  0) quit
EOF
    read -r -p "Select command: " choice
    case "$choice" in
      1) status ;;
      2) doctor ;;
      3) sync_pull ;;
      4) sync_push ;;
      5) prior_status ;;
      6) prior_extract ;;
      7) prior_activate ;;
      8) ensure_borders ;;
      9) train_tmux ;;
      10) start_sync_tmux ;;
      11) eval_latest ;;
      12) sweep ;;
      13) open_shell ;;
      0) exit 0 ;;
      *) log "Unknown choice: $choice" ;;
    esac
  done
}

setup() {
  ensure_rclone_config_dir
  write_env_file
  install_system_tools
  pull_code
  install_env
  configure_wandb
  sync_pull
  doctor
  log "Setup complete. Next run: bash $(repo_script_path) menu"
}

main() {
  local command="${1:-help}"
  case "$command" in
    setup) setup ;;
    pull-code) pull_code ;;
    install-env) install_env ;;
    sync-pull) sync_pull ;;
    sync-push) sync_push ;;
    sync-loop) sync_loop ;;
    start-sync-tmux) start_sync_tmux ;;
    prior-status) prior_status ;;
    prior-extract) prior_extract ;;
    prior-activate) prior_activate ;;
    borders) ensure_borders ;;
    doctor) doctor ;;
    train) train ;;
    train-tmux) train_tmux ;;
    eval-latest) eval_latest ;;
    sweep) sweep ;;
    status) status ;;
    shell) open_shell ;;
    menu) menu ;;
    help|-h|--help) usage ;;
    *) usage; die "Unknown command: $command" ;;
  esac
}

main "$@"
