#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

REG03_ENV_FILE="${REG03_ENV_FILE:-$HOME/.reg03_vast_env}"
REG03_ENV_OVERRIDE_NAMES=(
  REG03_REPO
  REG03_BRANCH
  REG03_WORKDIR
  REG03_ENV
  REG03_CONDA_DIR
  REG03_REMOTE
  REG03_SYNC_INTERVAL
  REG03_TMUX_SESSION
  REG03_DEVICE
  REG03_WANDB_API_KEY
  REG03_WANDB_MODE
  REG03_TABICL_REPO
  REG03_TABICL_REPO_URL
  REG03_TABICL_BRANCH
  REG03_PRIOR_DIR
  REG03_PRIOR_ARCHIVE_REMOTE
  REG03_PRIOR_TMP_DIR
  REG03_PRIOR_EXPECTED_BATCHES
  REG03_PRIOR_OVERWRITE_TMP
  REG03_WANDB_DIR
  REG03_CHECKPOINT_DIR
  REG03_AUTO_RESUME_LATEST
  REG03_PIP_INDEX_URL
  REG03_PIP_TRUSTED_HOST
  REG03_CONDA_FORGE_CHANNEL
  REG03_CUDA_CHANNEL
)
for name in "${REG03_ENV_OVERRIDE_NAMES[@]}"; do
  if [[ -v $name ]]; then
    printf -v "REG03_OVERRIDE_$name" '%s' "${!name}"
  fi
done
if [[ -f "$REG03_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$REG03_ENV_FILE"
fi
for name in "${REG03_ENV_OVERRIDE_NAMES[@]}"; do
  override="REG03_OVERRIDE_$name"
  if [[ -v $override ]]; then
    printf -v "$name" '%s' "${!override}"
    export "$name"
  fi
done
unset REG03_ENV_OVERRIDE_NAMES name override

REG03_REPO="${REG03_REPO:-https://github.com/Bornhug/ndp_pytorch_refactor.git}"
REG03_BRANCH="${REG03_BRANCH:-branch01}"
REG03_WORKDIR="${REG03_WORKDIR:-/workspace/ndp_pytorch_refactor}"
REG03_ENV="${REG03_ENV:-ndp_sim}"
REG03_CONDA_DIR="${REG03_CONDA_DIR:-$HOME/miniconda3}"
REG03_REMOTE="${REG03_REMOTE:-}"
REG03_SYNC_INTERVAL="${REG03_SYNC_INTERVAL:-600}"
REG03_TMUX_SESSION="${REG03_TMUX_SESSION:-reg03-train}"
REG03_DEVICE="${REG03_DEVICE:-cuda}"
REG03_WANDB_API_KEY="${REG03_WANDB_API_KEY:-${WANDB_API_KEY:-}}"
REG03_TABICL_REPO="${REG03_TABICL_REPO:-/workspace/tabicl}"
REG03_TABICL_REPO_URL="${REG03_TABICL_REPO_URL:-https://github.com/soda-inria/tabicl.git}"
REG03_TABICL_BRANCH="${REG03_TABICL_BRANCH:-main}"
REG03_PRIOR_DIR="${REG03_PRIOR_DIR:-/workspace/tabicl/data_regression/stage1}"
REG03_PRIOR_TMP_DIR="${REG03_PRIOR_TMP_DIR:-${REG03_PRIOR_DIR}.tmp}"
REG03_PRIOR_EXPECTED_BATCHES="${REG03_PRIOR_EXPECTED_BATCHES:-15000}"
REG03_PRIOR_OVERWRITE_TMP="${REG03_PRIOR_OVERWRITE_TMP:-0}"
if [[ -z "${REG03_PRIOR_ARCHIVE_REMOTE:-}" && -n "$REG03_REMOTE" ]]; then
  REG03_PRIOR_ARCHIVE_REMOTE="$REG03_REMOTE/runs/prior/stage1_15000batches.zip"
else
  REG03_PRIOR_ARCHIVE_REMOTE="${REG03_PRIOR_ARCHIVE_REMOTE:-}"
fi
REG03_CONDA_FORGE_CHANNEL="${REG03_CONDA_FORGE_CHANNEL:-conda-forge}"
REG03_CUDA_CHANNEL="${REG03_CUDA_CHANNEL:-nvidia}"

REG03_ROOT="$REG03_WORKDIR/TabICL_regression03"
REG03_SKLEARN_DATA="$REG03_ROOT/.sklearn_data"
REG03_TABARENA_CACHE="$REG03_ROOT/tabicl_style/.tabarena_cache"
REG03_WANDB_DIR="${REG03_WANDB_DIR:-$REG03_ROOT/wandb}"
REG03_CHECKPOINT_DIR="${REG03_CHECKPOINT_DIR:-}"
REG03_AUTO_RESUME_LATEST="${REG03_AUTO_RESUME_LATEST:-}"
REG03_LOG_DIR="$REG03_ROOT/runs/vast_logs"

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

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif have sudo; then
    sudo "$@"
  else
    die "Need root privileges or sudo to run: $*"
  fi
}

usage() {
  cat <<'EOF'
Usage:
  bash TabICL_regression03/scripts/vast_regression03.sh <command>

Required for cloud sync:
  REG03_REMOTE="gdrive:ndp-regression03"

Common configuration:
  REG03_REPO="https://github.com/Bornhug/ndp_pytorch_refactor.git"
  REG03_BRANCH="branch01"
  REG03_WORKDIR="/workspace/ndp_pytorch_refactor"
  REG03_ENV="ndp_sim"
  REG03_ENV_FILE="$HOME/.reg03_vast_env"

Training input paths:
  REG03_TABICL_REPO="/workspace/tabicl"
  REG03_TABICL_REPO_URL="https://github.com/soda-inria/tabicl.git"
  REG03_TABICL_BRANCH="main"
  REG03_PRIOR_DIR="/workspace/tabicl/data_regression/stage1"
  REG03_PRIOR_ARCHIVE_REMOTE="$REG03_REMOTE/runs/prior/stage1_15000batches.zip"
  REG03_PRIOR_TMP_DIR="$REG03_PRIOR_DIR.tmp"
  REG03_PRIOR_EXPECTED_BATCHES="15000"
  REG03_PRIOR_OVERWRITE_TMP="0"
  REG03_CHECKPOINT_DIR=""       Optional explicit training.checkpoint_dir.
  REG03_AUTO_RESUME_LATEST=""   Optional training.auto_resume_latest override: 1/0, true/false.
  REG03_CONDA_FORGE_CHANNEL="conda-forge"
  REG03_CUDA_CHANNEL="nvidia"

Optional W&B configuration:
  REG03_WANDB_API_KEY="..."  Login key used by setup/train, also exported as WANDB_API_KEY.
  REG03_WANDB_MODE="online"  Use "disabled" to turn W&B off for training.

Commands:
  setup          Install tools/env, sparse-checkout regression03, pull cloud state, run doctor.
  pull-code      Clone or update branch01 with sparse checkout for TabICL_regression03 only.
  pull-tabicl    Clone or update the external TabICL source repo.
  install-env    Install Miniconda if needed, create/update ndp_sim, install requirements.
  sync-pull      Copy runs/caches/wandb from REG03_REMOTE to this instance.
  sync-push      Copy local runs/caches/wandb back to REG03_REMOTE.
  sync-loop      Run sync-push every REG03_SYNC_INTERVAL seconds.
  prior-status   Show prior archive, disk, temp/final counts, and samples.
  prior-extract  Stream prior ZIP from rclone into REG03_PRIOR_TMP_DIR only.
  prior-activate Verify temp prior data, then move it to REG03_PRIOR_DIR.
  doctor         Check GPU, conda, PyTorch CUDA, repo, rclone, caches, and training inputs.
  train          Run regression03 training with auto-resume.
  train-tmux     Start train inside tmux session reg03-train.
  eval-latest    Evaluate the latest step-*.pt checkpoint.
  qice-latest    Compute QICE for the latest evaluation JSON.
  status         Show latest checkpoint/results, branch, cache sizes, and recent logs.
  shell          Open an activated regression03 conda shell.
  menu           Interactive command menu.
  help           Show this message.
EOF
}

ensure_reg03_dirs() {
  mkdir -p "$REG03_ROOT/runs" "$REG03_SKLEARN_DATA" "$REG03_TABARENA_CACHE" \
    "$REG03_WANDB_DIR" "$REG03_LOG_DIR"
}

export_runtime_env() {
  export SCIKIT_LEARN_DATA="$REG03_SKLEARN_DATA"
  export TABARENA_CACHE="$REG03_TABARENA_CACHE"
  export WANDB_DIR="$REG03_WANDB_DIR"
  if [[ -n "$REG03_WANDB_API_KEY" && "$REG03_WANDB_API_KEY" != \<*\> ]]; then
    export WANDB_API_KEY="$REG03_WANDB_API_KEY"
  fi
  ensure_reg03_dirs
}

ensure_rclone_config_dir() {
  mkdir -p "$HOME/.config/rclone"
}

write_env_file() {
  local target
  target="$REG03_ENV_FILE"
  mkdir -p "$(dirname "$target")"
  umask 077
  {
    printf '# TabICL_regression03 Vast environment. Generated by vast_regression03.sh setup.\n'
    printf 'export REG03_REPO=%q\n' "$REG03_REPO"
    printf 'export REG03_BRANCH=%q\n' "$REG03_BRANCH"
    printf 'export REG03_WORKDIR=%q\n' "$REG03_WORKDIR"
    printf 'export REG03_ENV=%q\n' "$REG03_ENV"
    printf 'export REG03_CONDA_DIR=%q\n' "$REG03_CONDA_DIR"
    printf 'export REG03_REMOTE=%q\n' "$REG03_REMOTE"
    printf 'export REG03_SYNC_INTERVAL=%q\n' "$REG03_SYNC_INTERVAL"
    printf 'export REG03_TMUX_SESSION=%q\n' "$REG03_TMUX_SESSION"
    printf 'export REG03_DEVICE=%q\n' "$REG03_DEVICE"
    printf 'export REG03_TABICL_REPO=%q\n' "$REG03_TABICL_REPO"
    printf 'export REG03_TABICL_REPO_URL=%q\n' "$REG03_TABICL_REPO_URL"
    printf 'export REG03_TABICL_BRANCH=%q\n' "$REG03_TABICL_BRANCH"
    printf 'export REG03_PRIOR_DIR=%q\n' "$REG03_PRIOR_DIR"
    printf 'export REG03_PRIOR_ARCHIVE_REMOTE=%q\n' "$REG03_PRIOR_ARCHIVE_REMOTE"
    printf 'export REG03_PRIOR_TMP_DIR=%q\n' "$REG03_PRIOR_TMP_DIR"
    printf 'export REG03_PRIOR_EXPECTED_BATCHES=%q\n' "$REG03_PRIOR_EXPECTED_BATCHES"
    printf 'export REG03_PRIOR_OVERWRITE_TMP=%q\n' "$REG03_PRIOR_OVERWRITE_TMP"
    printf 'export REG03_CONDA_FORGE_CHANNEL=%q\n' "$REG03_CONDA_FORGE_CHANNEL"
    printf 'export REG03_CUDA_CHANNEL=%q\n' "$REG03_CUDA_CHANNEL"
    printf 'export REG03_WANDB_DIR=%q\n' "$REG03_WANDB_DIR"
    if [[ -n "$REG03_CHECKPOINT_DIR" ]]; then
      printf 'export REG03_CHECKPOINT_DIR=%q\n' "$REG03_CHECKPOINT_DIR"
    fi
    if [[ -n "$REG03_AUTO_RESUME_LATEST" ]]; then
      printf 'export REG03_AUTO_RESUME_LATEST=%q\n' "$REG03_AUTO_RESUME_LATEST"
    fi
    if [[ -n "$REG03_WANDB_API_KEY" && "$REG03_WANDB_API_KEY" != \<*\> ]]; then
      printf 'export REG03_WANDB_API_KEY=%q\n' "$REG03_WANDB_API_KEY"
    fi
    if [[ -n "${REG03_WANDB_MODE:-}" ]]; then
      printf 'export REG03_WANDB_MODE=%q\n' "$REG03_WANDB_MODE"
    fi
  } >"$target"
  chmod 600 "$target"
  log "Saved regression03 Vast environment to $target."
}

source_conda() {
  if have conda; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
  elif [[ -f "$REG03_CONDA_DIR/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$REG03_CONDA_DIR/etc/profile.d/conda.sh"
  else
    die "conda is not available. Run install-env or setup first."
  fi
}

conda_env_exists() {
  source_conda
  conda env list | awk '{print $1}' | grep -qx "$REG03_ENV"
}

run_in_env() {
  source_conda
  export_runtime_env
  cd "$REG03_WORKDIR"
  conda run --no-capture-output -n "$REG03_ENV" "$@"
}

install_system_tools() {
  log "Installing system tools."
  if have apt-get; then
    as_root apt-get update
    as_root apt-get install -y git curl ca-certificates tmux rclone bzip2
    as_root apt-get install -y libarchive-tools
  else
    for tool in git curl tmux rclone bsdtar; do
      have "$tool" || die "Missing $tool and apt-get is unavailable."
    done
  fi
}

install_miniconda_if_needed() {
  if have conda || [[ -f "$REG03_CONDA_DIR/etc/profile.d/conda.sh" ]]; then
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
  log "Installing Miniconda to $REG03_CONDA_DIR."
  curl -fsSL "https://repo.anaconda.com/miniconda/$installer" -o "$tmp_dir/miniconda.sh"
  bash "$tmp_dir/miniconda.sh" -b -p "$REG03_CONDA_DIR"
  rm -rf "$tmp_dir"
}

pull_code() {
  log "Preparing sparse checkout at $REG03_WORKDIR."
  if [[ ! -d "$REG03_WORKDIR/.git" ]]; then
    mkdir -p "$(dirname "$REG03_WORKDIR")"
    git clone --filter=blob:none --no-checkout --branch "$REG03_BRANCH" \
      "$REG03_REPO" "$REG03_WORKDIR"
    git -C "$REG03_WORKDIR" sparse-checkout init --cone
    git -C "$REG03_WORKDIR" sparse-checkout set TabICL_regression03
    git -C "$REG03_WORKDIR" checkout "$REG03_BRANCH"
  else
    git -C "$REG03_WORKDIR" fetch origin "$REG03_BRANCH"
    git -C "$REG03_WORKDIR" checkout "$REG03_BRANCH"
    git -C "$REG03_WORKDIR" sparse-checkout init --cone || true
    git -C "$REG03_WORKDIR" sparse-checkout set TabICL_regression03
    git -C "$REG03_WORKDIR" pull --ff-only origin "$REG03_BRANCH"
  fi

  [[ -d "$REG03_ROOT" ]] || die "Sparse checkout did not create $REG03_ROOT"
  ensure_reg03_dirs
}

pull_tabicl() {
  log "Preparing TabICL source repo at $REG03_TABICL_REPO."
  if [[ -d "$REG03_TABICL_REPO/.git" ]]; then
    git -C "$REG03_TABICL_REPO" fetch origin "$REG03_TABICL_BRANCH"
    git -C "$REG03_TABICL_REPO" checkout "$REG03_TABICL_BRANCH"
    git -C "$REG03_TABICL_REPO" pull --ff-only origin "$REG03_TABICL_BRANCH"
  elif [[ -d "$REG03_TABICL_REPO/src" ]]; then
    log "TabICL source directory already exists: $REG03_TABICL_REPO/src"
  else
    mkdir -p "$(dirname "$REG03_TABICL_REPO")"
    git clone --branch "$REG03_TABICL_BRANCH" \
      "$REG03_TABICL_REPO_URL" "$REG03_TABICL_REPO"
  fi

  [[ -d "$REG03_TABICL_REPO/src" ]] || die \
    "TabICL source repo missing at $REG03_TABICL_REPO/src."
}

install_env() {
  [[ -f "$REG03_ROOT/requirements.txt" ]] || pull_code
  install_miniconda_if_needed
  source_conda

  if conda_env_exists; then
    log "Conda env $REG03_ENV already exists."
  else
    log "Creating conda env $REG03_ENV with Python 3.11."
    conda create -n "$REG03_ENV" -y --override-channels \
      -c "$REG03_CONDA_FORGE_CHANNEL" python=3.11 pip
  fi

  log "Ensuring pip is installed in $REG03_ENV."
  conda install -n "$REG03_ENV" -y --override-channels \
    -c "$REG03_CONDA_FORGE_CHANNEL" pip

  log "Installing CUDA compiler tools into $REG03_ENV."
  conda install -n "$REG03_ENV" -y --override-channels \
    -c "$REG03_CUDA_CHANNEL" -c "$REG03_CONDA_FORGE_CHANNEL" cuda-nvcc=12.9.86

  log "Installing regression03 Python requirements."
  conda run --no-capture-output -n "$REG03_ENV" python -m pip install --upgrade pip
  conda run --no-capture-output -n "$REG03_ENV" python -m pip install \
    -i "${REG03_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    --trusted-host "${REG03_PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}" \
    -r "$REG03_ROOT/requirements.txt"
}

configure_wandb() {
  export_runtime_env
  case "${REG03_WANDB_MODE:-}" in
    disabled|offline)
      log "W&B mode is ${REG03_WANDB_MODE}; skipping online login."
      return
      ;;
  esac

  if [[ -z "$REG03_WANDB_API_KEY" || "$REG03_WANDB_API_KEY" == \<*\> ]]; then
    log "W&B API key not set; set REG03_WANDB_API_KEY or WANDB_API_KEY for online logging."
    return
  fi

  conda_env_exists || die "Conda env $REG03_ENV does not exist. Run install-env first."
  log "Configuring W&B login from REG03_WANDB_API_KEY/WANDB_API_KEY."
  conda run --no-capture-output -n "$REG03_ENV" python - <<'PY'
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
  [[ -n "$REG03_REMOTE" ]] || die "REG03_REMOTE is required for sync commands."
  have rclone || die "rclone is not installed. Run setup or install rclone first."
}

remote_path() {
  printf '%s/%s' "$REG03_REMOTE" "$1"
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
  ensure_reg03_dirs
  copy_remote_dir_if_present "runs" "$REG03_ROOT/runs" --exclude "prior/**"
  copy_remote_dir_if_present "sklearn_data" "$REG03_SKLEARN_DATA"
  copy_remote_dir_if_present "tabarena_cache" "$REG03_TABARENA_CACHE"
  copy_remote_dir_if_present "wandb" "$REG03_WANDB_DIR"
}

sync_push() {
  require_remote
  ensure_reg03_dirs
  copy_local_dir "runs" "$REG03_ROOT/runs" --exclude "prior/**"
  copy_local_dir "sklearn_data" "$REG03_SKLEARN_DATA"
  copy_local_dir "tabarena_cache" "$REG03_TABARENA_CACHE"
  copy_local_dir "wandb" "$REG03_WANDB_DIR"
}

require_prior_archive_remote() {
  [[ -n "$REG03_PRIOR_ARCHIVE_REMOTE" ]] || die \
    "REG03_PRIOR_ARCHIVE_REMOTE is required. Set REG03_REMOTE or REG03_PRIOR_ARCHIVE_REMOTE."
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
  if [[ -n "$REG03_PRIOR_ARCHIVE_REMOTE" ]] && have rclone; then
    rclone lsl "$REG03_PRIOR_ARCHIVE_REMOTE" || true
    rclone md5sum "$REG03_PRIOR_ARCHIVE_REMOTE" || true
  elif [[ -z "$REG03_PRIOR_ARCHIVE_REMOTE" ]]; then
    log "WARNING: REG03_PRIOR_ARCHIVE_REMOTE is not set."
  else
    log "WARNING: rclone is not installed."
  fi

  log "Prior: disk"
  df_for_path "$REG03_PRIOR_DIR"

  log "Prior: final directory"
  du_if_present "$REG03_PRIOR_DIR"
  printf 'batch count: %s\n' "$(batch_count "$REG03_PRIOR_DIR")"
  sample_batches "$REG03_PRIOR_DIR"

  log "Prior: temp directory"
  du_if_present "$REG03_PRIOR_TMP_DIR"
  printf 'batch count: %s\n' "$(batch_count "$REG03_PRIOR_TMP_DIR")"
  sample_batches "$REG03_PRIOR_TMP_DIR"
}

prior_extract() {
  require_prior_archive_remote
  have bsdtar || die "bsdtar is missing. Run setup to install libarchive-tools."

  [[ ! -e "$REG03_PRIOR_DIR" ]] || die \
    "Final prior path already exists: $REG03_PRIOR_DIR. Refusing to create a second copy."

  if [[ -e "$REG03_PRIOR_TMP_DIR" ]]; then
    if [[ "$REG03_PRIOR_OVERWRITE_TMP" != "1" ]]; then
      die "Temp prior path already exists: $REG03_PRIOR_TMP_DIR. Set REG03_PRIOR_OVERWRITE_TMP=1 to replace it."
    fi
    [[ -n "$REG03_PRIOR_TMP_DIR" && "$REG03_PRIOR_TMP_DIR" != "/" && "$REG03_PRIOR_TMP_DIR" == *.tmp ]] || \
      die "Refusing to remove unsafe temp path: $REG03_PRIOR_TMP_DIR"
    log "Removing existing temp prior path because REG03_PRIOR_OVERWRITE_TMP=1: $REG03_PRIOR_TMP_DIR"
    rm -rf -- "$REG03_PRIOR_TMP_DIR"
  fi

  mkdir -p "$REG03_PRIOR_TMP_DIR"
  log "Streaming prior archive without storing the ZIP locally:"
  log "  $REG03_PRIOR_ARCHIVE_REMOTE -> $REG03_PRIOR_TMP_DIR"
  rclone cat "$REG03_PRIOR_ARCHIVE_REMOTE" | bsdtar -xf - -C "$REG03_PRIOR_TMP_DIR"

  local count
  count="$(batch_count "$REG03_PRIOR_TMP_DIR")"
  log "Extracted $count batch files into temp prior path."
  if [[ "$REG03_PRIOR_EXPECTED_BATCHES" != "0" && "$count" != "$REG03_PRIOR_EXPECTED_BATCHES" ]]; then
    die "Expected $REG03_PRIOR_EXPECTED_BATCHES batch files, found $count. Data remains in $REG03_PRIOR_TMP_DIR for inspection."
  fi
  log "Prior extraction complete. Inspect $REG03_PRIOR_TMP_DIR, then run prior-activate when ready."
}

prior_activate() {
  [[ -d "$REG03_PRIOR_TMP_DIR" ]] || die "Temp prior path missing: $REG03_PRIOR_TMP_DIR"
  [[ ! -e "$REG03_PRIOR_DIR" ]] || die "Final prior path already exists: $REG03_PRIOR_DIR"

  local count
  count="$(batch_count "$REG03_PRIOR_TMP_DIR")"
  if [[ "$REG03_PRIOR_EXPECTED_BATCHES" != "0" && "$count" != "$REG03_PRIOR_EXPECTED_BATCHES" ]]; then
    die "Expected $REG03_PRIOR_EXPECTED_BATCHES batch files in $REG03_PRIOR_TMP_DIR, found $count."
  fi

  mkdir -p "$(dirname "$REG03_PRIOR_DIR")"
  mv "$REG03_PRIOR_TMP_DIR" "$REG03_PRIOR_DIR"
  log "Activated prior data: $REG03_PRIOR_DIR"
  prior_status
}

sync_loop() {
  require_remote
  mkdir -p "$REG03_LOG_DIR"
  log "Starting sync loop every $REG03_SYNC_INTERVAL seconds."
  while true; do
    sync_push 2>&1 | tee -a "$REG03_LOG_DIR/sync-loop.log" || true
    sleep "$REG03_SYNC_INTERVAL"
  done
}

latest_checkpoint() {
  find "$REG03_ROOT/runs" -type f -name 'step-*.pt' 2>/dev/null | sort -V | tail -n 1
}

latest_eval_json() {
  find "$REG03_ROOT/runs" -type f -name '*.json' \
    ! -name '*uncertainty*.json' 2>/dev/null | sort -V | tail -n 1
}

latest_qice_json() {
  find "$REG03_ROOT/runs" -type f -name '*uncertainty*.json' 2>/dev/null | sort -V | tail -n 1
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

training_tabicl_repo() {
  printf '%s' "$REG03_TABICL_REPO"
}

training_prior_dir() {
  printf '%s' "$REG03_PRIOR_DIR"
}

check_training_inputs() {
  local tabicl_repo prior_dir count
  tabicl_repo="$(training_tabicl_repo)"
  prior_dir="$(training_prior_dir)"

  [[ -d "$tabicl_repo/src" ]] || die \
    "TabICL source repo missing at $tabicl_repo/src. Set REG03_TABICL_REPO."
  [[ -d "$prior_dir" ]] || die \
    "Pre-generated prior directory missing at $prior_dir. Set REG03_PRIOR_DIR."
  compgen -G "$prior_dir/batch_*.pt" >/dev/null || die \
    "No batch_*.pt files found in $prior_dir."
  count="$(batch_count "$prior_dir")"
  if [[ "$REG03_PRIOR_EXPECTED_BATCHES" != "0" && "$count" != "$REG03_PRIOR_EXPECTED_BATCHES" ]]; then
    die "Expected $REG03_PRIOR_EXPECTED_BATCHES batch_*.pt files in $prior_dir, found $count."
  fi
}

warn_training_inputs() {
  local tabicl_repo prior_dir count ok=1
  tabicl_repo="$(training_tabicl_repo)"
  prior_dir="$(training_prior_dir)"

  [[ -d "$tabicl_repo/src" ]] || {
    log "WARNING: TabICL source repo missing at $tabicl_repo/src."
    ok=0
  }
  [[ -d "$prior_dir" ]] || {
    log "WARNING: prior directory missing at $prior_dir."
    ok=0
  }
  if [[ -d "$prior_dir" ]] && ! compgen -G "$prior_dir/batch_*.pt" >/dev/null; then
    log "WARNING: no batch_*.pt files found in $prior_dir."
    ok=0
  fi
  if [[ -d "$prior_dir" ]] && compgen -G "$prior_dir/batch_*.pt" >/dev/null; then
    count="$(batch_count "$prior_dir")"
    if [[ "$REG03_PRIOR_EXPECTED_BATCHES" != "0" && "$count" != "$REG03_PRIOR_EXPECTED_BATCHES" ]]; then
      log "WARNING: expected $REG03_PRIOR_EXPECTED_BATCHES batch_*.pt files in $prior_dir, found $count."
      ok=0
    fi
  fi
  if [[ "$ok" -eq 0 ]]; then
    log "Training will fail until REG03_TABICL_REPO and REG03_PRIOR_DIR point to valid paths."
  fi
}

doctor() {
  log "Doctor: repository"
  if [[ -d "$REG03_WORKDIR/.git" ]]; then
    git -C "$REG03_WORKDIR" status --short --branch || true
    git -C "$REG03_WORKDIR" sparse-checkout list || true
  else
    log "Repository missing at $REG03_WORKDIR."
  fi

  log "Doctor: GPU"
  if have nvidia-smi; then
    nvidia-smi
  else
    log "WARNING: nvidia-smi not found."
  fi

  log "Doctor: conda/PyTorch"
  if have conda || [[ -f "$REG03_CONDA_DIR/etc/profile.d/conda.sh" ]]; then
    source_conda
    if conda_env_exists; then
      export_runtime_env
      conda run --no-capture-output -n "$REG03_ENV" python - <<'PY' || true
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
    else
      log "WARNING: conda env $REG03_ENV does not exist."
    fi
  else
    log "WARNING: conda not installed."
  fi

  log "Doctor: rclone"
  if [[ -n "$REG03_REMOTE" ]] && have rclone; then
    rclone lsd "$REG03_REMOTE" || true
  elif [[ -z "$REG03_REMOTE" ]]; then
    log "WARNING: REG03_REMOTE is not set."
  else
    log "WARNING: rclone is not installed."
  fi

  log "Doctor: regression03 directories"
  du_if_present "$REG03_ROOT/runs"
  du_if_present "$REG03_SKLEARN_DATA"
  du_if_present "$REG03_TABARENA_CACHE"
  du_if_present "$REG03_WANDB_DIR"

  log "Doctor: training inputs"
  warn_training_inputs
}

train() {
  [[ -d "$REG03_ROOT" ]] || die "$REG03_ROOT not found. Run pull-code first."
  conda_env_exists || die "Conda env $REG03_ENV does not exist. Run install-env first."
  configure_wandb
  check_training_inputs
  export_runtime_env

  log "Starting regression03 training."
  cd "$REG03_ROOT"
  conda run --no-capture-output -n "$REG03_ENV" python - <<'PY'
import os

from tabicl_style.config import Config
from tabicl_style.train import Trainer

config = Config()
tc = config.training
tc.tabicl_repo = os.environ.get("REG03_TABICL_REPO", tc.tabicl_repo)
tc.prior_dir = os.environ.get("REG03_PRIOR_DIR", tc.prior_dir)
tc.wandb_dir = os.environ.get("REG03_WANDB_DIR", tc.wandb_dir)
if os.environ.get("REG03_CHECKPOINT_DIR"):
    tc.checkpoint_dir = os.environ["REG03_CHECKPOINT_DIR"]
if os.environ.get("REG03_AUTO_RESUME_LATEST"):
    value = os.environ["REG03_AUTO_RESUME_LATEST"].strip().lower()
    if value in {"1", "true", "yes", "on"}:
        tc.auto_resume_latest = True
    elif value in {"0", "false", "no", "off"}:
        tc.auto_resume_latest = False
    else:
        raise ValueError(
            "REG03_AUTO_RESUME_LATEST must be one of "
            "1/0, true/false, yes/no, on/off."
        )
if os.environ.get("REG03_WANDB_MODE"):
    tc.wandb_mode = os.environ["REG03_WANDB_MODE"]

trainer = Trainer(config)
trainer.train()
PY
}

train_tmux() {
  have tmux || die "tmux is not installed. Run setup first."
  if tmux has-session -t "$REG03_TMUX_SESSION" 2>/dev/null; then
    log "tmux session already exists: $REG03_TMUX_SESSION"
    log "Attach with: tmux attach -t $REG03_TMUX_SESSION"
    return
  fi
  tmux new-session -d -s "$REG03_TMUX_SESSION" \
    "cd '$REG03_WORKDIR' && bash '$SCRIPT_PATH' train"
  log "Started training tmux session: $REG03_TMUX_SESSION"
  log "Attach with: tmux attach -t $REG03_TMUX_SESSION"
}

eval_latest() {
  conda_env_exists || die "Conda env $REG03_ENV does not exist. Run install-env first."
  local checkpoint method steps repeats output
  checkpoint="${REG03_EVAL_CHECKPOINT:-$(latest_checkpoint)}"
  [[ -n "$checkpoint" ]] || die "No step-*.pt checkpoint found under $REG03_ROOT/runs."

  method="${REG03_EVAL_METHOD:-ddpm}"
  steps="${REG03_EVAL_STEPS:-500}"
  repeats="${REG03_EVAL_REPEATS:-100}"
  output="${REG03_EVAL_OUTPUT:-${checkpoint%.pt}_${method}${steps}_repeats${repeats}.json}"

  local cmd=(
    python TabICL_regression03/tabicl_style/evaluation.py
    --checkpoint "$checkpoint"
    --device "$REG03_DEVICE"
    --amp
    --amp-dtype auto
    --num-sampling-steps "$steps"
    --sampling-method "$method"
    --ddim-eta "${REG03_EVAL_DDIM_ETA:-0.0}"
    --max-features-eval "${REG03_EVAL_MAX_FEATURES:-32}"
    --max-rows-eval "${REG03_EVAL_MAX_ROWS:-0}"
    --new-instances-eval "${REG03_EVAL_NEW_INSTANCES:-200}"
    --n-splits "${REG03_EVAL_SPLITS:-5}"
    --n-repeats "$repeats"
    --random-state "${REG03_EVAL_RANDOM_STATE:-0}"
    --output-json "$output"
  )
  if [[ -n "${REG03_EVAL_DATASETS:-}" ]]; then
    cmd+=(--datasets "$REG03_EVAL_DATASETS")
  fi

  log "Evaluating checkpoint: $checkpoint"
  run_in_env "${cmd[@]}"
}

qice_latest() {
  conda_env_exists || die "Conda env $REG03_ENV does not exist. Run install-env first."
  local input output
  input="${REG03_EVAL_JSON:-$(latest_eval_json)}"
  [[ -n "$input" ]] || die "No evaluation JSON found under $REG03_ROOT/runs."
  output="${REG03_QICE_OUTPUT:-${input%.json}_uncertainty.json}"

  log "Computing QICE for: $input"
  run_in_env python TabICL_regression03/tabicl_style/evaluation_uncertainty.py \
    --input-json "$input" \
    --num-bins "${REG03_QICE_BINS:-10}" \
    --output-json "$output"
}

status() {
  log "Status"
  if [[ -d "$REG03_WORKDIR/.git" ]]; then
    git -C "$REG03_WORKDIR" status --short --branch || true
    git -C "$REG03_WORKDIR" log -1 --oneline --decorate || true
  fi
  printf 'Latest checkpoint: %s\n' "$(latest_checkpoint)"
  printf 'Latest eval JSON: %s\n' "$(latest_eval_json)"
  printf 'Latest QICE JSON: %s\n' "$(latest_qice_json)"
  du_if_present "$REG03_ROOT/runs"
  du_if_present "$REG03_SKLEARN_DATA"
  du_if_present "$REG03_TABARENA_CACHE"
  du_if_present "$REG03_WANDB_DIR"
  if [[ -f "$REG03_LOG_DIR/sync-loop.log" ]]; then
    log "Recent sync-loop log"
    tail -n 20 "$REG03_LOG_DIR/sync-loop.log"
  fi
}

open_shell() {
  source_conda
  export_runtime_env
  cd "$REG03_WORKDIR"
  conda activate "$REG03_ENV"
  log "Activated $REG03_ENV in $REG03_WORKDIR."
  exec bash -i
}

menu() {
  while true; do
    cat <<'EOF'

Regression03 Vast menu
  1) status
  2) doctor
  3) sync-pull
  4) sync-push
  5) prior-status
  6) prior-extract
  7) prior-activate
  8) train-tmux
  9) eval-latest
  10) qice-latest
  11) sync-loop
  12) shell
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
      8) train_tmux ;;
      9) eval_latest ;;
      10) qice_latest ;;
      11) sync_loop ;;
      12) open_shell ;;
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
  pull_tabicl
  install_env
  configure_wandb
  sync_pull
  doctor
  log "Setup complete. Run: bash TabICL_regression03/scripts/vast_regression03.sh menu"
}

main() {
  local command="${1:-help}"
  case "$command" in
    setup) setup ;;
    pull-code) pull_code ;;
    pull-tabicl) pull_tabicl ;;
    install-env) install_env ;;
    sync-pull) sync_pull ;;
    sync-push) sync_push ;;
    sync-loop) sync_loop ;;
    prior-status) prior_status ;;
    prior-extract) prior_extract ;;
    prior-activate) prior_activate ;;
    doctor) doctor ;;
    train) train ;;
    train-tmux) train_tmux ;;
    eval-latest) eval_latest ;;
    qice-latest) qice_latest ;;
    status) status ;;
    shell) open_shell ;;
    menu) menu ;;
    help|-h|--help) usage ;;
    *) usage; die "Unknown command: $command" ;;
  esac
}

main "$@"
