# TabICL_regression03 Vast.ai Quick Start

Use this on disposable Vast.ai CUDA Ubuntu instances. Do not run the local/WSL
environment setup in `command.md`; the Vast shell script handles setup.

## 1. Start Setup

Get the SSH command from the Vast.ai instance page. It usually looks like:

```bash
ssh root@<host> -p <port>
```

If Vast gives you a private key, save it locally and restrict permissions:

```bash
chmod 600 ~/.ssh/<vast_key>
ssh -i ~/.ssh/<vast_key> root@<host> -p <port>
```

Optional local SSH alias:

```sshconfig
Host vast-reg03
  HostName <host>
  Port <port>
  User root
  IdentityFile ~/.ssh/<vast_key>
  ServerAliveInterval 30
  ServerAliveCountMax 4
```

Then connect with:

```bash
ssh vast-reg03
```

After SSH connects, run this block on the Vast instance. Replace the three
exported values with your actual storage and prior-data locations.

```bash
export REG03_REMOTE="REG03:ndp-regression03"
export REG03_TABICL_REPO="/workspace/tabicl"
export REG03_PRIOR_DIR="/workspace/tabicl/data_regression/stage1"

mkdir -p ~/.config/rclone
nano ~/.config/rclone/rclone.conf

curl -fsSL \
  https://raw.githubusercontent.com/Bornhug/ndp_pytorch_refactor/branch01/TabICL_regression03/scripts/vast_regression03.sh \
  -o /tmp/vast_regression03.sh && \
bash /tmp/vast_regression03.sh setup
```

## 2. Why These Values Are Not In The Shell

The shell owns deterministic defaults:

```bash
REG03_REPO="https://github.com/Bornhug/ndp_pytorch_refactor.git"
REG03_BRANCH="branch01"
REG03_WORKDIR="/workspace/ndp_pytorch_refactor"
REG03_ENV="ndp_sim"
```

These values stay in this guide because they are user- or instance-specific:

- `REG03_REMOTE`: your `rclone` remote/path for durable checkpoints and caches.
- `REG03_TABICL_REPO`: your external TabICL source repo path.
- `REG03_PRIOR_DIR`: your pre-generated `batch_*.pt` prior data path.
- `~/.config/rclone/rclone.conf`: your private cloud credentials/config.

The setup command installs tools, sparse-checks out only `TabICL_regression03`,
installs the conda environment, pulls saved state from `REG03_REMOTE`, and runs
`doctor`.

## 3. Run Training, Sync, Evaluation

Start training and periodic cloud upload:

```bash
bash TabICL_regression03/scripts/vast_regression03.sh train-tmux
tmux new -s reg03-sync "bash TabICL_regression03/scripts/vast_regression03.sh sync-loop"
```

Check the instance:

```bash
bash TabICL_regression03/scripts/vast_regression03.sh doctor
bash TabICL_regression03/scripts/vast_regression03.sh status
```

Evaluate latest checkpoint and compute QICE:

```bash
bash TabICL_regression03/scripts/vast_regression03.sh eval-latest
bash TabICL_regression03/scripts/vast_regression03.sh qice-latest
```

Before stopping the instance:

```bash
bash TabICL_regression03/scripts/vast_regression03.sh sync-push
```

The shell copies, rather than deletes/syncs, these paths:

```text
$REG03_REMOTE/runs           <-> TabICL_regression03/runs
$REG03_REMOTE/sklearn_data   <-> TabICL_regression03/.sklearn_data
$REG03_REMOTE/tabarena_cache <-> TabICL_regression03/tabicl_style/.tabarena_cache
$REG03_REMOTE/wandb          <-> TabICL_regression03/wandb
```
