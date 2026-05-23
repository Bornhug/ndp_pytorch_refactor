# TabICL_regression03 Vast.ai Quick Start

Use this on disposable Vast.ai CUDA Ubuntu instances. Do not run the local/WSL
environment setup in `command.md`; the Vast shell script handles setup.

## 1. Prepare Cloud Storage Once

Create durable cloud storage before renting the GPU. Since you have Google
Drive storage, use an `rclone` Google Drive remote.

You need:

- a Google Drive folder for regression03 state
- an `rclone` Google Drive remote name, for example `gdrive`
- a base path, for example `ndp-regression03`

The guide below assumes this remote path:

```bash
gdrive:ndp-regression03
```

On your local computer, install `rclone` if needed:

```bash
# macOS
brew install rclone

# Ubuntu / WSL
sudo apt-get update
sudo apt-get install -y rclone
```

Create the Google Drive remote:

```bash
rclone config
```

Use these choices in the interactive prompt:

```text
n) New remote
name> gdrive
Storage> drive
client_id> press Enter
client_secret> press Enter
scope> 1
root_folder_id> press Enter
service_account_file> press Enter
Edit advanced config? n
Use auto config? y
Configure this as a Shared Drive? n
Keep this "gdrive" remote? y
q) Quit config
```

Create the regression03 folder and test access:

```bash
rclone mkdir gdrive:ndp-regression03
rclone lsd gdrive:ndp-regression03
```

During `rclone config`, create a Google Drive remote named `gdrive`. The Vast
instance then needs the same rclone config at:

```bash
~/.config/rclone/rclone.conf
```

That config contains your private cloud credentials, so do not commit it to git.

To copy it to Vast manually, open it locally:

```bash
cat ~/.config/rclone/rclone.conf
```

Then paste the contents into the Vast instance when this guide tells you to run:

```bash
nano ~/.config/rclone/rclone.conf
```

## 2. Start Setup

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

After SSH connects, run this block on the Vast instance. Replace the exported
values with your actual storage, prior-data, and W&B locations/secrets.

```bash
export REG03_REMOTE="gdrive:ndp-regression03"
export REG03_TABICL_REPO="/workspace/tabicl"
export REG03_PRIOR_DIR="/workspace/tabicl/data_regression/stage1"
export REG03_SYNC_INTERVAL=1800
export REG03_WANDB_API_KEY="<your_wandb_api_key>"

mkdir -p ~/.config/rclone
nano ~/.config/rclone/rclone.conf

curl -fsSL \
  https://raw.githubusercontent.com/Bornhug/ndp_pytorch_refactor/branch01/TabICL_regression03/scripts/vast_regression03.sh \
  -o /tmp/vast_regression03.sh && \
bash /tmp/vast_regression03.sh setup
```

What this block does:

- `REG03_REMOTE` tells the shell where to pull and push durable state.
- `REG03_TABICL_REPO` points training to the external TabICL source repo.
- `REG03_PRIOR_DIR` points training to pre-generated prior `batch_*.pt` files.
- `REG03_SYNC_INTERVAL=1800` makes the background upload loop run every 30 minutes.
- `REG03_WANDB_API_KEY` lets the shell configure W&B login during setup/training.
- `mkdir -p ~/.config/rclone` creates the expected rclone config directory.
- `nano ~/.config/rclone/rclone.conf` is where you paste the Google Drive rclone config.
- `curl ... -o /tmp/vast_regression03.sh` downloads the helper shell from GitHub.
- `bash /tmp/vast_regression03.sh setup` runs full machine setup.

## 3. What `vast_regression03.sh` Does

The shell is the one-go bootstrap and run helper for a disposable Vast instance.
It makes the machine look like the expected regression03 environment, restores
saved state from Google Drive, and then runs training/evaluation inside the
same conda environment.

By default it uses:

```bash
REG03_REPO="https://github.com/Bornhug/ndp_pytorch_refactor.git"
REG03_BRANCH="branch01"
REG03_WORKDIR="/workspace/ndp_pytorch_refactor"
REG03_ENV="ndp_sim"
```

It creates and reuses these local paths:

```text
/workspace/ndp_pytorch_refactor/TabICL_regression03/runs
/workspace/ndp_pytorch_refactor/TabICL_regression03/.sklearn_data
/workspace/ndp_pytorch_refactor/TabICL_regression03/tabicl_style/.tabarena_cache
/workspace/ndp_pytorch_refactor/TabICL_regression03/wandb
```

When Python commands run, the shell exports:

```bash
SCIKIT_LEARN_DATA=TabICL_regression03/.sklearn_data
TABARENA_CACHE=TabICL_regression03/tabicl_style/.tabarena_cache
WANDB_DIR=TabICL_regression03/wandb
```

`setup` is the full bootstrap. In order, it:

1. Creates `~/.config/rclone` so your pasted Google Drive config has a home.
2. Installs system tools with `apt-get`: `git`, `curl`, `tmux`, `rclone`, and
   basic certificates/tools.
3. Clones or updates `branch01` with Git sparse checkout, downloading only
   `TabICL_regression03`.
4. Installs Miniconda if needed, creates/updates the `ndp_sim` conda env,
   installs `cuda-nvcc=12.9.86`, then installs `requirements.txt`.
5. Runs `sync-pull`, copying existing `runs`, dataset caches, TabArena cache,
   and W&B logs from Google Drive onto the instance.
6. Runs `doctor` so you can see whether GPU, PyTorch CUDA, rclone, repo state,
   caches, and training input paths are valid.

The sync commands use `rclone copy`, so they upload/download files without
deleting extra files on the other side:

```text
$REG03_REMOTE/runs           <-> TabICL_regression03/runs
$REG03_REMOTE/sklearn_data   <-> TabICL_regression03/.sklearn_data
$REG03_REMOTE/tabarena_cache <-> TabICL_regression03/tabicl_style/.tabarena_cache
$REG03_REMOTE/wandb          <-> TabICL_regression03/wandb
```

`train` checks that `REG03_TABICL_REPO/src` exists and that
`REG03_PRIOR_DIR` contains `batch_*.pt` files. It then runs
`tabicl_style.train.Trainer(config).train()` inside `ndp_sim`, overriding the
training config with your Vast paths. If `REG03_WANDB_API_KEY` or
`WANDB_API_KEY` is set, the shell also logs in to W&B and exports
`WANDB_API_KEY` before training. `train-tmux` starts the same command in a tmux
session named `reg03-train`, so training keeps running after SSH disconnects.

`eval-latest` finds the newest `step-*.pt` checkpoint under `runs`, runs
`evaluation.py`, and writes a detailed evaluation JSON next to the checkpoint.
You can override its defaults with env vars like `REG03_EVAL_REPEATS`,
`REG03_EVAL_SPLITS`, `REG03_EVAL_DATASETS`, and `REG03_EVAL_OUTPUT`.

`qice-latest` finds the newest non-uncertainty evaluation JSON, runs
`evaluation_uncertainty.py`, and writes a matching `_uncertainty.json` file.

`status` prints the current branch, latest checkpoint, latest evaluation JSON,
latest QICE JSON, cache/run sizes, and the tail of the background sync log.
`shell` opens an activated `ndp_sim` shell with the regression03 cache
environment variables already exported. `menu` gives an interactive wrapper
around the same commands.

## 4. Run Training, Sync, Evaluation

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
