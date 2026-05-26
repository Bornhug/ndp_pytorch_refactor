# TabICL_regression03 Vast.ai Quick Start

Use this guide for disposable Vast.ai CUDA Ubuntu instances with small root
disks. The workflow is:

1. Keep durable state in Google Drive.
2. Bootstrap the Vast machine from GitHub.
3. Stream the large prior ZIP from Google Drive directly into extracted
   `batch_*.pt` files.
4. Inspect the extracted data.
5. Start training only after you explicitly approve it.

The script does not start training during `setup` or `prior-extract`.

## 1. Prepare Google Drive Once

Create an `rclone` Google Drive remote locally. In WSL, run:

```bash
sudo apt-get update
sudo apt-get install -y rclone
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

Create and test the project folder:

```bash
rclone mkdir gdrive:ndp-regression03
rclone lsd gdrive:ndp-regression03
```

Use this Google Drive layout:

```text
gdrive:ndp-regression03/
  runs/
    runs05/
      step-42000.pt
    prior/
      stage1_15000batches.zip
  sklearn_data/
  tabarena_cache/
  wandb/
```

Put previous checkpoints under `runs/`. Put the large prior archive at:

```text
gdrive:ndp-regression03/runs/prior/stage1_15000batches.zip
```

The normal sync commands exclude `runs/prior/**`, so this large archive is not
accidentally downloaded during setup. The archive is handled by the explicit
`prior-extract` command instead.

## 2. Copy rclone Config to Vast

The Vast machine needs the same rclone credential file that works in WSL. This
file is private. Do not commit it to git.

On WSL, show the config:

```bash
cat ~/.config/rclone/rclone.conf
```

You should see a section like:

```ini
[gdrive]
type = drive
token = ...
```

SSH into the Vast machine and paste that config:

```bash
mkdir -p ~/.config/rclone
nano ~/.config/rclone/rclone.conf
```

Save in `nano` with:

```text
Ctrl+O
Enter
Ctrl+X
```

Test on Vast:

```bash
rclone lsd gdrive:
rclone lsd gdrive:ndp-regression03
```

If those commands list folders, Google Drive access is ready.

## 3. Bootstrap the Vast Machine

SSH into the Vast instance. Vast usually gives a command like:

```bash
ssh -p <port> root@<host> -L 8080:localhost:8080
```

On Vast, set the reusable environment variables:

```bash
export REG03_REMOTE="gdrive:ndp-regression03"
export REG03_TABICL_REPO="/workspace/tabicl"
export REG03_TABICL_REPO_URL="https://github.com/soda-inria/tabicl.git"
export REG03_TABICL_BRANCH="main"
export REG03_PRIOR_DIR="/workspace/tabicl/data_regression/stage1"
export REG03_PRIOR_ARCHIVE_REMOTE="$REG03_REMOTE/runs/prior/stage1_15000batches.zip"
export REG03_PRIOR_TMP_DIR="$REG03_PRIOR_DIR.tmp"
export REG03_PRIOR_EXPECTED_BATCHES=15000
export REG03_SYNC_INTERVAL=1800
# Optional:
# export REG03_WANDB_API_KEY="<your_wandb_api_key>"
# export REG03_WANDB_MODE="disabled"
```

Run setup:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/Bornhug/ndp_pytorch_refactor/branch01/TabICL_regression03/scripts/vast_regression03.sh \
  -o /tmp/vast_regression03.sh && \
bash /tmp/vast_regression03.sh setup
```

`setup` does these things only:

- saves the `REG03_*` values to `~/.reg03_vast_env`
- installs system tools, including `rclone`, `tmux`, and `bsdtar`
- sparse-checks out only `TabICL_regression03` from GitHub
- clones or updates the external TabICL repo at `/workspace/tabicl`
- creates or updates the `ndp_sim` conda environment
- syncs previous checkpoints, caches, and W&B files from Google Drive
- runs `doctor`

It does not download the prior ZIP, extract the prior dataset, or start
training.

## 4. Stream the Prior ZIP Without Storing It

The Vast disk is too small to hold both the 12 GB ZIP and the extracted prior
dataset at the same time. Do not run `rclone copy` for the ZIP.

Instead, stream it:

```bash
cd /workspace/ndp_pytorch_refactor
bash TabICL_regression03/scripts/vast_regression03.sh prior-extract
```

Internally this runs the equivalent of:

```bash
rclone cat gdrive:ndp-regression03/runs/prior/stage1_15000batches.zip \
  | bsdtar -xf - -C /workspace/tabicl/data_regression/stage1.tmp
```

That means:

- the ZIP stays on Google Drive
- no local ZIP copy is written to the Vast disk
- extracted batches are written to `REG03_PRIOR_TMP_DIR`
- the active `REG03_PRIOR_DIR` is not created yet
- training is not started

Check progress or final state:

```bash
bash TabICL_regression03/scripts/vast_regression03.sh prior-status
```

Expected final status:

```text
batch count: 15000
temp directory: /workspace/tabicl/data_regression/stage1.tmp
final directory: missing /workspace/tabicl/data_regression/stage1
```

If a previous temp extraction exists and you intentionally want to replace it:

```bash
export REG03_PRIOR_OVERWRITE_TMP=1
bash TabICL_regression03/scripts/vast_regression03.sh prior-extract
```

## 5. Inspect, Then Activate the Prior Data

Inspect the extracted temp dataset first:

```bash
ls -lh /workspace/tabicl/data_regression/stage1.tmp | head
ls -lh /workspace/tabicl/data_regression/stage1.tmp | tail
df -h /workspace
bash TabICL_regression03/scripts/vast_regression03.sh prior-status
```

After inspection, activate it:

```bash
bash TabICL_regression03/scripts/vast_regression03.sh prior-activate
```

`prior-activate` verifies the expected batch count and then moves:

```text
/workspace/tabicl/data_regression/stage1.tmp
```

to:

```text
/workspace/tabicl/data_regression/stage1
```

Training uses `REG03_PRIOR_DIR`, so training cannot start successfully until
this activation step is done.

## 6. Check Before Training

Run:

```bash
bash TabICL_regression03/scripts/vast_regression03.sh doctor
bash TabICL_regression03/scripts/vast_regression03.sh status
```

`doctor` should show:

- PyTorch CUDA is available
- `rclone` can list `gdrive:ndp-regression03`
- the external TabICL repo has `src/`
- `REG03_PRIOR_DIR` exists
- `REG03_PRIOR_DIR` contains exactly `15000` `batch_*.pt` files

If `doctor` warns about the prior directory, do not train yet.

## 7. Start Training Only When Ready

Start training in tmux:

```bash
bash TabICL_regression03/scripts/vast_regression03.sh train-tmux
```

Start periodic checkpoint/cache upload in a separate tmux session:

```bash
tmux new-session -d -s reg03-sync \
  "bash /workspace/ndp_pytorch_refactor/TabICL_regression03/scripts/vast_regression03.sh sync-loop"
```

Attach to training:

```bash
tmux attach -t reg03-train
```

Detach without stopping training:

```text
Ctrl+B
D
```

Before stopping the Vast instance, push state back to Google Drive:

```bash
bash TabICL_regression03/scripts/vast_regression03.sh sync-push
```

## 8. Common Commands

```bash
# Show help.
bash TabICL_regression03/scripts/vast_regression03.sh help

# Re-run setup checks.
bash TabICL_regression03/scripts/vast_regression03.sh doctor

# Show checkpoints, cache sizes, sync logs, and latest outputs.
bash TabICL_regression03/scripts/vast_regression03.sh status

# Show prior archive, disk space, counts, and sample files.
bash TabICL_regression03/scripts/vast_regression03.sh prior-status

# Stream-extract prior ZIP into stage1.tmp only.
bash TabICL_regression03/scripts/vast_regression03.sh prior-extract

# Move inspected stage1.tmp to active stage1.
bash TabICL_regression03/scripts/vast_regression03.sh prior-activate

# Start training only after prior activation.
bash TabICL_regression03/scripts/vast_regression03.sh train-tmux

# Push runs/caches/wandb to Google Drive.
bash TabICL_regression03/scripts/vast_regression03.sh sync-push
```
