# nanoTabPFN_version10_original02 Vast.ai Quick Start

Use this guide for disposable Vast.ai CUDA Ubuntu instances. The workflow is:

1. Keep durable state in Google Drive.
2. Bootstrap the Vast machine from GitHub.
3. Prepare the TabPFN v2 regression border asset.
4. Stream the large TabICL prior ZIP from Google Drive into extracted `batch_*.pt` files.
5. Inspect the extracted data.
6. Start training only after you explicitly approve it.

The helper script does not start training during `setup`, `borders`,
`prior-extract`, or `prior-activate`.

Important: `train.py` currently starts a fresh run. It does not resume model
weights from existing checkpoints. The Vast script refuses to train into a
checkpoint directory that already contains `model_step_*.pt` or `model_final.pt`
unless you explicitly set `NT10_ALLOW_NONEMPTY_CHECKPOINT_DIR=1`.

## 1. Commit and Push These Files First

The bootstrap command below downloads the shell script from GitHub. Commit and
push these new files to `branch01` before using the raw GitHub URL:

```text
nanoTabPFN_version10_original02/requirements.txt
nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh
nanoTabPFN_version10_original02/vast_ai.md
```

## 2. Prepare Google Drive Once

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
rclone mkdir gdrive:ndp-nanotabpfn-v10-original02
rclone lsd gdrive:ndp-nanotabpfn-v10-original02
```

Use this Google Drive layout:

```text
gdrive:ndp-nanotabpfn-v10-original02/
  runs/
    prior/
      stage1_continuous_y_seed123.zip
    vast_l5_bars5000_seed123/
  assets/
    tabpfn_v2_regressor_borders_5000.pt
  sklearn_data/
  wandb/
```

Put the TabICL prior archive at:

```text
gdrive:ndp-nanotabpfn-v10-original02/runs/prior/stage1_continuous_y_seed123.zip
```

The archive should extract `batch_*.pt` files directly into the target directory.
If you create it locally from the prior directory, use:

```bash
cd /mnt/c/Users/apex/Code/Python/tabicl/data/stage1_continuous_y_seed123
zip -r /tmp/stage1_continuous_y_seed123.zip .
rclone copy /tmp/stage1_continuous_y_seed123.zip \
  gdrive:ndp-nanotabpfn-v10-original02/runs/prior
```

The normal sync commands exclude `runs/prior/**`, so the large archive is not
accidentally downloaded during setup. The archive is handled only by the explicit
`prior-extract` command.

The `assets/` folder is optional. If the border asset is not already synced from
Google Drive, the Vast script can generate it with the `borders` command.

## 3. Copy rclone Config to Vast

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
rclone lsd gdrive:ndp-nanotabpfn-v10-original02
```

If those commands list folders, Google Drive access is ready.

## 4. Bootstrap the Vast Machine

SSH into the Vast instance. Vast usually gives a command like:

```bash
ssh -p <port> root@<host> -L 8080:localhost:8080
```

On Vast, set reusable environment variables:

```bash
export NT10_REMOTE="gdrive:ndp-nanotabpfn-v10-original02"
export NT10_RUN_NAME="vast_l5_bars5000_seed123"
export NT10_PRIOR_DIR="/workspace/prior/stage1_continuous_y_seed123"
export NT10_PRIOR_ARCHIVE_REMOTE="$NT10_REMOTE/runs/prior/stage1_continuous_y_seed123.zip"
export NT10_PRIOR_EXPECTED_BATCHES=15000
export NT10_SYNC_INTERVAL=1800
export NT10_DISABLE_WANDB=1

# Optional W&B online logging:
# export NT10_DISABLE_WANDB=0
# export NT10_WANDB_API_KEY="<your_wandb_api_key>"
# export NT10_WANDB_RUN_NAME="$NT10_RUN_NAME"

# Optional model/training overrides:
# export NT10_NUM_EPOCHS=10
# export NT10_BATCH_SIZE=32
# export NT10_MICRO_BATCH_SIZE=4
# export NT10_NUM_LAYERS=5
# export NT10_NUM_BARS=5000
```

Run setup:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/Bornhug/ndp_pytorch_refactor/branch01/nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh \
  -o /tmp/vast_nt10.sh && \
bash /tmp/vast_nt10.sh setup
```

`setup` does these things only:

- saves the `NT10_*` values to `~/.nt10_vast_env`
- installs system tools, including `rclone`, `tmux`, and `bsdtar`
- sparse-checks out only `nanoTabPFN_version10_original02` from GitHub
- creates or updates the `ndp_sim` conda environment
- installs `nanoTabPFN_version10_original02/requirements.txt`
- syncs previous runs, assets, scikit-learn cache, and W&B files from Google Drive
- runs `doctor`

It does not download the prior ZIP, extract prior data, or start training.

## 5. Prepare the Border Asset

After setup, run:

```bash
cd /workspace/ndp_pytorch_refactor
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh borders
```

If `assets/tabpfn_v2_regressor_borders_5000.pt` already exists, this verifies
it. If it is missing, this generates it from the installed `tabpfn` package.

Push the asset to Google Drive so future Vast instances can reuse it:

```bash
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh sync-push
```

## 6. Stream the Prior ZIP Without Storing It

The Vast disk may be too small to hold both the ZIP and extracted prior data.
Do not run `rclone copy` for the ZIP.

Instead, stream it:

```bash
cd /workspace/ndp_pytorch_refactor
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh prior-extract
```

Internally this runs the equivalent of:

```bash
rclone cat gdrive:ndp-nanotabpfn-v10-original02/runs/prior/stage1_continuous_y_seed123.zip \
  | bsdtar -xf - -C /workspace/prior/stage1_continuous_y_seed123.tmp
```

That means:

- the ZIP stays on Google Drive
- no local ZIP copy is written to the Vast disk
- extracted batches are written to `NT10_PRIOR_TMP_DIR`
- the active `NT10_PRIOR_DIR` is not created yet
- training is not started

Check progress or final state:

```bash
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh prior-status
```

Expected final status:

```text
batch count: 15000
temp directory: /workspace/prior/stage1_continuous_y_seed123.tmp
final directory: missing /workspace/prior/stage1_continuous_y_seed123
```

If a previous temp extraction exists and you intentionally want to replace it:

```bash
export NT10_PRIOR_OVERWRITE_TMP=1
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh prior-extract
```

## 7. Inspect, Then Activate the Prior Data

Inspect the extracted temp dataset first:

```bash
ls -lh /workspace/prior/stage1_continuous_y_seed123.tmp | head
ls -lh /workspace/prior/stage1_continuous_y_seed123.tmp | tail
df -h /workspace
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh prior-status
```

After inspection, activate it:

```bash
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh prior-activate
```

`prior-activate` verifies the expected batch count and then moves:

```text
/workspace/prior/stage1_continuous_y_seed123.tmp
```

to:

```text
/workspace/prior/stage1_continuous_y_seed123
```

Training uses `NT10_PRIOR_DIR`, so training cannot start successfully until this
activation step is done.

## 8. Check Before Training

Run:

```bash
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh doctor
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh status
```

`doctor` should show:

- PyTorch CUDA is available
- `rclone` can list `gdrive:ndp-nanotabpfn-v10-original02`
- the border asset exists and has 5001 borders for 5000 bars
- `NT10_PRIOR_DIR` exists
- `NT10_PRIOR_DIR` contains exactly `15000` `batch_*.pt` files

If `doctor` warns about borders or prior data, do not train yet.

## 9. Start Training Only When Ready

Start training in tmux:

```bash
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh train-tmux
```

Start periodic checkpoint/cache upload in a separate tmux session:

```bash
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh start-sync-tmux
```

Attach to training:

```bash
tmux attach -t nt10-train
```

Detach without stopping training:

```text
Ctrl+B
D
```

Before stopping the Vast instance, push state back to Google Drive:

```bash
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh sync-push
```

## 10. Starting a New Run

Because `train.py` does not resume model weights, use a fresh run directory for
another training attempt:

```bash
export NT10_RUN_NAME="vast_l5_bars5000_seed123_run02"
export NT10_CHECKPOINT_DIR="/workspace/ndp_pytorch_refactor/nanoTabPFN_version10_original02/runs/$NT10_RUN_NAME"
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh train-tmux
```

Only bypass the checkpoint guard when you intentionally accept overwriting or
mixing outputs:

```bash
export NT10_ALLOW_NONEMPTY_CHECKPOINT_DIR=1
```

## 11. Evaluate and Sweep

Evaluate the latest checkpoint under `NT10_CHECKPOINT_DIR`:

```bash
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh eval-latest
```

Sweep all checkpoints in the run directory:

```bash
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh sweep
```

Then push results:

```bash
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh sync-push
```

## 12. Common Commands

```bash
# Show help.
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh help

# Re-run setup checks.
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh doctor

# Show checkpoints, prior counts, cache sizes, sync logs, and latest outputs.
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh status

# Verify or create the TabPFN v2 regressor border asset.
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh borders

# Show prior archive, disk space, counts, and sample files.
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh prior-status

# Stream-extract prior ZIP into stage1_continuous_y_seed123.tmp only.
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh prior-extract

# Move inspected temp prior data to active prior directory.
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh prior-activate

# Start training only after borders and prior activation.
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh train-tmux

# Push runs/assets/sklearn cache/wandb to Google Drive.
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh sync-push

# Open an interactive menu.
bash nanoTabPFN_version10_original02/scripts/vast_nanotabpfn_v10_original02.sh menu
```
