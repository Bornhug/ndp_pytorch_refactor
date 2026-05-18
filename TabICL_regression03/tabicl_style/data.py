"""Pre-generated TabICL prior dataset loading for NDP regression training.

Training consumes saved ``batch_*.pt`` files through TabICL's
``LoadPriorDataset``. Live prior generation is intentionally not supported in
this training path.

Each batch yields:
    (X, y, d, seq_len, train_size)
where X is [B, N, D] float32, y is [B, N] float32 (continuous targets),
d is [B], seq_len is [B], train_size is [B].
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader


def add_tabicl_repo(tabicl_repo: str | Path) -> Path:
    repo = Path(tabicl_repo).expanduser().resolve()
    src = repo / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"tabicl repo not found at {repo}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def build_dataset(config: Any, *, start_from: int | None = None):
    """Build a pre-generated prior dataset from ``config.prior_dir``."""
    if config.tabicl_repo is None:
        raise ValueError(
            "--tabicl-repo is required (path to local tabicl repo root). "
            "Example: --tabicl-repo /mnt/c/Users/apex/Code/Python/tabicl"
        )
    if config.prior_dir is None:
        raise ValueError(
            "training.prior_dir is required; live TabICL prior generation is "
            "not supported in TabICL_regression03 training."
        )

    prior_dir = Path(config.prior_dir).expanduser()
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_dir not found: {prior_dir}")
    if not any(prior_dir.glob("batch_*.pt")):
        raise FileNotFoundError(
            f"no pre-generated batch_*.pt files found in {prior_dir}"
        )

    add_tabicl_repo(config.tabicl_repo)
    from tabicl.prior.genload import LoadPriorDataset

    return LoadPriorDataset(
        data_dir=str(prior_dir),
        batch_size=config.batch_size,
        ddp_world_size=1,
        ddp_rank=0,
        start_from=config.load_prior_start if start_from is None else int(start_from),
        delete_after_load=config.delete_after_load,
        device=config.prior_device,
    )


def build_dataloader(config: Any, dataset) -> DataLoader:
    """Create the training DataLoader using configured worker processes."""
    num_workers = int(config.num_workers)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    pin_memory = True if config.prior_device == "cpu" else False
    pin_memory_device = config.device if config.prior_device == "cpu" else ""
    kwargs = {
        "batch_size": None,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["pin_memory_device"] = pin_memory_device
    return DataLoader(dataset, **kwargs)
