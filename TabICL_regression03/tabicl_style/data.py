"""TabICL prior dataset loading for NDP regression training.

Supports two modes:
  1. Live generation via TabICL's PriorDataset (requires tabicl repo).
  2. Loading pre-generated batches from disk (batch_*.pt files).

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


def build_dataset(config: Any):
    """Build a dataset from config (either live or pre-generated).

    When prior_dir is set, loads pre-generated batch_*.pt files using TabICL's
    LoadPriorDataset.  When prior_dir is None, generates tasks live via
    TabICL's PriorDataset.  Both paths require the tabicl repo on sys.path.
    """
    if config.tabicl_repo is None:
        raise ValueError(
            "--tabicl-repo is required (path to local tabicl repo root). "
            "Example: --tabicl-repo /mnt/c/Users/apex/Code/Python/tabicl"
        )

    add_tabicl_repo(config.tabicl_repo)
    from tabicl.prior.genload import LoadPriorDataset

    if config.prior_dir is None:
        from tabicl.prior.dataset import PriorDataset

        dataset = PriorDataset(
            batch_size=config.batch_size,
            batch_size_per_gp=config.batch_size_per_gp,
            min_features=config.min_features,
            max_features=config.max_features,
            max_classes=1,  # regression: single continuous output
            min_seq_len=config.min_seq_len,
            max_seq_len=config.max_seq_len,
            log_seq_len=config.log_seq_len,
            seq_len_per_gp=config.seq_len_per_gp,
            min_train_size=config.min_train_size,
            max_train_size=config.max_train_size,
            replay_small=config.replay_small,
            prior_type=config.prior_type,
            device=config.prior_device,
            n_jobs=1,
        )
    else:
        dataset = LoadPriorDataset(
            data_dir=config.prior_dir,
            batch_size=config.batch_size,
            ddp_world_size=1,
            ddp_rank=0,
            start_from=config.load_prior_start,
            delete_after_load=config.delete_after_load,
            device=config.prior_device,
        )
    return dataset


def build_dataloader(config: Any, dataset) -> DataLoader:
    pin_memory = True if config.prior_device == "cpu" else False
    pin_memory_device = config.device if config.prior_device == "cpu" else ""
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=1,
        prefetch_factor=4,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
    )
