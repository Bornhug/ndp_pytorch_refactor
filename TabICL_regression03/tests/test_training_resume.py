from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabicl_style.config import Config
from tabicl_style.train import configure_checkpoint_dir
from tabicl_style.utils import (
    checkpoint_step,
    get_latest_checkpoint,
    infer_next_checkpoint_dir,
    infer_steps_per_epoch,
    prior_start_for_step,
)


def _write_checkpoint(path: Path, curr_step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"curr_step": int(curr_step)}, path)


def _write_prior_batch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"batch")


def test_prior_start_for_step_matches_resume_batch() -> None:
    assert prior_start_for_step(0, curr_step=0, steps_per_epoch=30) == 0
    assert prior_start_for_step(0, curr_step=5, steps_per_epoch=30) == 5
    assert prior_start_for_step(0, curr_step=30, steps_per_epoch=30) == 0
    assert prior_start_for_step(10, curr_step=5, steps_per_epoch=30) == 15


def test_infer_steps_per_epoch_respects_load_prior_start(tmp_path: Path) -> None:
    prior_dir = tmp_path / "prior"
    for index in range(5):
        _write_prior_batch(prior_dir / f"batch_{index:06d}.pt")

    config = Config()
    config.training.prior_dir = str(prior_dir)
    config.training.load_prior_start = 2

    assert infer_steps_per_epoch(config) == 3


def test_latest_checkpoint_uses_numeric_step_order(tmp_path: Path) -> None:
    run_dir = tmp_path / "run01"
    _write_checkpoint(run_dir / "step-20.pt", curr_step=20)
    _write_checkpoint(run_dir / "step-100.pt", curr_step=100)

    latest = get_latest_checkpoint(run_dir)

    assert latest is not None
    assert latest.name == "step-100.pt"
    assert checkpoint_step(latest) == 100


def test_configure_checkpoint_dir_creates_run_when_no_runs(tmp_path: Path) -> None:
    config = Config()
    config.training.checkpoint_dir = None
    config.training.checkpoint_path = None

    checkpoint = configure_checkpoint_dir(config, total_steps=10, runs_root=tmp_path)

    assert checkpoint is None
    assert Path(config.training.checkpoint_dir).name == "run01"
    assert infer_next_checkpoint_dir(tmp_path).name == "run02"


def test_configure_checkpoint_dir_resumes_latest_unfinished_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run01"
    checkpoint_path = run_dir / "step-5.pt"
    _write_checkpoint(checkpoint_path, curr_step=5)

    config = Config()
    config.training.checkpoint_dir = None
    config.training.checkpoint_path = None

    checkpoint = configure_checkpoint_dir(config, total_steps=10, runs_root=tmp_path)

    assert checkpoint == checkpoint_path
    assert Path(config.training.checkpoint_dir) == run_dir


def test_configure_checkpoint_dir_starts_new_run_after_completed_run(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path / "run01" / "step-10.pt", curr_step=10)

    config = Config()
    config.training.checkpoint_dir = None
    config.training.checkpoint_path = None

    checkpoint = configure_checkpoint_dir(config, total_steps=10, runs_root=tmp_path)

    assert checkpoint is None
    assert Path(config.training.checkpoint_dir).name == "run02"
