from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabicl_style.config import Config
from tabicl_style.lora import trainable_parameters
from tabicl_style.train import Trainer, configure_checkpoint_dir
from tabicl_style.utils import (
    checkpoint_step,
    compute_lr_from_schedule_steps,
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


def test_infer_steps_per_epoch_always_uses_all_remaining_batches(tmp_path: Path) -> None:
    prior_dir = tmp_path / "prior"
    for index in range(8):
        _write_prior_batch(prior_dir / f"batch_{index:06d}.pt")

    config = Config()
    config.training.prior_dir = str(prior_dir)
    config.training.load_prior_start = 2

    assert infer_steps_per_epoch(config) == 6


def test_trainer_total_steps_uses_all_batches_for_all_epochs(
    tmp_path: Path, monkeypatch
) -> None:
    prior_dir = tmp_path / "prior"
    for index in range(4):
        _write_prior_batch(prior_dir / f"batch_{index:06d}.pt")

    config = Config()
    config.training.prior_dir = str(prior_dir)
    config.training.num_epochs = 3
    config.training.device = "cpu"
    config.training.wandb_log = False
    config.training.checkpoint_dir = str(tmp_path / "runs" / "run01")
    config.model.embedding_size = 8
    config.model.num_attention_heads = 2
    config.model.num_layers = 1
    config.diffusion.timesteps = 4

    monkeypatch.setattr(
        "tabicl_style.train.build_model_and_process",
        lambda *_: (torch.nn.Linear(1, 1), None),
    )
    monkeypatch.setattr("tabicl_style.train.EMA", lambda model, decay: object())
    monkeypatch.setattr("tabicl_style.train.maybe_compile_model", lambda model, config: model)
    monkeypatch.setattr("tabicl_style.train.build_dataset", lambda *_, **__: [])
    monkeypatch.setattr("tabicl_style.train.build_dataloader", lambda *_, **__: [])
    monkeypatch.setattr(torch.optim, "AdamW", lambda *_, **__: object())

    trainer = Trainer(config)

    assert trainer.steps_per_epoch == 4
    assert trainer.total_steps == 12


def test_trainer_optimizer_receives_only_trainable_parameters(
    tmp_path: Path, monkeypatch
) -> None:
    prior_dir = tmp_path / "prior"
    for index in range(2):
        _write_prior_batch(prior_dir / f"batch_{index:06d}.pt")

    config = Config()
    config.training.prior_dir = str(prior_dir)
    config.training.num_epochs = 1
    config.training.device = "cpu"
    config.training.wandb_log = False
    config.training.checkpoint_dir = str(tmp_path / "runs" / "run01")

    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
    for param in model[0].parameters():
        param.requires_grad_(False)
    expected_params = trainable_parameters(model)
    captured_params: list[torch.nn.Parameter] = []

    def fake_adamw(params, *args, **kwargs):
        del args, kwargs
        captured_params.extend(list(params))
        return object()

    monkeypatch.setattr(
        "tabicl_style.train.build_model_and_process",
        lambda *_: (model, None),
    )
    monkeypatch.setattr("tabicl_style.train.EMA", lambda model, decay: object())
    monkeypatch.setattr("tabicl_style.train.maybe_compile_model", lambda model, config: model)
    monkeypatch.setattr("tabicl_style.train.build_dataset", lambda *_, **__: [])
    monkeypatch.setattr("tabicl_style.train.build_dataloader", lambda *_, **__: [])
    monkeypatch.setattr(torch.optim, "AdamW", fake_adamw)

    Trainer(config)

    assert [id(param) for param in captured_params] == [
        id(param) for param in expected_params
    ]


def test_lr_schedule_infers_from_full_run() -> None:
    config = Config()
    config.optimizer.init_lr = 2e-5
    config.optimizer.peak_lr = 2e-4
    config.optimizer.end_lr = 1e-5
    config.optimizer.warmup_fraction = 0.02
    config.optimizer.decay_fraction = 0.80

    warmup_steps = 600
    decay_steps = 24000
    first_step_lr = compute_lr_from_schedule_steps(
        config, 1, warmup_steps=warmup_steps, decay_steps=decay_steps
    )
    after_warmup_lr = compute_lr_from_schedule_steps(
        config, warmup_steps, warmup_steps=warmup_steps, decay_steps=decay_steps
    )
    final_lr = compute_lr_from_schedule_steps(
        config, 30000, warmup_steps=warmup_steps, decay_steps=decay_steps
    )

    assert first_step_lr > config.optimizer.init_lr
    assert after_warmup_lr == config.optimizer.peak_lr
    assert final_lr == config.optimizer.end_lr


def test_polynomial_lr_schedule_reaches_requested_endpoints() -> None:
    config = Config()
    config.optimizer.lr_schedule = "polynomial"
    config.optimizer.polynomial_power = 1.0
    config.optimizer.init_lr = 1e-5
    config.optimizer.peak_lr = 1e-5
    config.optimizer.end_lr = 5e-6

    start_lr = compute_lr_from_schedule_steps(
        config, 1, warmup_steps=0, decay_steps=3000
    )
    middle_lr = compute_lr_from_schedule_steps(
        config, 1500, warmup_steps=0, decay_steps=3000
    )
    final_lr = compute_lr_from_schedule_steps(
        config, 3000, warmup_steps=0, decay_steps=3000
    )

    assert start_lr == pytest.approx(1e-5)
    assert 5e-6 < middle_lr < 1e-5
    assert final_lr == pytest.approx(5e-6)


def test_invalid_lr_schedule_fails_clearly() -> None:
    config = Config()
    config.optimizer.lr_schedule = "not-a-schedule"

    with pytest.raises(ValueError, match="optimizer.lr_schedule"):
        compute_lr_from_schedule_steps(
            config, 1, warmup_steps=0, decay_steps=3000
        )


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
