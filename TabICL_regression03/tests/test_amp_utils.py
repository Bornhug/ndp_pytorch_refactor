from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabicl_style.utils import amp_dtype_name, resolve_amp_dtype, resolve_amp_settings
from tabicl_style.train import (
    optimizer_step_with_scaler_stats,
    skipped_optimizer_step_stats,
)


class _DummyOptimizer:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


class _DummyScaler:
    def __init__(
        self, *, enabled: bool, scale_before: float, scale_after: float
    ) -> None:
        self._enabled = enabled
        self._scale_before = scale_before
        self._scale_after = scale_after
        self._updated = False

    def is_enabled(self) -> bool:
        return self._enabled

    def get_scale(self) -> float:
        return self._scale_after if self._updated else self._scale_before

    def step(self, optimizer: _DummyOptimizer) -> None:
        if not self._enabled or self._scale_after >= self._scale_before:
            optimizer.step()

    def update(self) -> None:
        self._updated = True


def test_resolve_amp_dtype_auto_cpu_uses_float16() -> None:
    assert resolve_amp_dtype("auto", device=torch.device("cpu")) == torch.float16


def test_resolve_amp_dtype_aliases() -> None:
    assert resolve_amp_dtype("fp16") == torch.float16
    assert resolve_amp_dtype("half") == torch.float16
    assert resolve_amp_dtype("bf16") == torch.bfloat16
    assert resolve_amp_dtype("bfloat16") == torch.bfloat16
    assert resolve_amp_dtype("fp32") == torch.float32
    assert resolve_amp_dtype("float32") == torch.float32


def test_resolve_amp_dtype_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported AMP dtype"):
        resolve_amp_dtype("not-a-real-dtype")


def test_resolve_amp_settings_disables_cpu_amp() -> None:
    enabled, dtype = resolve_amp_settings(True, "auto", device="cpu")

    assert enabled is False
    assert dtype == torch.float16


def test_amp_dtype_name_is_stable() -> None:
    assert amp_dtype_name(torch.float16) == "float16"
    assert amp_dtype_name(torch.bfloat16) == "bfloat16"
    assert amp_dtype_name(torch.float32) == "float32"


def test_optimizer_step_with_scaler_stats_detects_skipped_step() -> None:
    optimizer = _DummyOptimizer()
    scaler = _DummyScaler(enabled=True, scale_before=128.0, scale_after=64.0)

    stats = optimizer_step_with_scaler_stats(optimizer, scaler, amp=True)

    assert optimizer.step_count == 0
    assert stats["scaler_step_skipped"] == 1


def test_optimizer_step_with_scaler_stats_records_applied_step() -> None:
    optimizer = _DummyOptimizer()
    scaler = _DummyScaler(enabled=True, scale_before=128.0, scale_after=128.0)

    stats = optimizer_step_with_scaler_stats(optimizer, scaler, amp=True)

    assert optimizer.step_count == 1
    assert stats["scaler_step_skipped"] == 0


def test_skipped_optimizer_step_stats_reports_no_scaler_skip() -> None:
    scaler = _DummyScaler(enabled=True, scale_before=128.0, scale_after=128.0)

    stats = skipped_optimizer_step_stats(scaler, amp=True)

    assert stats["scaler_step_skipped"] == 0
