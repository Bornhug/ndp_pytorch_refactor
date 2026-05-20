from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabicl_style.utils import amp_dtype_name, resolve_amp_dtype, resolve_amp_settings


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
