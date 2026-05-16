from __future__ import annotations

import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_diffusion_processes.process import GaussianDiffusion, cosine_schedule, loss
from neural_diffusion_processes.types import Batch
from tabicl_style.model import NDPRegressor
from tabicl_style.sampling import sample_predictions


def _make_model() -> NDPRegressor:
    torch.manual_seed(0)
    return NDPRegressor(
        embedding_size=16,
        num_attention_heads=4,
        num_layers=2,
    )


def _make_process(num_timesteps: int = 8) -> GaussianDiffusion:
    return GaussianDiffusion(cosine_schedule(0.0003, 0.5, timesteps=num_timesteps))


def test_forward_shape_without_context() -> None:
    model = _make_model()
    out = model(
        x_target=torch.randn(2, 5, 3),
        y_target=torch.randn(2, 5, 1),
        t=torch.tensor([1, 3], dtype=torch.long),
    )
    assert out.shape == (2, 5, 1)


def test_forward_shape_with_context_and_masks() -> None:
    model = _make_model()
    out = model(
        x_target=torch.randn(2, 4, 3),
        y_target=torch.randn(2, 4, 1),
        t=torch.tensor([0, 2], dtype=torch.long),
        mask_target=torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.float32),
        x_context=torch.randn(2, 6, 3),
        y_context=torch.randn(2, 6, 1),
        mask_context=torch.tensor(
            [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]],
            dtype=torch.float32,
        ),
    )
    assert out.shape == (2, 4, 1)


def test_combined_sequence_keeps_context_slice_stable() -> None:
    model = _make_model()
    layer = model.core.layers[0]
    original_forward = layer.forward
    records: list[tuple[torch.Tensor, int | None]] = []

    def wrapped_forward(self, s, t, mask=None, *, split_idx=None):
        records.append((s.detach().clone(), split_idx))
        return original_forward(s, t, mask, split_idx=split_idx)

    layer.forward = types.MethodType(wrapped_forward, layer)
    try:
        x_target = torch.randn(1, 3, 2)
        x_context = torch.randn(1, 4, 2)
        y_context = torch.randn(1, 4, 1)
        for timestep in (1, 4):
            model(
                x_target=x_target,
                y_target=torch.randn(1, 3, 1),
                t=torch.tensor([timestep], dtype=torch.long),
                mask_target=torch.zeros(1, 3),
                x_context=x_context,
                y_context=y_context,
                mask_context=torch.zeros(1, 4),
            )
    finally:
        layer.forward = original_forward

    assert len(records) == 2
    seq0, split0 = records[0]
    seq1, split1 = records[1]
    assert split0 == split1 == 4
    assert torch.allclose(seq0[:, :split0], seq1[:, :split1])
    assert not torch.allclose(seq0[:, split0:], seq1[:, split1:])


def test_loss_and_sampling_smoke_with_context() -> None:
    model = _make_model()
    process = _make_process(num_timesteps=8)
    x_target = torch.randn(2, 3, 2)
    y_target = torch.randn(2, 3, 1)
    x_context = torch.randn(2, 4, 2)
    y_context = torch.randn(2, 4, 1)
    mask_target = torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.float32)
    mask_context = torch.tensor([[0, 0, 0, 1], [0, 1, 0, 0]], dtype=torch.float32)

    batch = Batch(
        x_target=x_target,
        y_target=y_target,
        x_context=x_context,
        y_context=y_context,
        mask_target=mask_target,
        mask_context=mask_context,
    )
    loss_value = loss(
        process,
        model,
        batch,
        torch.Generator().manual_seed(0),
        num_timesteps=8,
        loss_type="l1",
    )
    assert loss_value.ndim == 0
    assert torch.isfinite(loss_value)

    for method in ("ddpm", "ddim"):
        preds = sample_predictions(
            process,
            model,
            x_target=x_target,
            x_context=x_context,
            y_context=y_context,
            mask_target=mask_target,
            mask_context=mask_context,
            num_steps=4,
            sampling_method=method,
        )
        assert preds.shape == (2, 3, 1)
        assert torch.isfinite(preds).all()


def test_ddim_final_step_returns_denoised_prediction() -> None:
    process = GaussianDiffusion(torch.linspace(1e-3, 2e-2, 5, dtype=torch.float32))
    yt = torch.tensor([[0.4], [-0.2]], dtype=torch.float32)
    eps_hat = torch.tensor([[0.1], [0.3]], dtype=torch.float32)

    x0_hat = process.predict_x0(eps_hat, yt, 3)
    out = process.ddim_backward_step(None, eps_hat, yt, 3, next_t=-1, eta=0.7)

    assert torch.allclose(out, x0_hat)
