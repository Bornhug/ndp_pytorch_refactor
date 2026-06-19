from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_diffusion_processes.process import (
    GaussianDiffusion,
    cosine_schedule,
    denoising_prediction_loss,
    loss,
    prepare_denoising_targets,
)
from neural_diffusion_processes.types import Batch
from neural_diffusion_processes.regressor import NDPRegressor, NDPRegressorWrapper
from tabicl_style.config import Config
from tabicl_style.lora import apply_lora, is_lora_adapter_key, trainable_parameter_names


def _make_model() -> NDPRegressor:
    torch.manual_seed(0)
    return NDPRegressor(
        embedding_size=16,
        num_attention_heads=4,
        num_layers=2,
        num_timesteps=8,
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


def test_forward_shape_with_context() -> None:
    model = _make_model()
    out = model(
        x_target=torch.randn(2, 4, 3),
        y_target=torch.randn(2, 4, 1),
        t=torch.tensor([0, 2], dtype=torch.long),
        x_context=torch.randn(2, 6, 3),
        y_context=torch.randn(2, 6, 1),
    )
    assert out.shape == (2, 4, 1)


def test_forward_shape_emb128_layers12() -> None:
    torch.manual_seed(0)
    model = NDPRegressor(
        embedding_size=128,
        num_attention_heads=8,
        num_layers=12,
        num_timesteps=8,
    )
    out = model(
        x_target=torch.randn(2, 4, 3),
        y_target=torch.randn(2, 4, 1),
        t=torch.tensor([0, 2], dtype=torch.long),
        x_context=torch.randn(2, 6, 3),
        y_context=torch.randn(2, 6, 1),
    )
    assert out.shape == (2, 4, 1)


def test_attention_blocks_use_pre_layernorm() -> None:
    model = _make_model()
    block = model.core.layers[0]

    assert isinstance(block.norm_s, torch.nn.LayerNorm)
    assert block.norm_s.normalized_shape == (block.hidden_dim,)


def test_lora_injection_preserves_forward_shape() -> None:
    config = Config()
    config.lora.enabled = True
    config.lora.rank = 4
    config.lora.alpha = 8
    model = apply_lora(_make_model(), config)

    out = model(
        x_target=torch.randn(2, 4, 3),
        y_target=torch.randn(2, 4, 1),
        t=torch.tensor([0, 2], dtype=torch.long),
        x_context=torch.randn(2, 5, 3),
        y_context=torch.randn(2, 5, 1),
    )

    assert out.shape == (2, 4, 1)


def test_lora_freezes_base_and_keeps_adaptation_params_trainable() -> None:
    config = Config()
    config.lora.enabled = True
    config.lora.rank = 4
    config.lora.alpha = 8
    config.lora.train_layer_norm = True
    config.lora.train_output_head = True
    model = apply_lora(_make_model(), config)

    trainable_names = trainable_parameter_names(model)

    assert any(".lora" in name for name in trainable_names)
    assert any("norm_s" in name for name in trainable_names)
    assert any("output_linear" in name for name in trainable_names)
    assert not model.core.input_linear.weight.requires_grad
    assert not model.core.layers[0].mha_d.attention.in_proj_weight.requires_grad
    assert model.core.layers[0].norm_s.weight.requires_grad
    assert model.core.output_linear.weight.requires_grad


def test_base_checkpoint_loads_into_lora_model_with_only_adapter_keys_missing() -> None:
    base_state = _make_model().state_dict()
    config = Config()
    config.lora.enabled = True
    config.lora.rank = 4
    config.lora.alpha = 8
    lora_model = apply_lora(_make_model(), config)

    load_result = lora_model.load_state_dict(base_state, strict=False)

    assert load_result.unexpected_keys == []
    assert load_result.missing_keys
    assert all(is_lora_adapter_key(key) for key in load_result.missing_keys)


def test_combined_sequence_keeps_context_slice_stable() -> None:
    model = _make_model()
    layer = model.core.layers[0]
    original_forward = layer.forward
    records: list[tuple[torch.Tensor, int | None]] = []

    def wrapped_forward(self, s, t, *, split_idx=None):
        records.append((s.detach().clone(), split_idx))
        return original_forward(s, t, split_idx=split_idx)

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
                x_context=x_context,
                y_context=y_context,
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
    batch = Batch(
        x_target=x_target,
        y_target=y_target,
        x_context=x_context,
        y_context=y_context,
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
        preds = process.sample(
            None,
            x_target,
            model=model,
            x_context=x_context,
            y_context=y_context,
            output_dim=y_context.shape[-1],
            num_sample_steps=4,
            method=method,
        )
        assert preds.shape == (2, 3, 1)
        assert torch.isfinite(preds).all()


def test_sampling_amp_request_is_cpu_safe() -> None:
    model = _make_model()
    process = _make_process(num_timesteps=8)
    preds = process.sample(
        None,
        torch.randn(2, 3, 2),
        model=model,
        x_context=torch.randn(2, 4, 2),
        y_context=torch.randn(2, 4, 1),
        output_dim=1,
        num_sample_steps=4,
        method="ddim",
        amp=True,
        amp_dtype=torch.float16,
    )

    assert preds.shape == (2, 3, 1)
    assert torch.isfinite(preds).all()


def test_split_loss_helpers_match_compat_loss() -> None:
    model = _make_model()
    process = _make_process(num_timesteps=8)
    batch = Batch(
        x_target=torch.randn(2, 3, 2),
        y_target=torch.randn(2, 3, 1),
        x_context=torch.randn(2, 4, 2),
        y_context=torch.randn(2, 4, 1),
    )

    t, yt, noise_true = prepare_denoising_targets(
        process,
        batch,
        torch.Generator().manual_seed(123),
        num_timesteps=8,
    )
    noise_hat = model(
        x_target=batch.x_target,
        y_target=yt,
        t=t.to(batch.x_target.device),
        x_context=batch.x_context,
        y_context=batch.y_context,
    )
    helper_loss = denoising_prediction_loss(
        noise_true,
        noise_hat,
        loss_type="l1",
    )
    compat_loss = loss(
        process,
        model,
        batch,
        torch.Generator().manual_seed(123),
        num_timesteps=8,
        loss_type="l1",
    )

    assert torch.allclose(helper_loss, compat_loss)


def test_denoising_prediction_loss_reduces_in_float32() -> None:
    noise_true = torch.randn(2, 3, 1, dtype=torch.float32)
    noise_hat = torch.randn(2, 3, 1, dtype=torch.bfloat16)

    loss_value = denoising_prediction_loss(noise_true, noise_hat, loss_type="l2")

    assert loss_value.dtype == torch.float32
    assert loss_value.ndim == 0


def test_regressor_wrapper_predict_repeated_samples_repeats_sequentially() -> None:
    model = _make_model()
    process = _make_process(num_timesteps=8)
    wrapper = NDPRegressorWrapper(
        model,
        process,
        torch.device("cpu"),
        num_sampling_steps=4,
        sampling_method="ddim",
    )
    X_train = torch.randn(5, 2).numpy()
    y_train = torch.randn(5).numpy()
    X_test = torch.randn(3, 2).numpy()

    wrapper.fit(X_train, y_train)
    repeated = wrapper.predict_repeated(X_test, 4)
    single = wrapper.predict(X_test)

    assert repeated.shape == (4, 3)
    assert single.shape == (3,)
    assert torch.isfinite(torch.from_numpy(repeated)).all()
    assert torch.isfinite(torch.from_numpy(single)).all()


def test_regressor_wrapper_passes_amp_settings_to_sampler() -> None:
    class DummyProcess:
        def __init__(self) -> None:
            self.sample_kwargs = None
            self.sample_count = 0
            self.batch_sizes = []

        def sample(self, key, x, **kwargs):
            del key
            self.sample_kwargs = kwargs
            self.sample_count += 1
            self.batch_sizes.append(x.shape[0])
            return torch.zeros(x.shape[0], x.shape[1], kwargs["output_dim"])

    process = DummyProcess()
    wrapper = NDPRegressorWrapper(
        torch.nn.Identity(),
        process,
        torch.device("cpu"),
        num_sampling_steps=4,
        sampling_method="ddim",
        amp=True,
        amp_dtype=torch.bfloat16,
    )

    wrapper.fit(torch.randn(5, 2).numpy(), torch.randn(5).numpy())
    out = wrapper.predict_repeated(torch.randn(3, 2).numpy(), 2)

    assert out.shape == (2, 3)
    assert process.sample_count == 2
    assert process.batch_sizes == [1, 1]
    assert process.sample_kwargs["amp"] is True
    assert process.sample_kwargs["amp_dtype"] == torch.bfloat16


def test_regressor_wrapper_predict_repeated_batches_parallel_repeats() -> None:
    class DummyProcess:
        def __init__(self) -> None:
            self.sample_count = 0
            self.batch_sizes = []

        def sample(self, key, x, **kwargs):
            del key
            self.sample_count += 1
            self.batch_sizes.append(x.shape[0])
            return torch.zeros(x.shape[0], x.shape[1], kwargs["output_dim"])

    process = DummyProcess()
    wrapper = NDPRegressorWrapper(
        torch.nn.Identity(),
        process,
        torch.device("cpu"),
        num_sampling_steps=4,
        sampling_method="ddim",
    )

    wrapper.fit(torch.randn(5, 2).numpy(), torch.randn(5).numpy())
    out = wrapper.predict_repeated(
        torch.randn(3, 2).numpy(),
        5,
        parallel_repeats=2,
    )

    assert out.shape == (5, 3)
    assert process.sample_count == 3
    assert process.batch_sizes == [2, 2, 1]


def test_regressor_wrapper_rejects_invalid_parallel_repeats() -> None:
    wrapper = NDPRegressorWrapper(
        torch.nn.Identity(),
        _make_process(num_timesteps=8),
        torch.device("cpu"),
        num_sampling_steps=4,
        sampling_method="ddim",
    )

    wrapper.fit(torch.randn(5, 2).numpy(), torch.randn(5).numpy())
    with pytest.raises(ValueError, match="parallel_repeats"):
        wrapper.predict_repeated(torch.randn(3, 2).numpy(), 2, parallel_repeats=0)


def test_ddim_final_step_returns_denoised_prediction() -> None:
    process = GaussianDiffusion(torch.linspace(1e-3, 2e-2, 5, dtype=torch.float32))
    yt = torch.tensor([[0.4], [-0.2]], dtype=torch.float32)
    eps_hat = torch.tensor([[0.1], [0.3]], dtype=torch.float32)

    x0_hat = process.predict_x0(eps_hat, yt, 3)
    out = process.ddim_backward_step(None, eps_hat, yt, 3, next_t=-1, eta=0.7)

    assert torch.allclose(out, x0_hat)
