from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bar_distribution import FullSupportBarDistribution
from model import NanoTabPFNModel, NanoTabPFNRegressor, TransformerEncoderLayer
from tabarena_eval.evaluation import eval_model_cv
from train import (
    _iter_micro_slices,
    _scaled_micro_loss,
    _validate_and_prepare_tasks,
    parse_args,
)


def test_full_support_nll_is_finite_for_outside_targets():
    dist = FullSupportBarDistribution(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
    logits = torch.zeros(4, dist.num_bars)
    targets = torch.tensor([-10.0, -0.5, 0.5, 10.0])

    loss = dist(logits, targets)

    assert loss.shape == targets.shape
    assert torch.isfinite(loss).all()


def test_middle_bin_nll_matches_width_corrected_ce():
    dist = FullSupportBarDistribution(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
    logits = torch.zeros(1, dist.num_bars)
    target = torch.tensor([0.25])

    loss = dist(logits, target)

    expected = -math.log(1.0 / dist.num_bars) + math.log(1.0)
    assert torch.allclose(loss, torch.tensor([expected], dtype=loss.dtype))


def test_edge_bin_nll_uses_halfnormal_tail_correction():
    dist = FullSupportBarDistribution(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
    logits = torch.zeros(1, dist.num_bars)
    target = torch.tensor([10.0])

    loss = dist(logits, target)
    tail = dist.halfnormal_with_p_weight_before(torch.tensor(1.0))
    expected_log_prob = math.log(1.0 / dist.num_bars) + float(
        tail.log_prob(torch.tensor(9.0))
    )

    assert torch.allclose(loss, torch.tensor([-expected_log_prob], dtype=loss.dtype))


def test_model_forward_outputs_bar_logits():
    model = NanoTabPFNModel(
        embedding_size=16,
        num_attention_heads=4,
        mlp_hidden_size=32,
        num_layers=1,
        num_bars=5000,
    )
    x = torch.randn(2, 6, 3)
    y = torch.randn(2, 3)

    logits = model((x, y), train_test_split_index=3)

    assert logits.shape == (2, 3, 5000)


def test_transformer_attention_disables_weight_outputs(monkeypatch):
    layer = TransformerEncoderLayer(embedding_size=8, nhead=2, mlp_hidden_size=16)
    calls = []

    def fake_attention(query, key, value, **kwargs):
        del key, value
        calls.append(dict(kwargs))
        return torch.zeros_like(query), None

    monkeypatch.setattr(layer.self_attention_between_features, "forward", fake_attention)
    monkeypatch.setattr(layer.self_attention_between_datapoints, "forward", fake_attention)

    src = torch.randn(2, 6, 3, 8)
    out = layer(src, train_test_split_index=3)

    assert out.shape == src.shape
    assert len(calls) == 3
    assert all(call.get("need_weights") is False for call in calls)


def test_micro_batch_loss_scaling_matches_full_batch_mean():
    loss_mat = torch.arange(1, 29, dtype=torch.float32).reshape(7, 4)
    total_target_elements = int(loss_mat.numel())

    micro_loss = sum(
        _scaled_micro_loss(loss_mat[micro_slice], total_target_elements)
        for micro_slice in _iter_micro_slices(loss_mat.shape[0], micro_batch_size=4)
    )

    assert torch.allclose(micro_loss, loss_mat.mean())


def test_parse_args_accepts_micro_batch_size(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--num-steps",
            "1",
            "--batch-size",
            "8",
            "--micro-batch-size",
            "4",
            "--disable-wandb",
        ],
    )

    args = parse_args()

    assert args.batch_size == 8
    assert args.micro_batch_size == 4


def test_regressor_decoding_shapes_and_monotonic_quantiles():
    borders = torch.linspace(-3.0, 3.0, 9)
    dist = FullSupportBarDistribution(borders)
    model = NanoTabPFNModel(
        embedding_size=16,
        num_attention_heads=4,
        mlp_hidden_size=32,
        num_layers=1,
        num_bars=dist.num_bars,
    )
    regressor = NanoTabPFNRegressor(model, torch.device("cpu"), bar_distribution=dist)
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(5, 2)).astype(np.float32)
    y_train = rng.normal(size=5).astype(np.float32)
    X_test = rng.normal(size=(4, 2)).astype(np.float32)
    regressor.fit(X_train, y_train)

    pred = regressor.predict(X_test)
    quantiles = regressor.predict(X_test, output_type="quantiles", quantiles=[0.05, 0.5, 0.95])
    full = regressor.predict_distribution(
        X_test,
        quantile_levels=np.asarray([0.5]),
        sharpness_levels=np.asarray([0.05, 0.95]),
        y_true=np.zeros(4, dtype=np.float32),
    )

    assert pred.shape == (4,)
    assert len(quantiles) == 3
    assert all(q.shape == (4,) for q in quantiles)
    assert np.all(quantiles[0] <= quantiles[1])
    assert np.all(quantiles[1] <= quantiles[2])
    assert full["quantile_boundaries"].shape == (1, 4)
    assert full["sharpness_boundaries"].shape == (2, 4)
    assert full["pit_values"].shape == (4,)


def test_eval_model_cv_outputs_uncertainty_and_sharpness_details():
    borders = torch.linspace(-3.0, 3.0, 9)
    dist = FullSupportBarDistribution(borders)
    model = NanoTabPFNModel(
        embedding_size=16,
        num_attention_heads=4,
        mlp_hidden_size=32,
        num_layers=1,
        num_bars=dist.num_bars,
    )
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 2)).astype(np.float32)
    y = rng.normal(size=12).astype(np.float32)

    metrics, details = eval_model_cv(
        model,
        {"toy": (X, y)},
        device=torch.device("cpu"),
        n_splits=2,
        random_state=0,
        num_bins=2,
        sharpness_coverage=0.90,
        bar_distribution=dist,
        return_details=True,
    )

    assert "R2" in metrics
    assert "QICE" in metrics
    assert "NMPIW" in metrics
    assert "overall_metrics" in details
    assert "overall_uncertainty" in details
    assert "uncertainty" in details["datasets"]["toy"]
    assert "sharpness" in details["datasets"]["toy"]


def test_training_task_preparation_clamps_normalized_context_and_targets():
    x = torch.zeros(1, 4, 2)
    y = torch.tensor([[0.0, 1.0, 100.0, -100.0]], dtype=torch.float32)

    prepared = _validate_and_prepare_tasks(
        x,
        y,
        train_test_split_index=2,
        std_min=1e-6,
        z_max=5.0,
        hard_z_threshold=1_000.0,
        max_hard_z_frac=1.0,
    )

    assert prepared["valid_tasks"] == 1
    assert prepared["clipped_target_values"] == 2
    assert torch.max(torch.abs(prepared["y_train_norm_valid"])) <= 5.0
    assert torch.max(torch.abs(prepared["y_target_norm_valid"])) <= 5.0
