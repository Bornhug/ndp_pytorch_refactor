from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
DEFAULT_NUM_BARS = 5000
DEFAULT_BORDERS_PATH = HERE / "assets" / "tabpfn_v2_regressor_borders_5000.pt"


def load_borders(
    borders_path: str | Path | None = None,
    *,
    expected_num_bars: int = DEFAULT_NUM_BARS,
) -> torch.Tensor:
    """Load and validate normalized target borders for the bar regressor."""
    path = Path(borders_path) if borders_path is not None else DEFAULT_BORDERS_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"TabPFN v2 bar borders asset not found: {path}. "
            "Run nanoTabPFN_version10_original02/scripts/extract_tabpfn_v2_borders.py "
            "to create it."
        )

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        borders = payload.get("borders")
    else:
        borders = payload
    if borders is None:
        raise ValueError(f"Border asset {path} does not contain a 'borders' tensor.")

    borders = torch.as_tensor(borders, dtype=torch.float32).flatten().contiguous()
    expected_len = int(expected_num_bars) + 1
    if borders.numel() != expected_len:
        raise ValueError(
            f"Expected {expected_len} borders for {expected_num_bars} bars, "
            f"got {borders.numel()} in {path}."
        )
    if not torch.isfinite(borders).all():
        raise ValueError(f"Border asset {path} contains non-finite values.")
    if not torch.all(borders[1:] > borders[:-1]):
        raise ValueError(f"Border asset {path} must be strictly increasing.")
    return borders


class FullSupportBarDistribution(nn.Module):
    """TabPFN-style continuous density over finite bars plus tail buckets."""

    def __init__(self, borders: torch.Tensor, *, ignore_nan_targets: bool = True):
        super().__init__()
        borders = torch.as_tensor(borders, dtype=torch.float32).flatten().contiguous()
        if borders.ndim != 1 or borders.numel() < 3:
            raise ValueError("borders must be a 1-D tensor with at least 3 entries")
        if not torch.all(torch.isfinite(borders)):
            raise ValueError("borders must be finite")
        if not torch.all(borders[1:] > borders[:-1]):
            raise ValueError("borders must be strictly increasing")
        self.register_buffer("borders", borders)
        self.ignore_nan_targets = bool(ignore_nan_targets)

    @property
    def num_bars(self) -> int:
        return int(self.borders.numel() - 1)

    @property
    def bucket_widths(self) -> torch.Tensor:
        return self.borders[1:] - self.borders[:-1]

    @staticmethod
    def halfnormal_with_p_weight_before(
        range_max: torch.Tensor | float,
        *,
        p: float = 0.5,
    ) -> torch.distributions.HalfNormal:
        range_tensor = torch.as_tensor(range_max)
        unit = torch.distributions.HalfNormal(range_tensor.new_tensor(1.0))
        scale = range_tensor / unit.icdf(range_tensor.new_tensor(float(p)))
        return torch.distributions.HalfNormal(scale.clamp_min(torch.finfo(scale.dtype).eps))

    def map_to_bucket_idx(self, y: torch.Tensor) -> torch.Tensor:
        target_sample = torch.searchsorted(self.borders, y) - 1
        target_sample = target_sample.to(torch.long)
        target_sample[y == self.borders[0]] = 0
        target_sample[y == self.borders[-1]] = self.num_bars - 1
        return target_sample

    def compute_scaled_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        return torch.log_softmax(logits, dim=-1) - torch.log(widths)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return negative log density with shape ``logits.shape[:-1]``."""
        if logits.shape[-1] != self.num_bars:
            raise ValueError(f"logits last dim {logits.shape[-1]} != num_bars {self.num_bars}")

        y = y.to(device=logits.device, dtype=logits.dtype).reshape(*logits.shape[:-1])
        nan_mask = torch.isnan(y)
        if nan_mask.any() and not self.ignore_nan_targets:
            raise ValueError("NaN targets are not allowed when ignore_nan_targets=False")
        y_for_map = torch.nan_to_num(y, nan=0.0)
        target_sample = self.map_to_bucket_idx(y_for_map).clamp_(0, self.num_bars - 1)

        log_probs = self.compute_scaled_log_probs(logits).gather(
            -1,
            target_sample.unsqueeze(-1),
        ).squeeze(-1)

        widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        eps = torch.finfo(logits.dtype).eps
        left_mask = target_sample == 0
        right_mask = target_sample == self.num_bars - 1

        if left_mask.any():
            left_tail = self.halfnormal_with_p_weight_before(widths[0])
            distance = (borders[1] - y_for_map[left_mask]).clamp_min(eps)
            log_probs[left_mask] += left_tail.log_prob(distance) + torch.log(widths[0])
        if right_mask.any():
            right_tail = self.halfnormal_with_p_weight_before(widths[-1])
            distance = (y_for_map[right_mask] - borders[-2]).clamp_min(eps)
            log_probs[right_mask] += right_tail.log_prob(distance) + torch.log(widths[-1])

        nll = -log_probs
        if nan_mask.any():
            nll = nll.masked_fill(nan_mask, 0.0)
        return nll

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        bucket_means = borders[:-1] + widths / 2.0
        left_tail = self.halfnormal_with_p_weight_before(widths[0])
        right_tail = self.halfnormal_with_p_weight_before(widths[-1])
        bucket_means = bucket_means.clone()
        bucket_means[0] = borders[1] - left_tail.mean
        bucket_means[-1] = borders[-2] + right_tail.mean
        return probs @ bucket_means

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        density = torch.softmax(logits, dim=-1) / widths
        mode_idx = torch.argmax(density, dim=-1)
        bucket_means = self.borders.to(device=logits.device, dtype=logits.dtype)[:-1] + widths / 2.0
        return bucket_means[mode_idx]

    def median(self, logits: torch.Tensor) -> torch.Tensor:
        return self.icdf(logits, 0.5)

    def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        target = float(left_prob) * torch.ones(
            *cumprobs.shape[:-1],
            1,
            device=logits.device,
            dtype=logits.dtype,
        )
        idx = torch.searchsorted(cumprobs, target).squeeze(-1).clamp(0, self.num_bars - 1)
        cumprobs_with_zero = torch.cat(
            [torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device, dtype=logits.dtype), cumprobs],
            dim=-1,
        )
        prob_before = cumprobs_with_zero.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        bucket_prob = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        rest_prob = target.squeeze(-1) - prob_before
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        left_border = borders[idx]
        right_border = borders[idx + 1]
        return left_border + (right_border - left_border) * rest_prob / bucket_prob.clamp_min(
            torch.finfo(logits.dtype).eps
        )

    def cdf(self, logits: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Return CDF values for target positions in ``ys``."""
        ys = ys.to(device=logits.device, dtype=logits.dtype)
        if ys.ndim == 1 and ys.shape[0] == logits.shape[-2]:
            ys = ys.reshape(*logits.shape[:-1])
        if ys.shape != logits.shape[:-1]:
            ys = ys.reshape(*logits.shape[:-1])

        probs = torch.softmax(logits, dim=-1)
        idx = self.map_to_bucket_idx(ys).clamp(0, self.num_bars - 1)
        prob_before = (torch.cumsum(probs, dim=-1) - probs).gather(
            -1,
            idx.unsqueeze(-1),
        ).squeeze(-1)
        bucket_prob = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

        widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        share = ((ys - borders[idx]) / widths[idx]).clamp(0.0, 1.0)

        left_mask = idx == 0
        right_mask = idx == self.num_bars - 1
        if left_mask.any():
            left_tail = self.halfnormal_with_p_weight_before(widths[0])
            distance = (borders[1] - ys[left_mask]).clamp_min(torch.finfo(logits.dtype).eps)
            share[left_mask] = 1.0 - left_tail.cdf(distance)
        if right_mask.any():
            right_tail = self.halfnormal_with_p_weight_before(widths[-1])
            distance = (ys[right_mask] - borders[-2]).clamp_min(0.0)
            share[right_mask] = right_tail.cdf(distance)

        return (prob_before + bucket_prob * share).clamp(0.0, 1.0)


def build_distribution(
    borders_path: str | Path | None = None,
    *,
    expected_num_bars: int = DEFAULT_NUM_BARS,
) -> FullSupportBarDistribution:
    return FullSupportBarDistribution(
        load_borders(borders_path, expected_num_bars=expected_num_bars)
    )


def border_asset_metadata(
    borders_path: str | Path | None = None,
    *,
    expected_num_bars: int = DEFAULT_NUM_BARS,
) -> dict[str, Any]:
    path = Path(borders_path) if borders_path is not None else DEFAULT_BORDERS_PATH
    borders = load_borders(path, expected_num_bars=expected_num_bars)
    return {
        "borders_path": str(path),
        "num_bars": int(borders.numel() - 1),
        "border_count": int(borders.numel()),
        "border_min": float(borders[0].item()),
        "border_max": float(borders[-1].item()),
    }
