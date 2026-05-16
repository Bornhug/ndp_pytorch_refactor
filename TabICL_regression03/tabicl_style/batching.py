"""Batch helpers mirroring tabicl training flow."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch


def pad_nested_batch(batch: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    padded = []
    for t in batch:
        if hasattr(t, "is_nested") and t.is_nested:
            padded.append(t.to_padded_tensor(padding=0.0))
        else:
            padded.append(t)
    return padded


def split_micro_batches(
    batch: Sequence[torch.Tensor], micro_batch_size: int
) -> Tuple[List[Tuple[torch.Tensor, ...]], int]:
    batch_size = batch[0].shape[0]
    num_micro_batches = int(math.ceil(batch_size / float(micro_batch_size)))
    micro_batches = [torch.split(t, micro_batch_size, dim=0) for t in batch]
    return list(zip(*micro_batches)), num_micro_batches


def validate_micro_batch(
    micro_seq_len: torch.Tensor, micro_train_size: torch.Tensor
) -> Tuple[int, int]:
    if len(torch.unique(micro_seq_len)) > 1:
        raise ValueError(
            "All datasets in the micro batch must have the same sequence length."
        )
    if len(torch.unique(micro_train_size)) > 1:
        raise ValueError(
            "All datasets in the micro batch must have the same training size."
        )
    seq_len = int(micro_seq_len[0].item())
    train_size = int(micro_train_size[0].item())
    return seq_len, train_size


def align_micro_batch(
    micro_X: torch.Tensor, micro_y: torch.Tensor, micro_d: torch.Tensor, seq_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if micro_X.shape[1] > seq_len:
        micro_X = micro_X[:, :seq_len]
    if micro_y.shape[1] > seq_len:
        micro_y = micro_y[:, :seq_len]
    max_features = int(micro_d.max().item())
    if micro_X.shape[-1] > max_features:
        micro_X = micro_X[..., :max_features]
    return micro_X, micro_y
