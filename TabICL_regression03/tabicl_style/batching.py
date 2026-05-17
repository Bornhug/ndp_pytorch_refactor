"""Batch shape helpers for TabICL pre-generated prior batches.

TabICL prior batches can contain tasks with different sequence lengths and
feature counts. The training loop pads the DataLoader batch, splits it into
smaller task groups for memory, then trims each micro-batch back to the active
sequence/feature sizes before context/target splitting.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch


def pad_nested_batch(batch: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Convert nested DataLoader tensors into padded dense tensors.

    Pre-generated TabICL batches may arrive as PyTorch nested tensors when
    tasks have different shapes. The rest of the training code expects normal
    dense tensors, so nested entries are padded with zeros while already-dense
    entries are passed through unchanged.
    """
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
    """Split a full task batch into smaller micro-batches along batch dim 0.

    The DataLoader batch size controls how many tasks make one optimizer step.
    ``micro_batch_size`` controls how many of those tasks are processed in one
    forward/backward pass. Gradients from the returned micro-batches are later
    accumulated before a single optimizer update.
    """
    batch_size = batch[0].shape[0]
    num_micro_batches = int(math.ceil(batch_size / float(micro_batch_size)))
    micro_batches = [torch.split(t, micro_batch_size, dim=0) for t in batch]
    return list(zip(*micro_batches)), num_micro_batches


def validate_micro_batch(
    micro_seq_len: torch.Tensor, micro_train_size: torch.Tensor
) -> Tuple[int, int]:
    """Return shared sequence/context sizes for one micro-batch.

    ``split_context_target`` needs one ``seq_len`` and one ``train_size`` for
    the whole micro-batch. If tasks inside the same micro-batch disagree on
    either value, the code cannot slice them consistently, so this fails early.
    """
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
    """Trim padded tensors to the active sequence length and feature count.

    ``micro_X`` and ``micro_y`` may include zero padding from
    ``pad_nested_batch``. ``seq_len`` removes padded rows, and ``micro_d`` stores
    each task's feature count, so the maximum active feature count in the
    micro-batch removes padded feature columns.
    """
    if micro_X.shape[1] > seq_len:
        micro_X = micro_X[:, :seq_len]
    if micro_y.shape[1] > seq_len:
        micro_y = micro_y[:, :seq_len]
    max_features = int(micro_d.max().item())
    if micro_X.shape[-1] > max_features:
        micro_X = micro_X[..., :max_features]
    return micro_X, micro_y
