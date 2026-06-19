"""Small local LoRA helpers for TabICL regression models."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_diffusion_processes.model import MultiHeadAttention


class LoRAProjection(nn.Module):
    """Low-rank residual projection used by LoRA-wrapped modules."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with frozen base weights and trainable LoRA residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.lora = LoRAProjection(
            base.in_features,
            base.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) + self.lora(x)


class LoRAMultiheadAttention(nn.Module):
    """Batch-first MHA with frozen base projections plus LoRA q/k/v/out adapters."""

    def __init__(
        self,
        base: nn.MultiheadAttention,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not base.batch_first:
            raise ValueError("LoRAMultiheadAttention only supports batch_first=True.")
        if not base._qkv_same_embed_dim:
            raise ValueError("LoRAMultiheadAttention only supports shared q/k/v dims.")
        if base.in_proj_weight is None:
            raise ValueError("LoRAMultiheadAttention requires in_proj_weight.")

        self.embed_dim = base.embed_dim
        self.kdim = base.kdim
        self.vdim = base.vdim
        self.num_heads = base.num_heads
        self.dropout = float(base.dropout)
        self.batch_first = True
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.in_proj_weight = nn.Parameter(
            base.in_proj_weight.detach().clone(),
            requires_grad=False,
        )
        if base.in_proj_bias is None:
            self.register_parameter("in_proj_bias", None)
        else:
            self.in_proj_bias = nn.Parameter(
                base.in_proj_bias.detach().clone(),
                requires_grad=False,
            )

        self.out_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=base.out_proj.bias is not None,
        )
        self.out_proj.weight.data.copy_(base.out_proj.weight.detach())
        self.out_proj.weight.requires_grad_(False)
        if self.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(base.out_proj.bias.detach())
            self.out_proj.bias.requires_grad_(False)

        self.lora_q = LoRAProjection(
            self.embed_dim,
            self.embed_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.lora_k = LoRAProjection(
            self.embed_dim,
            self.embed_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.lora_v = LoRAProjection(
            self.embed_dim,
            self.embed_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.lora_out = LoRAProjection(
            self.embed_dim,
            self.embed_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _project_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)

        q = F.linear(query, q_weight, q_bias) + self.lora_q(query)
        k = F.linear(key, k_weight, k_bias) + self.lora_k(key)
        v = F.linear(value, v_weight, v_bias) + self.lora_v(value)
        return q, k, v

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del average_attn_weights
        if attn_mask is not None or key_padding_mask is not None:
            raise ValueError("LoRA attention does not support masks in this model path.")

        q, k, v = self._project_qkv(query, key, value)
        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=bool(is_causal),
        )
        batch, _, q_seq_len, _ = attn.shape
        attn = attn.transpose(1, 2).contiguous().view(batch, q_seq_len, self.embed_dim)
        out = self.out_proj(attn) + self.lora_out(attn)
        weights = None
        if need_weights:
            weights = torch.empty(0, device=out.device, dtype=out.dtype)
        return out, weights


def is_lora_adapter_key(key: str) -> bool:
    """Return whether a state-dict key belongs to an adapter parameter."""
    return ".lora_" in key or ".lora." in key


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return parameters that should be optimized."""
    return [p for p in model.parameters() if p.requires_grad]


def trainable_parameter_names(model: nn.Module) -> list[str]:
    """Return names of parameters that should be optimized."""
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _replace_child_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    setattr(parent, child_name, new_module)


def _named_child_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module, str, nn.Module]]:
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            yield full_name, parent, child_name, child


def _should_lora_wrap_linear(name: str) -> bool:
    return name.endswith("input_linear") or name.endswith("linear_t") or name.endswith("proj_eps")


def apply_lora(model: nn.Module, config) -> nn.Module:
    """Freeze base weights and inject LoRA adapters when ``config.lora`` is enabled."""
    lora_config = getattr(config, "lora", None)
    if lora_config is None or not bool(getattr(lora_config, "enabled", False)):
        return model

    rank = int(lora_config.rank)
    alpha = float(lora_config.alpha)
    dropout = float(lora_config.dropout)

    for param in model.parameters():
        param.requires_grad_(False)

    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    for full_name, parent, child_name, child in _named_child_modules(model):
        if isinstance(child, nn.Linear) and _should_lora_wrap_linear(full_name):
            replacements.append(
                (
                    parent,
                    child_name,
                    LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
            )
    for parent, child_name, new_module in replacements:
        _replace_child_module(parent, child_name, new_module)

    for module in model.modules():
        if isinstance(module, MultiHeadAttention) and isinstance(
            module.attention, nn.MultiheadAttention
        ):
            module.attention = LoRAMultiheadAttention(
                module.attention,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

    if bool(getattr(lora_config, "train_layer_norm", True)):
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad_(True)

    if bool(getattr(lora_config, "train_output_head", True)):
        core = getattr(model, "core", None)
        output_head = getattr(core, "output_linear", None)
        if output_head is not None:
            for param in output_head.parameters():
                param.requires_grad_(True)

    trainable = trainable_parameters(model)
    if not trainable:
        raise ValueError("LoRA is enabled but no trainable parameters remain.")
    return model
