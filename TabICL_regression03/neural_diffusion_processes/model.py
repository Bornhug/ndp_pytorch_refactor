import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


def timestep_embedding(
    t: torch.Tensor, embedding_dim: int, max_positions: int = 10_000
):
    """Sinusoidal embedding."""
    if t.ndim == 0:
        t = t.unsqueeze(0)

    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: [..., num_heads, seq_len, depth]
    attn_mask = None
    if mask is not None:
        # Existing masks use 1 for masked positions and 0 for valid attention.
        # Avoid passing all-zero masks so CUDA SDPA can use FlashAttention.
        if torch.count_nonzero(mask).item() > 0:
            attn_mask = mask == 0

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, sparse=False):
        del sparse
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0

        self.depth = d_model // num_heads
        self.attention = scaled_dot_product_attention

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, v, k, q, mask=None):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        rearrange_arg = "... seq_len (num_heads depth) -> ... num_heads seq_len depth"
        q = rearrange(q, rearrange_arg, num_heads=self.num_heads, depth=self.depth)
        k = rearrange(k, rearrange_arg, num_heads=self.num_heads, depth=self.depth)
        v = rearrange(v, rearrange_arg, num_heads=self.num_heads, depth=self.depth)

        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            mask_seq_q = mask[..., :, None]
            mask_seq_v = mask[..., None, :]
            mask = mask_seq_q + mask_seq_v
            mask = torch.where(mask == 0.0, mask, torch.ones_like(mask))
            mask = mask[..., None, :, :]

        scaled_attention = self.attention(q, k, v, mask)
        scaled_attention = rearrange(
            scaled_attention,
            "... num_heads seq_len depth -> ... seq_len (num_heads depth)",
        )
        return self.dense(scaled_attention)


class BiDimensionalAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.linear_t = nn.Linear(hidden_dim, hidden_dim)
        self.mha_d = MultiHeadAttention(2 * hidden_dim, num_heads)
        self.mha_n = MultiHeadAttention(2 * hidden_dim, num_heads)

    def forward(self, s, t, mask=None, *, split_idx: int | None = None):
        # s: [B, N, D, H]
        # t: [B, H]
        t = self.linear_t(t)[:, None, None, :]
        y = s + t
        y = torch.cat([y, y], dim=-1)  # [B, N, D, 2H]

        y_att_d = self.mha_d(y, y, y)

        y_r = y.transpose(1, 2)  # [B, D, N, 2H]
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, N]
        y_att_n = self.mha_n(y_r, y_r, y_r, mask)
        y_att_n = y_att_n.transpose(1, 2)  # [B, N, D, 2H]

        y = y_att_n + y_att_d
        residual, skip = torch.chunk(y, 2, dim=-1)
        residual = F.gelu(residual)
        skip = F.gelu(skip)

        if split_idx is not None:
            skip = skip[:, split_idx:, ...]

        return (s + residual) / math.sqrt(2.0), skip


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, sparse=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.linear_t = nn.Linear(hidden_dim, hidden_dim)
        self.mha_d = MultiHeadAttention(2 * hidden_dim, num_heads, sparse=sparse)

    def forward(self, s, t):
        t = self.linear_t(t)[:, None, :]
        y = s + t
        y = torch.cat([y, y], dim=-1)
        y_att_d = self.mha_d(y, y, y)
        residual, skip = torch.chunk(y_att_d, 2, dim=-1)
        residual = F.gelu(residual)
        skip = F.gelu(skip)
        return (s + residual) / math.sqrt(2.0), skip


class BiDimensionalAttentionModel(nn.Module):
    """
    Regression NDP core that keeps the public target/context signature but
    internally runs the attention stack over one combined sequence:
    `[context, target]`.
    """

    def __init__(self, n_layers, hidden_dim, num_heads, init_zero=True):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.init_zero = init_zero

        self.input_linear = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList(
            [BiDimensionalAttentionBlock(hidden_dim, num_heads) for _ in range(n_layers)]
        )
        self.proj_eps = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, 1)
        if init_zero:
            nn.init.zeros_(self.output_linear.weight)
            nn.init.zeros_(self.output_linear.bias)

    def process_inputs(self, x, y):
        if x is None or y is None:
            return None

        if x.ndim == 2:
            x = x.unsqueeze(0)
        if y.ndim == 2:
            y = y.unsqueeze(0)

        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if y.ndim == 3:
            y = y.unsqueeze(2)

        if x.size(0) != y.size(0) or x.size(1) != y.size(1):
            raise ValueError(f"Mismatched batch/point dims for x {x.shape} and y {y.shape}")

        if y.size(2) == 1 and x.size(2) > 1:
            y = y.expand(-1, -1, x.size(2), -1)
        elif y.size(2) != x.size(2):
            raise ValueError(f"Expected y D-dim to match x ({x.size(2)}), got {y.size(2)}")

        return torch.cat([x, y], dim=-1)

    def _prepare_mask(
        self,
        mask: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != 2:
            raise ValueError(f"Expected mask with shape [B, N] or [N], got {mask.shape}")
        if mask.size(0) == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        if mask.size(0) != batch_size or mask.size(1) != seq_len:
            raise ValueError(
                f"Expected mask shape [{batch_size}, {seq_len}], got {tuple(mask.shape)}"
            )
        return mask.to(device=device, dtype=dtype)

    def _combine_target_context(
        self,
        s_tgt: torch.Tensor,
        mask_tgt: torch.Tensor | None,
        *,
        s_ctx: torch.Tensor | None,
        mask_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int | None]:
        batch_size, n_tgt = s_tgt.shape[:2]
        mask_tgt = self._prepare_mask(
            mask_tgt,
            batch_size=batch_size,
            seq_len=n_tgt,
            device=s_tgt.device,
            dtype=s_tgt.dtype,
        )

        if s_ctx is None:
            return s_tgt, mask_tgt, None

        if s_ctx.size(0) != batch_size:
            raise ValueError(
                f"Context batch size {s_ctx.size(0)} must match target batch size {batch_size}"
            )
        if s_ctx.size(2) != s_tgt.size(2):
            raise ValueError(
                f"Context feature dim {s_ctx.size(2)} must match target feature dim {s_tgt.size(2)}"
            )

        n_ctx = s_ctx.size(1)
        mask_context = self._prepare_mask(
            mask_context,
            batch_size=batch_size,
            seq_len=n_ctx,
            device=s_tgt.device,
            dtype=s_tgt.dtype,
        )

        if mask_context is None and mask_tgt is None:
            mask = None
        else:
            if mask_context is None:
                mask_context = torch.zeros(
                    batch_size, n_ctx, device=s_tgt.device, dtype=s_tgt.dtype
                )
            if mask_tgt is None:
                mask_tgt = torch.zeros(
                    batch_size, n_tgt, device=s_tgt.device, dtype=s_tgt.dtype
                )
            mask = torch.cat([mask_context, mask_tgt], dim=1)
            if torch.count_nonzero(mask).item() == 0:
                mask = None

        s = torch.cat([s_ctx, s_tgt], dim=1)
        return s, mask, n_ctx

    def forward(
        self,
        x_tgt,
        y_tgt,
        t,
        mask_tgt=None,
        x_context=None,
        y_context=None,
        mask_context=None,
    ):
        if (x_context is None) != (y_context is None):
            raise ValueError("x_context and y_context must both be provided or both omitted.")
        if x_context is None and mask_context is not None:
            raise ValueError("mask_context requires x_context and y_context.")
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim > 1:
            t = t.view(t.shape[0])

        s_tgt = self.process_inputs(x_tgt, y_tgt)
        if s_tgt is None:
            raise ValueError("x_tgt and y_tgt are required.")

        s_ctx = None
        if x_context is not None:
            s_ctx = self.process_inputs(x_context, y_context)

        s, mask, split_idx = self._combine_target_context(
            s_tgt,
            mask_tgt,
            s_ctx=s_ctx,
            mask_context=mask_context,
        )
        s = F.gelu(self.input_linear(s))

        t_embedding = timestep_embedding(t, self.hidden_dim)

        skip = None
        for layer in self.layers:
            s, skip_connection = layer(s, t_embedding, mask, split_idx=split_idx)
            skip = skip_connection if skip is None else skip + skip_connection

        skip = reduce(skip, "b n d h -> b n h", "mean")
        eps = skip / math.sqrt(self.n_layers)
        eps = F.gelu(self.proj_eps(eps))
        eps = self.output_linear(eps)
        return eps


class AttentionModel(nn.Module):
    def __init__(
        self, n_layers, hidden_dim, num_heads, output_dim, sparse=False, init_zero=True
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.input_linear = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [AttentionBlock(hidden_dim, num_heads, sparse=sparse) for _ in range(n_layers)]
        )
        self.mid = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        if init_zero:
            nn.init.zeros_(self.output_linear.weight)
            nn.init.zeros_(self.output_linear.bias)

    def forward(self, x, y, t, mask=None):
        del mask
        x = torch.cat([x, y], dim=-1)
        x = F.gelu(self.input_linear(x))
        t_embedding = timestep_embedding(t, self.hidden_dim)

        skip = None
        for layer in self.layers:
            x, skip_connection = layer(x, t_embedding)
            skip = skip_connection if skip is None else skip + skip_connection

        eps = skip / math.sqrt(self.n_layers)
        eps = F.gelu(self.mid(eps))
        eps = self.output_linear(eps)
        return eps
