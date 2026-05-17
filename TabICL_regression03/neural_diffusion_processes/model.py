import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_timestep_embedding_table(
    num_timesteps: int,
    embedding_dim: int,
    max_positions: int = 10_000,
) -> torch.Tensor:
    """Precompute sinusoidal embeddings for all diffusion timesteps.

    Args:
        num_timesteps: Number of discrete diffusion timesteps.
        embedding_dim: Width of the returned embedding.
        max_positions: Frequency scale used by the sinusoidal basis.

    Returns:
        Tensor with shape ``[num_timesteps, embedding_dim]``.
    """
    t = torch.arange(num_timesteps, dtype=torch.float32)
    half_dim = embedding_dim // 2
    frequency_scale = math.log(max_positions) / (half_dim - 1)
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=torch.float32) * -frequency_scale
    )
    emb = t[:, None] * frequencies[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, sparse=False):
        """Create a PyTorch multi-head attention layer.

        The local wrapper exists only to adapt tensors with extra leading
        dimensions, such as ``[B, N, D, H]``, to PyTorch MHA's expected batched
        sequence shape ``[batch, seq, embed]``.
        """
        del sparse
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, v, k, q):
        """Apply attention over the second-to-last tensor dimension.

        Any dimensions before ``seq`` and ``embed`` are flattened into the batch
        dimension for ``nn.MultiheadAttention`` and restored afterward.
        """
        leading_shape = q.shape[:-2]
        q_seq_len, embed_dim = q.shape[-2:]
        k_seq_len = k.shape[-2]
        v_seq_len = v.shape[-2]
        q_flat = q.reshape(-1, q_seq_len, embed_dim)
        k_flat = k.reshape(-1, k_seq_len, embed_dim)
        v_flat = v.reshape(-1, v_seq_len, embed_dim)

        out, _ = self.attention(
            query=q_flat,
            key=k_flat,
            value=v_flat,
            need_weights=False,
        )
        return out.reshape(*leading_shape, q_seq_len, embed_dim)


class BiDimensionalAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        """Create one block that attends across feature and point axes."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.linear_t = nn.Linear(hidden_dim, hidden_dim)
        self.mha_d = MultiHeadAttention(2 * hidden_dim, num_heads)
        self.mha_n = MultiHeadAttention(2 * hidden_dim, num_heads)

    def forward(self, s, t, *, split_idx: int | None = None):
        """Run one bidimensional attention block.

        Args:
            s: Hidden state with shape ``[B, N, D, H]``.
            t: Timestep embedding with shape ``[B, H]``.
            split_idx: Number of leading context points in the combined
                sequence. When present, skip outputs are returned only for
                target points.

        Returns:
            Updated hidden state and target skip activations.
        """
        # s: [B, N, D, H]
        # t: [B, H]
        t = self.linear_t(t)[:, None, None, :]
        y = s + t
        y = torch.cat([y, y], dim=-1)  # [B, N, D, 2H]

        y_att_d = self.mha_d(y, y, y)

        y_r = y.transpose(1, 2)  # [B, D, N, 2H]
        y_att_n = self.mha_n(y_r, y_r, y_r)
        y_att_n = y_att_n.transpose(1, 2)  # [B, N, D, 2H]

        y = y_att_n + y_att_d
        residual, skip = torch.chunk(y, 2, dim=-1)
        residual = F.gelu(residual)
        skip = F.gelu(skip)

        if split_idx is not None:
            skip = skip[:, split_idx:, ...]

        return (s + residual) / math.sqrt(2.0), skip


class BiDimensionalAttentionModel(nn.Module):
    """
    Regression NDP core that keeps the public target/context signature but
    internally runs the attention stack over one combined sequence:
    `[context, target]`.
    """

    def __init__(
        self,
        n_layers,
        hidden_dim,
        num_heads,
        num_timesteps=500,
        init_zero=True,
    ):
        """Build the regression NDP attention stack.

        Args:
            n_layers: Number of bidimensional attention blocks.
            hidden_dim: Per-feature hidden width.
            num_heads: Number of attention heads in each block.
            num_timesteps: Number of discrete diffusion timesteps.
            init_zero: Whether to zero-initialize the final projection so the
                initial model predicts zero noise.
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_timesteps = num_timesteps
        self.init_zero = init_zero

        self.input_linear = nn.Linear(2, hidden_dim)
        timestep_embeddings = build_timestep_embedding_table(
            num_timesteps=num_timesteps,
            embedding_dim=hidden_dim,
        )
        self.register_buffer(
            "timestep_embeddings",
            timestep_embeddings,
            persistent=False,
        )

        self.layers = nn.ModuleList(
            [BiDimensionalAttentionBlock(hidden_dim, num_heads) for _ in range(n_layers)]
        )
        self.proj_eps = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, 1)
        if init_zero:
            nn.init.zeros_(self.output_linear.weight)
            nn.init.zeros_(self.output_linear.bias)

    def process_inputs(self, x, y):
        """Pack batched ``x`` features and ``y`` values into feature slots.

        Args:
            x: Feature tensor with shape ``[B, N, D]``.
            y: Target/value tensor with shape ``[B, N, 1]``.

        Returns:
            Tensor with shape ``[B, N, D, 2]`` containing ``[x_feature, y]`` for
            each feature slot.
        """
        x = x.unsqueeze(-1)
        y = y.unsqueeze(2).expand(-1, -1, x.size(2), -1)
        return torch.cat([x, y], dim=-1)

    def _combine_target_context(
        self,
        s_target: torch.Tensor,
        *,
        s_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int | None]:
        """Concatenate optional context points before target points.

        Returns the combined sequence and the context length used later to drop
        context skip activations from the prediction path.
        """
        if s_context is None:
            return s_target, None

        n_context = s_context.size(1)
        s = torch.cat([s_context, s_target], dim=1)
        return s, n_context

    def forward(
        self,
        x_target,
        y_target,
        t,
        x_context=None,
        y_context=None,
    ):
        """Predict diffusion noise for target values.

        Args:
            x_target: Target features with shape ``[B, N_target, D]``.
            y_target: Noised target values with shape ``[B, N_target, 1]``.
            t: Diffusion timesteps with shape ``[B]``.
            x_context: Optional context features with shape ``[B, N_context, D]``.
            y_context: Optional context values with shape ``[B, N_context, 1]``.

        Returns:
            Predicted noise with shape ``[B, N_target, 1]``.
        """
        s_target = self.process_inputs(x_target, y_target)

        s_context = None
        if x_context is not None and y_context is not None:
            s_context = self.process_inputs(x_context, y_context)

        s, split_idx = self._combine_target_context(
            s_target,
            s_context=s_context,
        )
        s = F.gelu(self.input_linear(s))

        t_embedding = self.timestep_embeddings[t]

        skip = None
        for layer in self.layers:
            s, skip_connection = layer(s, t_embedding, split_idx=split_idx)
            skip = skip_connection if skip is None else skip + skip_connection

        skip = skip.mean(dim=2)
        eps = skip / math.sqrt(self.n_layers)
        eps = F.gelu(self.proj_eps(eps))
        eps = self.output_linear(eps)
        return eps
