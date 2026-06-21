import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import LayerNorm, Linear, MultiheadAttention

from bar_distribution import DEFAULT_NUM_BARS, FullSupportBarDistribution, build_distribution


class NanoTabPFNModel(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_bars: int = DEFAULT_NUM_BARS,
    ):
        """Initializes feature/target encoders, transformer stack, and decoder."""
        super().__init__()
        self.num_bars = int(num_bars)
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerEncoderLayer(
                    embedding_size, num_attention_heads, mlp_hidden_size
                )
            )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, self.num_bars)

    def forward(
        self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int
    ) -> torch.Tensor:
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)

        # B=batches, R=rows, C=columns, E=embedding
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)

        src = torch.cat([x_src, y_src], 2)
        for block in self.transformer_blocks:
            src = block(src, train_test_split_index=train_test_split_index)

        # Select target-label column embeddings only.
        output = src[:, train_test_split_index:, -1, :]
        # For bar regression this is [B, num_targets, num_bars].
        output = self.decoder(output)
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """
        Normalize features per dataset across all rows, clip, and embed.
        """
        del train_test_split_index
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        x = (x - mean) / std
        x = torch.clip(x, min=-100.0, max=100.0)
        x = x.unsqueeze(-1)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """
        Pad y_train to full row count with train mean, then embed.
        """
        mean = torch.mean(y_train, dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderLayer(nn.Module):
    """Modified Transformer encoder layer with row/column attention."""

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size,
            nhead,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.self_attention_between_features = MultiheadAttention(
            embedding_size,
            nhead,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape

        # Attention between features.
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(
            src, src, src, need_weights=False
        )[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)

        # Attention between datapoints.
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)

        src_left = self.self_attention_between_datapoints(
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            need_weights=False,
        )[0]
        src_right = self.self_attention_between_datapoints(
            src[:, train_test_split_index:],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            need_weights=False,
        )[0]
        src = torch.cat([src_left, src_right], dim=1) + src

        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)

        # MLP after attention.
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_bars: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_bars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class NanoTabPFNRegressor:
    """scikit-learn-like interface for regression."""

    def __init__(
        self,
        model: NanoTabPFNModel,
        device: torch.device,
        *,
        bar_distribution: FullSupportBarDistribution | None = None,
        borders_path: str | None = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.bar_distribution = (
            bar_distribution
            if bar_distribution is not None
            else build_distribution(borders_path, expected_num_bars=model.num_bars)
        ).to(device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.float32)
        self.y_mean = float(np.mean(self.y_train))
        self.y_std = float(max(np.std(self.y_train), 1e-6))

    def _predict_logits(self, X_test: np.ndarray) -> torch.Tensor:
        x = np.concatenate((self.X_train, np.asarray(X_test, dtype=np.float32)), axis=0)
        y_train_norm = (self.y_train - self.y_mean) / self.y_std

        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0).to(torch.float32).to(self.device)
            y_t = (
                torch.from_numpy(y_train_norm)
                .unsqueeze(0)
                .to(torch.float32)
                .to(self.device)
            )
            logits = self.model(
                (x_t, y_t), train_test_split_index=len(self.X_train)
            ).squeeze(0)
        return logits

    def _to_raw_space(self, value: torch.Tensor) -> np.ndarray:
        return (value.detach().to("cpu").numpy() * self.y_std + self.y_mean).astype(
            np.float32
        )

    def predict(
        self,
        X_test: np.ndarray,
        *,
        output_type: str = "mean",
        quantiles: list[float] | None = None,
    ):
        logits = self._predict_logits(X_test)
        output_type = str(output_type).lower()

        if output_type == "mean":
            return self._to_raw_space(self.bar_distribution.mean(logits))
        if output_type == "median":
            return self._to_raw_space(self.bar_distribution.median(logits))
        if output_type == "mode":
            return self._to_raw_space(self.bar_distribution.mode(logits))
        if output_type == "quantiles":
            if quantiles is None:
                quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            return [
                self._to_raw_space(self.bar_distribution.icdf(logits, float(q)))
                for q in quantiles
            ]
        if output_type == "full":
            return self.predict_distribution(X_test)
        raise ValueError(
            "output_type must be one of: mean, median, mode, quantiles, full"
        )

    def predict_distribution(
        self,
        X_test: np.ndarray,
        *,
        quantile_levels: np.ndarray | None = None,
        sharpness_levels: np.ndarray | None = None,
        y_true: np.ndarray | None = None,
    ) -> dict:
        logits = self._predict_logits(X_test)
        result = {
            "logits": logits.detach().to("cpu"),
            "y_pred_mean": self._to_raw_space(self.bar_distribution.mean(logits)),
            "y_pred_median": self._to_raw_space(self.bar_distribution.median(logits)),
            "y_pred_mode": self._to_raw_space(self.bar_distribution.mode(logits)),
        }
        if quantile_levels is not None:
            result["quantile_boundaries"] = np.stack(
                [
                    self._to_raw_space(self.bar_distribution.icdf(logits, float(level)))
                    for level in np.asarray(quantile_levels, dtype=np.float64)
                ],
                axis=0,
            ).astype(np.float32)
        if sharpness_levels is not None:
            result["sharpness_boundaries"] = np.stack(
                [
                    self._to_raw_space(self.bar_distribution.icdf(logits, float(level)))
                    for level in np.asarray(sharpness_levels, dtype=np.float64)
                ],
                axis=0,
            ).astype(np.float32)
        if y_true is not None:
            y_norm = (
                torch.as_tensor(np.asarray(y_true, dtype=np.float32), device=self.device)
                - float(self.y_mean)
            ) / float(self.y_std)
            result["pit_values"] = (
                self.bar_distribution.cdf(logits, y_norm).detach().to("cpu").numpy()
            ).astype(np.float64)
        return result
