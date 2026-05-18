"""MLP projector from vision encoder space to LLM token space."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MLPProjector(nn.Module):
    """Multi-layer perceptron projector with GELU and LayerNorm.

    Maps each token independently while adding non-linear capacity between
    encoder and LLM hidden dimensions.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden layer width.
        out_dim: Output feature dimension.
        num_layers: Number of linear blocks (minimum 1).
        dropout: Dropout probability applied after each activation.

    Example:
        >>> proj = MLPProjector(1024, 2048, 4096, num_layers=2)
        >>> out = proj(torch.randn(2, 196, 1024))  # [2, 196, 4096]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for idx in range(num_layers):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(dims[idx + 1]))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Project encoder tokens through the MLP stack.

        Args:
            x: Encoder output of shape ``[batch, seq_len, in_dim]``.

        Returns:
            Projected tensor of shape ``[batch, seq_len, out_dim]``.
        """
        return self.out_norm(self.mlp(x))
