"""Linear projector from vision encoder space to LLM token space."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LinearProjector(nn.Module):
    """Single linear map from encoder hidden size to LLM hidden size.

    Applies the same projection to every token in the sequence, preserving
    batch and sequence dimensions.

    Args:
        in_dim: Input feature dimension (e.g. CLIP hidden size 1024).
        out_dim: Output feature dimension (e.g. LLM hidden size 4096).
        use_layer_norm: If True, apply LayerNorm on the projected features.

    Example:
        >>> proj = LinearProjector(1024, 4096)
        >>> out = proj(torch.randn(2, 196, 1024))  # [2, 196, 4096]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim) if use_layer_norm else None

    def forward(self, x: Tensor) -> Tensor:
        """Project encoder tokens into LLM token space.

        Args:
            x: Encoder output of shape ``[batch, seq_len, in_dim]``.

        Returns:
            Projected tensor of shape ``[batch, seq_len, out_dim]``.
        """
        x = self.linear(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x
