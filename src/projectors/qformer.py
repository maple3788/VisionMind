"""Q-Former projector: learnable queries compress vision tokens for the LLM."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class QFormerLayer(nn.Module):
    """One Q-Former block: cross-attention, self-attention, and feed-forward."""

    def __init__(
        self,
        llm_dim: int,
        encoder_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Queries attend to frozen/passed encoder patch features (cross-attention).
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            kdim=encoder_dim,
            vdim=encoder_dim,
            dropout=dropout,
            batch_first=True,
        )
        # Queries exchange information with each other (self-attention).
        self.self_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim * 4, llm_dim),
        )
        self.norm_cross = nn.LayerNorm(llm_dim)
        self.norm_self = nn.LayerNorm(llm_dim)
        self.norm_ffn = nn.LayerNorm(llm_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,
        encoder_output: Tensor,
        return_cross_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Run one Q-Former layer.

        Args:
            queries: Learnable query tokens ``[batch, num_queries, llm_dim]``.
            encoder_output: Vision encoder output ``[batch, seq_len, encoder_dim]``.
            return_cross_attn: If True, return averaged cross-attention weights.

        Returns:
            Updated queries and optional attention weights ``[batch, num_queries, seq_len]``.
        """
        # Step 1: queries pull information from image patches.
        cross_out, cross_weights = self.cross_attn(
            query=queries,
            key=encoder_output,
            value=encoder_output,
            need_weights=return_cross_attn,
            average_attn_weights=True,
        )
        queries = self.norm_cross(queries + self.dropout(cross_out))

        # Step 2: queries refine their summaries jointly.
        self_out, _ = self.self_attn(
            query=queries,
            key=queries,
            value=queries,
            need_weights=False,
        )
        queries = self.norm_self(queries + self.dropout(self_out))

        # Step 3: position-wise transformation on each query token.
        queries = self.norm_ffn(queries + self.dropout(self.ffn(queries)))

        attn = cross_weights if return_cross_attn else None
        return queries, attn


class QFormer(nn.Module):
    """Q-Former compresses variable-length vision tokens to fixed query tokens.

    Learnable query vectors act as "questions" about the image. Cross-attention
    reads patch features; self-attention lets queries collaborate. Output length
    is always ``num_queries``, independent of input image resolution.

    Args:
        num_queries: Number of output tokens passed to the LLM.
        encoder_dim: Vision encoder hidden size.
        llm_dim: LLM hidden size for projected query tokens.
        num_heads: Attention head count per layer.
        num_layers: Number of stacked Q-Former blocks.
        dropout: Dropout probability.

    Example:
        >>> qformer = QFormer(num_queries=32, encoder_dim=1024, llm_dim=4096)
        >>> out = qformer(torch.randn(2, 257, 1024))  # [2, 32, 4096]
    """

    def __init__(
        self,
        num_queries: int,
        encoder_dim: int,
        llm_dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim

        # Shared learnable queries expanded per batch — NOT image patches.
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_dim) * 0.02)

        self.layers = nn.ModuleList(
            [
                QFormerLayer(
                    llm_dim=llm_dim,
                    encoder_dim=encoder_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(llm_dim)

    def forward(
        self,
        encoder_output: Tensor,
        return_cross_attn: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compress encoder tokens into a fixed set of LLM-ready query tokens.

        Args:
            encoder_output: Vision features ``[batch, seq_len, encoder_dim]``.
            return_cross_attn: If True, also return last-layer cross-attention.

        Returns:
            Tensor ``[batch, num_queries, llm_dim]``, or tuple with attention map.
        """
        batch_size = encoder_output.shape[0]
        queries = self.queries.expand(batch_size, -1, -1)

        last_attn: Tensor | None = None
        for layer in self.layers:
            queries, attn = layer(
                queries,
                encoder_output,
                return_cross_attn=return_cross_attn,
            )
            if attn is not None:
                last_attn = attn

        queries = self.output_norm(queries)

        if return_cross_attn and last_attn is not None:
            return queries, last_attn
        return queries
