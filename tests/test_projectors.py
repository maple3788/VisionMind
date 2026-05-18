"""Unit tests for input projectors."""

from __future__ import annotations

import pytest
import torch

from src.projectors.linear_projector import LinearProjector
from src.projectors.mlp_projector import MLPProjector
from src.projectors.qformer import QFormer


@pytest.fixture
def encoder_tokens() -> torch.Tensor:
    """Dummy CLIP-like encoder output."""
    return torch.randn(2, 196, 1024)


class TestLinearProjector:
    """Tests for LinearProjector."""

    def test_forward_shape(self, encoder_tokens: torch.Tensor) -> None:
        """Linear projector preserves batch/seq and maps hidden dim."""
        projector = LinearProjector(in_dim=1024, out_dim=4096)
        output = projector(encoder_tokens)
        assert output.shape == (2, 196, 4096)

    def test_gradients_flow(self, encoder_tokens: torch.Tensor) -> None:
        """Backward pass updates linear weights."""
        projector = LinearProjector(in_dim=1024, out_dim=4096)
        output = projector(encoder_tokens)
        loss = output.sum()
        loss.backward()
        assert projector.linear.weight.grad is not None


class TestMLPProjector:
    """Tests for MLPProjector."""

    def test_forward_shape(self, encoder_tokens: torch.Tensor) -> None:
        """MLP projector maps to LLM hidden size."""
        projector = MLPProjector(
            in_dim=1024,
            hidden_dim=2048,
            out_dim=4096,
            num_layers=2,
        )
        output = projector(encoder_tokens)
        assert output.shape == (2, 196, 4096)

    def test_gradients_flow(self, encoder_tokens: torch.Tensor) -> None:
        """Backward pass runs without error."""
        projector = MLPProjector(1024, 2048, 4096, num_layers=2, dropout=0.1)
        output = projector(encoder_tokens)
        loss = output.pow(2).mean()
        loss.backward()
        assert any(p.grad is not None for p in projector.parameters())

    def test_num_layers_must_be_positive(self) -> None:
        """num_layers < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_layers"):
            MLPProjector(1024, 2048, 4096, num_layers=0)


class TestQFormer:
    """Tests for QFormer."""

    def test_output_shape_fixed_queries(self, encoder_tokens: torch.Tensor) -> None:
        """Output length equals num_queries regardless of input seq len."""
        qformer = QFormer(
            num_queries=32,
            encoder_dim=1024,
            llm_dim=4096,
            num_heads=8,
            num_layers=2,
        )
        output = qformer(encoder_tokens)
        assert output.shape == (2, 32, 4096)

    def test_different_input_seq_len_same_output_len(self) -> None:
        """Shorter or longer patch sequences still yield num_queries tokens."""
        qformer = QFormer(num_queries=16, encoder_dim=1024, llm_dim=4096, num_layers=1)

        short_seq = torch.randn(1, 50, 1024)
        long_seq = torch.randn(1, 400, 1024)

        out_short = qformer(short_seq)
        out_long = qformer(long_seq)

        assert out_short.shape == (1, 16, 4096)
        assert out_long.shape == (1, 16, 4096)

    def test_gradients_flow(self, encoder_tokens: torch.Tensor) -> None:
        """Learnable queries receive gradients."""
        qformer = QFormer(num_queries=8, encoder_dim=1024, llm_dim=512, num_layers=1)
        output = qformer(encoder_tokens)
        loss = output.mean()
        loss.backward()
        assert qformer.queries.grad is not None

    def test_return_cross_attention_weights(self, encoder_tokens: torch.Tensor) -> None:
        """Optional cross-attention map aligns with batch and sequence."""
        qformer = QFormer(num_queries=8, encoder_dim=1024, llm_dim=512, num_layers=1)
        output, attn = qformer(encoder_tokens, return_cross_attn=True)
        assert output.shape == (2, 8, 512)
        assert attn.shape == (2, 8, 196)
