"""Unit tests for vision encoders."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.encoders.clip_encoder import CLIPVisionEncoder
from src.encoders.vit_encoder import ViTEncoder, resolve_device


def _random_pil_image(size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Create a synthetic RGB PIL image for tests."""
    array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(array)


@pytest.fixture
def dog_image() -> Image.Image:
    """Synthetic stand-in for a dog image."""
    return _random_pil_image()


class TestResolveDevice:
    """Tests for device resolution helper."""

    def test_resolve_explicit_cpu(self) -> None:
        """Explicit cpu string resolves to CPU device."""
        assert resolve_device("cpu").type == "cpu"

    def test_resolve_auto_returns_device(self) -> None:
        """Auto mode returns a valid torch device."""
        device = resolve_device("auto")
        assert device.type in {"cpu", "cuda", "mps"}


class TestViTEncoder:
    """Tests for ViTEncoder with mocked HuggingFace models."""

    @staticmethod
    def _build_vit_mocks(
        mock_processor_cls: MagicMock,
        mock_model_cls: MagicMock,
        hidden_size: int = 768,
        num_tokens: int = 197,
        batch_size: int = 1,
    ) -> MagicMock:
        """Configure ViT processor/model mocks."""
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "pixel_values": torch.randn(batch_size, 3, 224, 224),
        }
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(batch_size, num_tokens, hidden_size)

        mock_model = MagicMock()
        mock_model.config.hidden_size = hidden_size
        mock_model.config.patch_size = 16
        mock_model.config.image_size = 224
        mock_model.parameters.return_value = []
        mock_model.to.return_value = mock_model
        mock_model.return_value = mock_output
        mock_model_cls.from_pretrained.return_value = mock_model
        return mock_model

    @patch("src.encoders.vit_encoder.ViTModel")
    @patch("src.encoders.vit_encoder.ViTImageProcessor")
    def test_encode_output_shape(
        self,
        mock_processor_cls: MagicMock,
        mock_model_cls: MagicMock,
        dog_image: Image.Image,
    ) -> None:
        """encode() returns [batch, num_patches+1, hidden_size]."""
        hidden_size = 768
        num_tokens = 197
        batch_size = 1
        self._build_vit_mocks(
            mock_processor_cls, mock_model_cls, hidden_size, num_tokens, batch_size
        )

        encoder = ViTEncoder(model_id="google/vit-base-patch16-224", device="cpu")
        output = encoder.encode([dog_image])

        assert output.shape == (batch_size, num_tokens, hidden_size)

    @patch("src.encoders.vit_encoder.ViTModel")
    @patch("src.encoders.vit_encoder.ViTImageProcessor")
    def test_cls_and_patch_feature_shapes(
        self,
        mock_processor_cls: MagicMock,
        mock_model_cls: MagicMock,
        dog_image: Image.Image,
    ) -> None:
        """CLS and patch helpers return expected rank-2 tensors."""
        hidden_size = 768
        num_tokens = 197
        self._build_vit_mocks(mock_processor_cls, mock_model_cls, hidden_size, num_tokens)

        encoder = ViTEncoder(device="cpu")
        cls_feat = encoder.get_cls_feature([dog_image])
        patch_feat = encoder.get_patch_features([dog_image])

        assert cls_feat.shape == (1, hidden_size)
        assert patch_feat.shape == (1, num_tokens - 1, hidden_size)


class TestCLIPVisionEncoder:
    """Tests for CLIPVisionEncoder with mocked HuggingFace models."""

    @staticmethod
    def _build_clip_mocks(
        mock_processor_cls: MagicMock,
        mock_model_cls: MagicMock,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> MagicMock:
        """Configure shared CLIP mocks for image/text encoding."""
        num_images = image_features.shape[0]
        num_texts = text_features.shape[0]

        mock_processor = MagicMock()
        mock_processor.return_value = {
            "pixel_values": torch.randn(num_images, 3, 224, 224),
            "input_ids": torch.ones(num_texts, 5, dtype=torch.long),
            "attention_mask": torch.ones(num_texts, 5, dtype=torch.long),
        }
        mock_processor_cls.from_pretrained.return_value = mock_processor

        vision_output = MagicMock()
        vision_output.last_hidden_state = torch.randn(num_images, 257, 1024)

        text_output = MagicMock()
        text_output.last_hidden_state = torch.randn(num_texts, 5, 1024)

        mock_model = MagicMock()
        mock_model.config.vision_config.hidden_size = 1024
        mock_model.config.vision_config.patch_size = 14
        mock_model.config.vision_config.image_size = 224
        mock_model.parameters.return_value = []
        mock_model.to.return_value = mock_model
        mock_model.vision_model.return_value = vision_output
        mock_model.text_model.return_value = text_output
        mock_model.get_image_features.return_value = image_features
        mock_model.get_text_features.return_value = text_features
        mock_model_cls.from_pretrained.return_value = mock_model
        return mock_model

    @patch("src.encoders.clip_encoder.CLIPModel")
    @patch("src.encoders.clip_encoder.CLIPProcessor")
    def test_encode_image_output_shape(
        self,
        mock_processor_cls: MagicMock,
        mock_model_cls: MagicMock,
        dog_image: Image.Image,
    ) -> None:
        """encode_image() returns [batch, seq_len, hidden_size]."""
        image_features = torch.randn(1, 768)
        text_features = torch.randn(1, 768)
        self._build_clip_mocks(mock_processor_cls, mock_model_cls, image_features, text_features)

        encoder = CLIPVisionEncoder(device="cpu")
        output = encoder.encode_image([dog_image])

        assert output.shape == (1, 257, 1024)

    @patch("src.encoders.clip_encoder.CLIPModel")
    @patch("src.encoders.clip_encoder.CLIPProcessor")
    def test_dog_image_text_similarity_above_threshold(
        self,
        mock_processor_cls: MagicMock,
        mock_model_cls: MagicMock,
        dog_image: Image.Image,
    ) -> None:
        """Matching image-text pair should exceed cosine similarity 0.2."""
        image_features = torch.tensor([[1.0, 0.0, 0.0]])
        text_dog = torch.tensor([[0.95, 0.31, 0.0]])
        text_cat = torch.tensor([[0.0, 1.0, 0.0]])

        self._build_clip_mocks(
            mock_processor_cls,
            mock_model_cls,
            image_features,
            torch.stack([text_dog.squeeze(0), text_cat.squeeze(0)]),
        )

        encoder = CLIPVisionEncoder(device="cpu")
        sim_matrix = encoder.compute_similarity([dog_image], ["a dog", "a cat"])

        assert sim_matrix.shape == (1, 2)
        assert sim_matrix[0, 0].item() > 0.2
        assert sim_matrix[0, 0].item() > sim_matrix[0, 1].item()

    @patch("src.encoders.clip_encoder.CLIPModel")
    @patch("src.encoders.clip_encoder.CLIPProcessor")
    def test_compute_similarity_matrix_shape(
        self,
        mock_processor_cls: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        """Similarity matrix shape matches batch dimensions."""
        images = [_random_pil_image(), _random_pil_image()]
        texts = ["label a", "label b", "label c"]

        image_features = torch.randn(2, 64)
        text_features = torch.randn(3, 64)
        self._build_clip_mocks(mock_processor_cls, mock_model_cls, image_features, text_features)

        encoder = CLIPVisionEncoder(device="cpu")
        sim_matrix = encoder.compute_similarity(images, texts)

        assert sim_matrix.shape == (2, 3)


class TestAsClipEmbeddingTensor:
    """Tests for CLIP feature tensor extraction (HF ModelOutput compatibility)."""

    def test_passes_through_tensor(self) -> None:
        """Raw tensors are returned unchanged."""
        from src.encoders.clip_encoder import _as_clip_embedding_tensor

        x = torch.randn(2, 768)
        assert torch.equal(_as_clip_embedding_tensor(x), x)

    def test_extracts_pooler_output(self) -> None:
        """BaseModelOutputWithPooling.pooler_output is used when present."""
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        from src.encoders.clip_encoder import _as_clip_embedding_tensor

        pooled = torch.randn(3, 512)
        out = BaseModelOutputWithPooling(
            pooler_output=pooled,
            last_hidden_state=torch.randn(3, 10, 512),
        )
        got = _as_clip_embedding_tensor(out)
        assert torch.equal(got, pooled)

    def test_falls_back_to_cls_from_last_hidden(self) -> None:
        """When pooler_output is None, use first sequence token."""
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        from src.encoders.clip_encoder import _as_clip_embedding_tensor

        last = torch.randn(2, 5, 128)
        out = BaseModelOutputWithPooling(pooler_output=None, last_hidden_state=last)
        got = _as_clip_embedding_tensor(out)
        assert torch.equal(got, last[:, 0, :])
