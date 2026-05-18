"""Unit tests for MultimodalLLM (mocked — no model downloads)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from src.projectors.linear_projector import LinearProjector
from src.llm.backbone import MultimodalLLM, create_projector


@pytest.fixture
def random_image() -> Image.Image:
    """Synthetic PIL image."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestCreateProjector:
    """Tests for projector factory."""

    def test_create_mlp_projector(self) -> None:
        """MLP projector is built from a config-like object."""
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            type="mlp", hidden_dim=256, num_layers=2, dropout=0.0
        )
        proj = create_projector(cfg, in_dim=64, llm_dim=32)
        assert isinstance(proj, nn.Module)
        out = proj(torch.randn(1, 10, 64))
        assert out.shape == (1, 10, 32)


class TestMultimodalLLM:
    """Tests for MultimodalLLM with mocked Qwen2-VL stack."""

    @staticmethod
    def _mock_stack(
        llm_hidden: int = 32,
        vocab_size: int = 1000,
        num_patches: int = 16,
        enc_dim: int = 64,
    ) -> tuple[MagicMock, MagicMock, MagicMock, LinearProjector]:
        """Build mock encoder, LLM, processor, and matching projector."""
        encoder = MagicMock()
        encoder.encode_image.return_value = torch.randn(1, num_patches + 1, enc_dim)
        encoder.get_patch_features.return_value = torch.randn(1, num_patches, enc_dim)

        class TinyLanguageModel(nn.Module):
            """Minimal LM stub with real embeddings."""

            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, llm_hidden)

            def get_input_embeddings(self) -> nn.Embedding:
                return self.embed_tokens

            def generate(self, **kwargs: object) -> torch.Tensor:
                return torch.tensor([[10, 11, 12]])

        language_model = TinyLanguageModel()
        model_container = MagicMock()
        model_container.language_model = language_model
        llm = MagicMock()
        llm.model = model_container
        llm.generate.return_value = torch.tensor([[1, 2, 3, 10, 11, 12]])

        def tokenizer_call(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([[5, 6]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        tokenizer = MagicMock(side_effect=tokenizer_call)
        tokenizer.eos_token_id = 0
        tokenizer.decode.return_value = "mocked answer"

        processor = MagicMock()
        processor.tokenizer = tokenizer
        processor.apply_chat_template.return_value = "chat prompt"
        processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        projector = LinearProjector(in_dim=enc_dim, out_dim=llm_hidden)
        return encoder, llm, processor, projector

    @patch.object(MultimodalLLM, "_load_llm_and_processor")
    @patch("src.llm.backbone.AutoConfig")
    def test_prepare_inputs_shapes(
        self,
        mock_config_cls: MagicMock,
        mock_load: MagicMock,
        random_image: Image.Image,
    ) -> None:
        """prepare_inputs merges visual and text token dimensions."""
        enc_dim, llm_hidden, num_patches = 64, 32, 16
        encoder, llm, processor, projector = self._mock_stack(
            llm_hidden=llm_hidden, enc_dim=enc_dim, num_patches=num_patches
        )

        text_cfg = MagicMock()
        text_cfg.hidden_size = llm_hidden
        mock_config_cls.from_pretrained.return_value = MagicMock(
            get_text_config=MagicMock(return_value=text_cfg)
        )
        mock_load.return_value = (llm, processor)

        model = MultimodalLLM(
            encoder=encoder,
            projector=projector,
            llm_model_id="mock/model",
            device="cpu",
            load_in_4bit=False,
        )

        inputs = model.prepare_inputs(random_image, "What is in the image?")
        assert inputs["inputs_embeds"].shape[1] == num_patches + 2
        assert inputs["num_visual_tokens"] == num_patches
        assert inputs["num_text_tokens"] == 2

    @patch.object(MultimodalLLM, "_load_llm_and_processor")
    @patch("src.llm.backbone.AutoConfig")
    def test_generate_custom_calls_language_model(
        self,
        mock_config_cls: MagicMock,
        mock_load: MagicMock,
        random_image: Image.Image,
    ) -> None:
        """Custom generate path uses language_model.generate with inputs_embeds."""
        encoder, llm, processor, projector = self._mock_stack()

        text_cfg = MagicMock()
        text_cfg.hidden_size = 32
        mock_config_cls.from_pretrained.return_value = MagicMock(
            get_text_config=MagicMock(return_value=text_cfg)
        )
        mock_load.return_value = (llm, processor)

        model = MultimodalLLM(
            encoder=encoder,
            projector=projector,
            llm_model_id="mock/model",
            device="cpu",
            load_in_4bit=False,
        )

        language_model = llm.model.language_model
        with patch.object(language_model, "generate", return_value=torch.tensor([[10, 11, 12]])) as mock_gen:
            answer = model.generate(random_image, "Describe this.", max_new_tokens=8)
        assert answer == "mocked answer"
        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args.kwargs
        assert "inputs_embeds" in call_kwargs
        assert "attention_mask" in call_kwargs

    @patch("src.llm.backbone.AutoProcessor")
    @patch("src.llm.backbone.Qwen2VLForConditionalGeneration")
    @patch("src.llm.backbone.AutoConfig")
    def test_projector_dim_mismatch_raises(
        self,
        mock_config_cls: MagicMock,
        mock_llm_cls: MagicMock,
        mock_processor_cls: MagicMock,
    ) -> None:
        """Mismatched projector/LLM dims raise ValueError."""
        text_cfg = MagicMock()
        text_cfg.hidden_size = 32
        mock_config_cls.from_pretrained.return_value = MagicMock(
            get_text_config=MagicMock(return_value=text_cfg)
        )

        encoder = MagicMock()
        bad_projector = LinearProjector(in_dim=64, out_dim=16)

        with pytest.raises(ValueError, match="Projector out dim"):
            MultimodalLLM(
                encoder=encoder,
                projector=bad_projector,
                llm_model_id="mock/model",
                device="cpu",
                load_in_4bit=False,
            )
