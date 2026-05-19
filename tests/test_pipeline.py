"""Unit tests for MultimodalQAPipeline (mocked — no model downloads)."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.pipeline.multimodal_qa import ImageLoadError, MultimodalQAPipeline


@pytest.fixture
def random_image() -> Image.Image:
    """Synthetic RGB PIL image."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def pipeline(tmp_path: Path) -> MultimodalQAPipeline:
    """Pipeline with mocked MultimodalLLM."""
    cfg_path = tmp_path / "model_config.yaml"
    cfg_path.write_text(
        "llm:\n  max_new_tokens: 64\npipeline:\n  max_new_tokens: 32\n",
        encoding="utf-8",
    )
    with patch("src.pipeline.multimodal_qa.MultimodalLLM") as mock_cls:
        mock_model = MagicMock()
        mock_model.generate.return_value = "pipeline answer"
        mock_cls.from_config.return_value = mock_model
        pipe = MultimodalQAPipeline(cfg_path, device="cpu")
    pipe.model = mock_model
    return pipe


class TestLoadImage:
    """Tests for flexible image input handling."""

    def test_load_pil_image(self, pipeline: MultimodalQAPipeline, random_image: Image.Image) -> None:
        """PIL images are converted to RGB."""
        out = pipeline._load_image(random_image)
        assert out.mode == "RGB"
        assert out.size == random_image.size

    def test_load_file_path(
        self, pipeline: MultimodalQAPipeline, random_image: Image.Image, tmp_path: Path
    ) -> None:
        """File paths load correctly."""
        path = tmp_path / "test.png"
        random_image.save(path)
        out = pipeline._load_image(str(path))
        assert out.size == random_image.size

    def test_load_base64(
        self, pipeline: MultimodalQAPipeline, random_image: Image.Image
    ) -> None:
        """Base64 strings decode to images."""
        buf = BytesIO()
        random_image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        out = pipeline._load_image(encoded)
        assert out.size == random_image.size

    def test_load_base64_data_uri(
        self, pipeline: MultimodalQAPipeline, random_image: Image.Image
    ) -> None:
        """Data-URI base64 payloads are supported."""
        buf = BytesIO()
        random_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        uri = f"data:image/png;base64,{b64}"
        out = pipeline._load_image(uri)
        assert out.mode == "RGB"

    def test_invalid_path_raises(self, pipeline: MultimodalQAPipeline) -> None:
        """Missing files raise ImageLoadError."""
        with pytest.raises(ImageLoadError, match="Not a valid"):
            pipeline._load_image("/nonexistent/path/image.png")

    @patch("src.pipeline.multimodal_qa.requests.get")
    def test_load_url(
        self,
        mock_get: MagicMock,
        pipeline: MultimodalQAPipeline,
        random_image: Image.Image,
    ) -> None:
        """HTTP URLs fetch image bytes."""
        buf = BytesIO()
        random_image.save(buf, format="PNG")
        mock_resp = MagicMock()
        mock_resp.content = buf.getvalue()
        mock_resp.headers = {"Content-Type": "image/png"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        out = pipeline._load_image("https://example.com/photo.png")
        assert out.size == random_image.size
        mock_get.assert_called_once()

    @patch("src.pipeline.multimodal_qa.requests.get")
    def test_load_url_html_raises(
        self, mock_get: MagicMock, pipeline: MultimodalQAPipeline
    ) -> None:
        """HTML responses from URLs are rejected."""
        mock_resp = MagicMock()
        mock_resp.content = b"<html></html>"
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(ImageLoadError, match="not an image"):
            pipeline._load_image("https://example.com/page")


class TestBuildPrompt:
    """Tests for prompt construction."""

    def test_question_only(self, pipeline: MultimodalQAPipeline) -> None:
        """Single question passes through unchanged."""
        assert pipeline._build_prompt("What is this?", None) == "What is this?"

    def test_with_history(self, pipeline: MultimodalQAPipeline) -> None:
        """History is formatted with roles and trailing Assistant prompt."""
        history = [
            {"role": "user", "content": "What animal?"},
            {"role": "assistant", "content": "A dog."},
        ]
        prompt = pipeline._build_prompt("What color is it?", history)
        assert "User: What animal?" in prompt
        assert "Assistant: A dog." in prompt
        assert "User: What color is it?" in prompt
        assert prompt.endswith("Assistant:")


class TestAnswer:
    """Tests for the public answer API."""

    def test_answer_text_only(self, pipeline: MultimodalQAPipeline) -> None:
        """Text-only questions call generate without an image."""
        result = pipeline.answer("Hello?", image=None)
        assert result == "pipeline answer"
        pipeline.model.generate.assert_called_once()
        call_kwargs = pipeline.model.generate.call_args.kwargs
        assert call_kwargs["image"] is None
        assert call_kwargs["prompt"] == "Hello?"

    def test_answer_with_pil_image(
        self, pipeline: MultimodalQAPipeline, random_image: Image.Image
    ) -> None:
        """PIL images are passed to the model."""
        pipeline.answer("Describe.", image=random_image)
        assert pipeline.model.generate.call_args.kwargs["image"] is not None

    def test_answer_with_history(self, pipeline: MultimodalQAPipeline) -> None:
        """History is folded into the prompt."""
        history = [{"role": "user", "content": "Hi"}]
        pipeline.answer("Follow up?", history=history)
        prompt = pipeline.model.generate.call_args.kwargs["prompt"]
        assert "User: Hi" in prompt
        assert "Follow up?" in prompt

    def test_empty_question_raises(self, pipeline: MultimodalQAPipeline) -> None:
        """Blank questions are rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            pipeline.answer("   ")

    def test_bad_image_raises(
        self, pipeline: MultimodalQAPipeline, random_image: Image.Image
    ) -> None:
        """Invalid image paths surface as ImageLoadError."""
        with pytest.raises(ImageLoadError):
            pipeline.answer("Q?", image="/no/such/file.jpg")

    def test_streaming_delegates(self, pipeline: MultimodalQAPipeline) -> None:
        """stream=True is forwarded to the model."""
        pipeline.model.generate.return_value = iter(["chunk"])
        result = pipeline.answer("Hi", stream=True)
        assert list(result) == ["chunk"]
        assert pipeline.model.generate.call_args.kwargs["stream"] is True
