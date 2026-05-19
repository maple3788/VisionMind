"""Unit tests for multimodal RAG retriever (mocked CLIP)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from src.pipeline.multimodal_qa import MultimodalQAPipeline
from src.pipeline.rag_retriever import MultimodalRetriever, format_retrieval_context


def _unit_vector(dim: int, index: int) -> torch.Tensor:
    """One-hot style unit vector for deterministic retrieval."""
    v = torch.zeros(1, dim)
    v[0, index % dim] = 1.0
    return v


@pytest.fixture
def mock_encoder() -> MagicMock:
    """CLIP encoder returning orthogonal unit vectors per call index."""
    encoder = MagicMock()
    dim = 8
    img_counter = {"n": 0}
    txt_counter = {"n": 0}

    def img_features(images: list) -> torch.Tensor:
        del images
        idx = img_counter["n"]
        img_counter["n"] += 1
        return _unit_vector(dim, idx)

    def txt_features(texts: list) -> torch.Tensor:
        del texts
        idx = txt_counter["n"]
        txt_counter["n"] += 1
        return _unit_vector(dim, idx + 4)

    encoder._projected_image_features.side_effect = img_features
    encoder._projected_text_features.side_effect = txt_features
    return encoder


@pytest.fixture
def sample_docs(tmp_path: Path) -> list[dict]:
    """Three documents with distinct solid-color images."""
    docs = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    captions = ["a red object", "a green object", "a blue object"]
    for i, (color, caption) in enumerate(zip(colors, captions)):
        path = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (32, 32), color).save(path)
        docs.append(
            {
                "image": path,
                "text": caption,
                "metadata": {"id": i},
            }
        )
    return docs


class TestMultimodalRetriever:
    """Tests for FAISS-backed retrieval."""

    def test_index_and_retrieve_text(
        self, mock_encoder: MagicMock, sample_docs: list[dict]
    ) -> None:
        """Text query returns ranked documents."""
        retriever = MultimodalRetriever(mock_encoder)
        retriever.index_documents(sample_docs)
        assert retriever.num_documents == 3

        results = retriever.retrieve(query_text="a green object", top_k=2)
        assert len(results) <= 2
        assert "score" in results[0]
        assert "text" in results[0]

    def test_retrieve_image(self, mock_encoder: MagicMock, sample_docs: list[dict]) -> None:
        """Image query returns documents."""
        retriever = MultimodalRetriever(mock_encoder)
        retriever.index_documents(sample_docs)
        query_img = Image.new("RGB", (32, 32), (0, 255, 0))
        results = retriever.retrieve(query_image=query_img, top_k=1)
        assert len(results) == 1

    def test_save_and_load_index(
        self,
        mock_encoder: MagicMock,
        sample_docs: list[dict],
        tmp_path: Path,
    ) -> None:
        """Index round-trips through disk."""
        retriever = MultimodalRetriever(mock_encoder)
        retriever.index_documents(sample_docs)
        index_dir = tmp_path / "index"
        retriever.save_index(index_dir)

        loaded = MultimodalRetriever(mock_encoder)
        loaded.load_index(index_dir)
        assert loaded.num_documents == 3
        results = loaded.retrieve(query_text="a blue object", top_k=1)
        assert len(results) == 1

    def test_retrieve_requires_query(self, mock_encoder: MagicMock, sample_docs: list[dict]) -> None:
        """Empty query raises ValueError."""
        retriever = MultimodalRetriever(mock_encoder)
        retriever.index_documents(sample_docs)
        with pytest.raises(ValueError, match="query_image"):
            retriever.retrieve()


class TestFormatContext:
    """Tests for context formatting."""

    def test_format_retrieval_context(self) -> None:
        """Context includes document text and scores."""
        docs = [{"rank": 1, "score": 0.9, "text": "hello", "metadata": {"id": 1}}]
        ctx = format_retrieval_context(docs)
        assert "hello" in ctx
        assert "0.900" in ctx or "0.9" in ctx


class TestPipelineRAG:
    """Tests for RAG integration in MultimodalQAPipeline."""

    def test_answer_with_rag_prepends_context(
        self, mock_encoder: MagicMock, sample_docs: list[dict], tmp_path: Path
    ) -> None:
        """Pipeline passes RAG-augmented prompt to the model."""
        from unittest.mock import patch

        retriever = MultimodalRetriever(mock_encoder)
        retriever.index_documents(sample_docs)

        cfg_path = tmp_path / "model_config.yaml"
        cfg_path.write_text("llm:\n  max_new_tokens: 64\n", encoding="utf-8")

        with patch("src.pipeline.multimodal_qa.MultimodalLLM") as mock_cls:
            mock_model = MagicMock()
            mock_model.generate.return_value = "rag answer"
            mock_cls.from_config.return_value = mock_model

            pipe = MultimodalQAPipeline(cfg_path, device="cpu", retriever=retriever)
            pipe.answer("What color?", use_rag=True)

        prompt = mock_model.generate.call_args.kwargs["prompt"]
        assert "reference documents" in prompt.lower() or "Doc" in prompt
        assert "What color?" in prompt
