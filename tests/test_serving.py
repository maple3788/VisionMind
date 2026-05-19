"""Unit tests for FastAPI serving layer (mocked pipeline)."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.serving.api_server import ChatMessage, app, parse_chat_messages


@pytest.fixture(autouse=True)
def reset_pipeline() -> None:
    """Reset global pipeline between tests."""
    import src.serving.api_server as api_mod

    api_mod._pipeline = None
    yield
    api_mod._pipeline = None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Test client with API key auth enabled."""
    monkeypatch.setenv("API_KEY", "test-secret-key")
    import src.serving.api_server as api_mod

    api_mod.API_KEY = "test-secret-key"
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-secret-key"}


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Mock pipeline returned by get_pipeline."""
    pipe = MagicMock()
    pipe.answer.return_value = "mocked assistant reply"
    pipe.retriever = None
    return pipe


class TestParseMessages:
    """Tests for OpenAI message parsing."""

    def test_text_only(self) -> None:
        """Plain string content is parsed."""
        msgs = [
            ChatMessage(role="user", content="What is this?"),
        ]
        q, img, hist = parse_chat_messages(msgs)
        assert q == "What is this?"
        assert img is None
        assert hist == []

    def test_multimodal_content(self) -> None:
        """Image URL blocks are extracted."""
        buf = BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        uri = f"data:image/png;base64,{b64}"
        msgs = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Describe the image."},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            ),
        ]
        q, img, _ = parse_chat_messages(msgs)
        assert "Describe" in q
        assert img is not None
        assert img.startswith("data:image")


class TestAPIEndpoints:
    """HTTP endpoint tests."""

    def test_health_no_auth(self, client: TestClient) -> None:
        """Health check is public."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_models_requires_auth(self, client: TestClient) -> None:
        """Models endpoint rejects missing key."""
        resp = client.get("/v1/models")
        assert resp.status_code == 401

    def test_models_with_auth(self, client: TestClient, auth_headers: dict) -> None:
        """Models list returns visionmind model."""
        resp = client.get("/v1/models", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) >= 1

    @patch("src.serving.api_server.get_pipeline")
    def test_chat_completions(
        self,
        mock_get: MagicMock,
        client: TestClient,
        auth_headers: dict,
        mock_pipeline: MagicMock,
    ) -> None:
        """Chat completion returns OpenAI-shaped JSON."""
        mock_get.return_value = mock_pipeline
        payload = {
            "model": "visionmind-qwen2-vl",
            "messages": [{"role": "user", "content": "Hello?"}],
        }
        resp = client.post("/v1/chat/completions", json=payload, headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "mocked assistant reply"
        mock_pipeline.answer.assert_called_once()

    @patch("src.serving.api_server.get_pipeline")
    def test_chat_completions_stream(
        self,
        mock_get: MagicMock,
        client: TestClient,
        auth_headers: dict,
        mock_pipeline: MagicMock,
    ) -> None:
        """Streaming returns SSE chunks."""
        mock_pipeline.answer.return_value = iter(["Hello", " world"])
        mock_get.return_value = mock_pipeline

        payload = {
            "model": "visionmind-qwen2-vl",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        with client.stream(
            "POST", "/v1/chat/completions", json=payload, headers=auth_headers
        ) as resp:
            assert resp.status_code == 200
            text = "".join(resp.iter_text())
        assert "data:" in text
        assert "[DONE]" in text
