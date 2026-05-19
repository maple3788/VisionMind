"""OpenAI-compatible FastAPI server for VisionMind multimodal QA."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel, Field

from src.pipeline.multimodal_qa import MultimodalQAPipeline

load_dotenv()

DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "visionmind-qwen2-vl")
DEFAULT_CONFIG = os.getenv("MODEL_CONFIG_PATH", "config/model_config.yaml")
API_KEY = os.getenv("API_KEY", "")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEVICE = os.getenv("DEVICE", "auto")
USE_RAG = os.getenv("USE_RAG", "false").lower() in ("1", "true", "yes")

_bearer = HTTPBearer(auto_error=False)

_pipeline: MultimodalQAPipeline | None = None


class ChatMessage(BaseModel):
    """OpenAI-style chat message."""

    role: str
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    """Request body for ``/v1/chat/completions``."""

    model: str = DEFAULT_MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ModelCard(BaseModel):
    """OpenAI model list entry."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "visionmind"


def get_pipeline() -> MultimodalQAPipeline:
    """Return the singleton pipeline, loading on first use."""
    global _pipeline
    if _pipeline is None:
        config_path = Path(DEFAULT_CONFIG)
        if not config_path.is_file():
            raise RuntimeError(f"Model config not found: {config_path}")

        retriever = None
        if USE_RAG:
            from src.encoders.clip_encoder import CLIPVisionEncoder
            from src.pipeline.rag_retriever import MultimodalRetriever

            rag_path = os.getenv("RAG_INDEX_PATH", "data/rag_index")
            encoder = CLIPVisionEncoder(device=DEVICE)
            retriever = MultimodalRetriever(encoder)
            if Path(rag_path).is_dir():
                retriever.load_index(rag_path)
                logger.info("Loaded RAG index from {}", rag_path)
            else:
                logger.warning("RAG enabled but index missing at {}", rag_path)

        logger.info("Loading MultimodalQAPipeline from {}", config_path)
        _pipeline = MultimodalQAPipeline(
            config_path=config_path,
            device=DEVICE,
            encoder_on_cpu=DEVICE == "mps",
            retriever=retriever,
        )
    return _pipeline


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """Validate Bearer token when ``API_KEY`` is configured."""
    if not API_KEY:
        return
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _content_to_text(content: str | list[dict[str, Any]]) -> str:
    """Extract plain text from message content."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(p for p in parts if p).strip()


def _extract_image_url(content: str | list[dict[str, Any]]) -> Optional[str]:
    """Extract the first image URL or data-URI from multimodal content."""
    if isinstance(content, str):
        return None
    for block in content:
        if block.get("type") == "image_url":
            image_url = block.get("image_url", {})
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if url:
                    return str(url)
    return None


def parse_chat_messages(
    messages: list[ChatMessage],
) -> tuple[str, Optional[str], list[dict[str, str]]]:
    """Parse OpenAI messages into question, image, and history.

    Args:
        messages: Chat messages from the API request.

    Returns:
        Tuple of (question, image_url_or_data_uri, history).

    Raises:
        HTTPException: If messages are empty or last turn is not from user.
    """
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    history: list[dict[str, str]] = []
    for msg in messages[:-1]:
        if msg.role in ("user", "assistant"):
            history.append({"role": msg.role, "content": _content_to_text(msg.content)})

    last = messages[-1]
    if last.role != "user":
        raise HTTPException(status_code=400, detail="Last message must have role 'user'")

    question = _content_to_text(last.content)
    if not question:
        raise HTTPException(status_code=400, detail="User message must include text")

    image_ref = _extract_image_url(last.content)
    return question, image_ref, history


def _chat_completion_response(
    model: str,
    content: str,
    completion_id: str | None = None,
) -> dict[str, Any]:
    """Build a non-streaming OpenAI chat completion payload."""
    return {
        "id": completion_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _stream_chunk(model: str, content: str, completion_id: str) -> str:
    """Format one SSE chunk."""
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_answer(
    pipeline: MultimodalQAPipeline,
    question: str,
    image: Optional[str],
    history: list[dict[str, str]],
    model: str,
    max_tokens: Optional[int],
    use_rag: bool,
) -> AsyncIterator[str]:
    """Yield SSE events for a streaming completion."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    result = pipeline.answer(
        question,
        image=image,
        history=history or None,
        stream=True,
        max_new_tokens=max_tokens,
        use_rag=use_rag,
    )

    if isinstance(result, str):
        yield _stream_chunk(model, result, completion_id)
    else:
        for chunk in result:
            if chunk:
                yield _stream_chunk(model, str(chunk), completion_id)

    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="VisionMind API",
        description="OpenAI-compatible multimodal QA API",
        version="1.0.0",
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Log request method, path, and latency."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "{} {} → {} ({:.1f}ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "model": DEFAULT_MODEL_ID}

    @app.get("/v1/models")
    async def list_models(_: None = Depends(verify_api_key)) -> dict[str, Any]:
        """List available models (OpenAI-compatible)."""
        return {
            "object": "list",
            "data": [ModelCard(id=DEFAULT_MODEL_ID).model_dump()],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest,
        _: None = Depends(verify_api_key),
    ) -> Response:
        """OpenAI-compatible chat completions with optional image input."""
        question, image_ref, history = parse_chat_messages(body.messages)
        logger.info(
            "chat/completions | model={} | stream={} | has_image={}",
            body.model,
            body.stream,
            image_ref is not None,
        )

        try:
            pipeline = get_pipeline()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        use_rag = USE_RAG and pipeline.retriever is not None

        if body.stream:
            return StreamingResponse(
                _stream_answer(
                    pipeline,
                    question,
                    image_ref,
                    history,
                    body.model,
                    body.max_tokens,
                    use_rag,
                ),
                media_type="text/event-stream",
            )

        try:
            answer = pipeline.answer(
                question,
                image=image_ref,
                history=history or None,
                max_new_tokens=body.max_tokens,
                use_rag=use_rag,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if not isinstance(answer, str):
            answer = "".join(answer)

        return JSONResponse(_chat_completion_response(body.model, answer))

    return app


app = create_app()


def main() -> None:
    """Run the API server with uvicorn."""
    import uvicorn

    uvicorn.run(
        "src.serving.api_server:app",
        host=API_HOST,
        port=API_PORT,
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
