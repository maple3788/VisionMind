"""End-to-end multimodal QA pipeline (Phase 4)."""

from __future__ import annotations

import base64
import binascii
import re
from io import BytesIO
from pathlib import Path
from typing import Iterator, Optional, Union

import requests
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, UnidentifiedImageError

from src.llm.backbone import HistoryTurn, MultimodalLLM
from src.pipeline.rag_retriever import MultimodalRetriever, format_retrieval_context

ImageInput = Union[Image.Image, str, bytes, Path, None]

_DEFAULT_USER_AGENT = (
    "VisionMind/1.0 (multimodal-qa; educational project; +https://github.com/)"
)


class ImageLoadError(ValueError):
    """Raised when an image cannot be loaded from the given input."""


class MultimodalQAPipeline:
    """Unified interface for multimodal question answering.

    Wraps :class:`~src.llm.backbone.MultimodalLLM` with flexible image inputs,
    conversation history, optional streaming, and text-only fallback.

    Args:
        config_path: Path to ``config/model_config.yaml``.
        device: Compute device (``auto``, ``cuda``, ``mps``, ``cpu``).
        encoder_on_cpu: Offload vision encoder and projector to CPU.
        use_native: Use Qwen2-VL native vision path instead of custom CLIP+projector.
        request_timeout_sec: HTTP timeout when loading images from URLs.
        retriever: Optional ``MultimodalRetriever`` for RAG-augmented answers.
        rag_top_k: Number of documents to retrieve when RAG is enabled.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        device: str = "auto",
        encoder_on_cpu: bool = False,
        use_native: bool = False,
        request_timeout_sec: float = 30.0,
        retriever: Optional[MultimodalRetriever] = None,
        rag_top_k: int = 3,
    ) -> None:
        self.config_path = Path(config_path)
        self.use_native = use_native
        self.request_timeout_sec = request_timeout_sec
        self.retriever = retriever
        self.rag_top_k = rag_top_k

        cfg = OmegaConf.load(self.config_path)
        pipeline_cfg = getattr(cfg, "pipeline", OmegaConf.create({}))
        self.max_new_tokens = int(
            getattr(pipeline_cfg, "max_new_tokens", cfg.llm.max_new_tokens)
        )
        if getattr(pipeline_cfg, "use_native", None) is not None:
            self.use_native = bool(pipeline_cfg.use_native)
        if getattr(pipeline_cfg, "encoder_on_cpu", None) is not None:
            encoder_on_cpu = bool(pipeline_cfg.encoder_on_cpu)
        if getattr(pipeline_cfg, "rag_top_k", None) is not None:
            self.rag_top_k = int(pipeline_cfg.rag_top_k)

        logger.info("Initializing MultimodalQAPipeline from {}", self.config_path)
        self.model = MultimodalLLM.from_config(
            self.config_path,
            device=device,
            encoder_on_cpu=encoder_on_cpu,
        )

    def answer(
        self,
        question: str,
        image: ImageInput = None,
        history: Optional[list[HistoryTurn]] = None,
        stream: bool = False,
        max_new_tokens: Optional[int] = None,
        use_native: Optional[bool] = None,
        use_rag: Optional[bool] = None,
        rag_top_k: Optional[int] = None,
    ) -> str | Iterator[str]:
        """Answer a question, optionally conditioned on an image.

        Args:
            question: User question or instruction.
            image: Optional image as PIL, file path, URL, or base64 string/bytes.
            history: Prior turns ``{"role": "user"|"assistant", "content": str}``.
            stream: If True, yield decoded text chunks (when supported).
            max_new_tokens: Override default generation length.
            use_native: Override pipeline default for native vs custom path.
            use_rag: If True and a retriever is set, augment the prompt with context.
            rag_top_k: Override number of retrieved documents.

        Returns:
            Answer string, or a generator when ``stream=True``.

        Raises:
            ImageLoadError: If ``image`` is provided but cannot be loaded.
            ValueError: If ``question`` is empty.
        """
        question = question.strip()
        if not question:
            raise ValueError("question must be a non-empty string")

        pil_image: Optional[Image.Image] = None
        if image is not None:
            try:
                pil_image = self._load_image(image)
            except (ImageLoadError, OSError, UnidentifiedImageError) as exc:
                raise ImageLoadError(f"Failed to load image: {exc}") from exc

        should_rag = use_rag if use_rag is not None else self.retriever is not None
        if should_rag and self.retriever is not None:
            question = self._augment_with_rag(
                question,
                pil_image,
                top_k=rag_top_k or self.rag_top_k,
            )

        prompt = self._build_prompt(question, history)
        native = self.use_native if use_native is None else use_native
        tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        logger.info(
            "answer | has_image={} | history_turns={} | stream={} | native={} | rag={}",
            pil_image is not None,
            len(history or []),
            stream,
            native,
            should_rag and self.retriever is not None,
        )

        return self.model.generate(
            image=pil_image,
            prompt=prompt,
            max_new_tokens=tokens,
            stream=stream,
            use_native=native and pil_image is not None,
        )

    def _load_image(self, source: ImageInput) -> Image.Image:
        """Load a PIL RGB image from multiple input types.

        Args:
            source: PIL image, filesystem path, HTTP(S) URL, or base64 payload.

        Returns:
            RGB ``PIL.Image``.

        Raises:
            ImageLoadError: Unsupported or invalid input.
        """
        if source is None:
            raise ImageLoadError("image source is None")

        if isinstance(source, Image.Image):
            return source.convert("RGB")

        if isinstance(source, Path):
            source = str(source)

        if isinstance(source, bytes):
            return self._open_image_bytes(source)

        if not isinstance(source, str):
            raise ImageLoadError(f"Unsupported image type: {type(source).__name__}")

        text = source.strip()
        if not text:
            raise ImageLoadError("empty image string")

        if text.startswith(("http://", "https://")):
            return self._load_image_from_url(text)

        if self._looks_like_base64(text):
            return self._load_image_from_base64(text)

        path = Path(text)
        if path.is_file():
            return self._open_image_file(path)

        raise ImageLoadError(f"Not a valid file path, URL, or base64 image: {text[:80]!r}")

    @staticmethod
    def _looks_like_base64(text: str) -> bool:
        """Heuristic: data-URI prefix or long alphanumeric payload."""
        if text.startswith("data:image"):
            return True
        if len(text) < 64:
            return False
        sample = text.split(",", 1)[-1][:256]
        return bool(re.fullmatch(r"[A-Za-z0-9+/=\s]+", sample))

    def _load_image_from_url(self, url: str) -> Image.Image:
        """Fetch and decode an image from a URL."""
        headers = {
            "User-Agent": _DEFAULT_USER_AGENT,
            "Accept": "image/*,*/*;q=0.8",
        }
        if "wikimedia.org" in url:
            headers["Referer"] = "https://commons.wikimedia.org/"

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=self.request_timeout_sec,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ImageLoadError(f"HTTP request failed for {url!r}: {exc}") from exc

        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("text/"):
            raise ImageLoadError(
                f"URL returned text ({content_type}), not an image: {url!r}"
            )

        return self._open_image_bytes(response.content)

    def _load_image_from_base64(self, payload: str) -> Image.Image:
        """Decode a base64 or data-URI image string."""
        if "," in payload and payload.startswith("data:"):
            payload = payload.split(",", 1)[1]
        try:
            raw = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ImageLoadError(f"Invalid base64 image data: {exc}") from exc
        return self._open_image_bytes(raw)

    @staticmethod
    def _open_image_file(path: Path) -> Image.Image:
        """Open an image from disk."""
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ImageLoadError(f"Cannot open image file {path}: {exc}") from exc

    @staticmethod
    def _open_image_bytes(data: bytes) -> Image.Image:
        """Open an image from raw bytes."""
        if not data:
            raise ImageLoadError("image bytes are empty")
        try:
            with Image.open(BytesIO(data)) as img:
                return img.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ImageLoadError(f"Cannot decode image bytes: {exc}") from exc

    def _augment_with_rag(
        self,
        question: str,
        image: Optional[Image.Image],
        top_k: int,
    ) -> str:
        """Retrieve context and prepend it to the user question."""
        assert self.retriever is not None
        retrieved = self.retriever.retrieve(
            query_image=image,
            query_text=question,
            top_k=top_k,
            hybrid=True,
        )
        context = format_retrieval_context(retrieved)
        if not context:
            return question
        logger.info("RAG retrieved {} documents", len(retrieved))
        return f"{context}\n\nQuestion: {question}"

    @staticmethod
    def _build_prompt(question: str, history: Optional[list[HistoryTurn]]) -> str:
        """Merge conversation history with the current question."""
        if not history:
            return question

        turns: list[HistoryTurn] = list(history) + [
            {"role": "user", "content": question},
        ]
        return MultimodalLLM._format_history(turns)
