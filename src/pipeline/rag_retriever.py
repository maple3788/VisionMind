"""Multimodal retrieval with CLIP embeddings and FAISS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from src.encoders.clip_encoder import CLIPVisionEncoder

ImageInput = Union[Image.Image, str, Path]


class MultimodalRetriever:
    """CLIP + FAISS retriever for image-text documents.

    Each document is embedded in CLIP's shared space (image + text towers).
    Hybrid queries combine image and text similarity with configurable weights.

    Args:
        encoder: ``CLIPVisionEncoder`` instance for embeddings.
        index_path: Optional directory to load a saved index on init.
        image_weight: Weight for image similarity in hybrid retrieval.
        text_weight: Weight for text similarity in hybrid retrieval.
    """

    def __init__(
        self,
        encoder: CLIPVisionEncoder,
        index_path: Optional[Union[str, Path]] = None,
        image_weight: float = 0.5,
        text_weight: float = 0.5,
    ) -> None:
        self.encoder = encoder
        self.image_weight = image_weight
        self.text_weight = text_weight

        self._index: faiss.Index | None = None
        self._image_index: faiss.Index | None = None
        self._text_index: faiss.Index | None = None
        self._documents: list[dict[str, Any]] = []
        self._dim: int | None = None

        if index_path is not None:
            self.load_index(index_path)

    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)

    def index_documents(self, docs: list[dict[str, Any]]) -> None:
        """Embed documents and build FAISS indices.

        Each document should contain:
            - ``image``: PIL image or filesystem path
            - ``text``: Caption or description
            - ``metadata``: Optional dict (id, source, etc.)

        Args:
            docs: List of document dicts to index.
        """
        if not docs:
            raise ValueError("docs must be a non-empty list")

        image_embeds: list[np.ndarray] = []
        text_embeds: list[np.ndarray] = []
        combined_embeds: list[np.ndarray] = []
        stored_docs: list[dict[str, Any]] = []

        for i, doc in enumerate(docs):
            if "text" not in doc:
                raise ValueError(f"Document {i} missing 'text' field")
            image = self._load_doc_image(doc.get("image"))
            text = str(doc["text"])
            metadata = dict(doc.get("metadata", {}))

            img_emb = self._encode_image(image)
            txt_emb = self._encode_text(text)
            combined = F.normalize((img_emb + txt_emb) / 2.0, dim=-1)

            image_embeds.append(self._to_vector(img_emb))
            text_embeds.append(self._to_vector(txt_emb))
            combined_embeds.append(self._to_vector(combined))

            stored_docs.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "image_path": doc.get("image_path"),
                }
            )
            if image is not None:
                stored_docs[-1]["_image"] = image

        self._dim = int(combined_embeds[0].shape[0])
        matrix = np.vstack(combined_embeds).astype(np.float32)
        img_matrix = np.vstack(image_embeds).astype(np.float32)
        txt_matrix = np.vstack(text_embeds).astype(np.float32)

        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(matrix)

        self._image_index = faiss.IndexFlatIP(self._dim)
        self._image_index.add(img_matrix)

        self._text_index = faiss.IndexFlatIP(self._dim)
        self._text_index.add(txt_matrix)

        self._documents = stored_docs
        logger.info("Indexed {} documents (dim={})", len(stored_docs), self._dim)

    def retrieve(
        self,
        query_image: ImageInput | None = None,
        query_text: Optional[str] = None,
        top_k: int = 3,
        hybrid: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k documents by CLIP similarity.

        Args:
            query_image: Optional query image (PIL or path).
            query_text: Optional query text.
            top_k: Number of results to return.
            hybrid: If True and both modalities present, fuse image + text scores.

        Returns:
            List of dicts with ``text``, ``metadata``, ``score``, and optional ``image``.
        """
        if self._index is None or not self._documents:
            raise RuntimeError("Index is empty. Call index_documents() or load_index() first.")
        if query_image is None and not query_text:
            raise ValueError("Provide query_image and/or query_text")

        k = min(top_k, len(self._documents))
        scores: np.ndarray | None = None
        indices: np.ndarray | None = None

        if hybrid and query_image is not None and query_text:
            img_q = self._encode_image(self._load_doc_image(query_image))
            txt_q = self._encode_text(query_text)
            query = F.normalize(
                self.image_weight * img_q + self.text_weight * txt_q,
                dim=-1,
            )
            scores, indices = self._index.search(
                query.cpu().numpy().astype(np.float32).reshape(1, -1),
                k,
            )
        elif query_image is not None:
            img_q = self._encode_image(self._load_doc_image(query_image))
            scores, indices = self._image_index.search(  # type: ignore[union-attr]
                img_q.cpu().numpy().astype(np.float32).reshape(1, -1),
                k,
            )
        else:
            txt_q = self._encode_text(str(query_text))
            scores, indices = self._text_index.search(  # type: ignore[union-attr]
                txt_q.cpu().numpy().astype(np.float32).reshape(1, -1),
                k,
            )

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            doc = dict(self._documents[int(idx)])
            doc["score"] = float(score)
            doc["rank"] = rank
            results.append(doc)
        return results

    def save_index(self, path: Union[str, Path]) -> None:
        """Persist FAISS indices and document metadata to disk."""
        if self._index is None:
            raise RuntimeError("No index to save")

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(out / "combined.index"))
        faiss.write_index(self._image_index, str(out / "image.index"))  # type: ignore[arg-type]
        faiss.write_index(self._text_index, str(out / "text.index"))  # type: ignore[arg-type]

        meta = {
            "dim": self._dim,
            "image_weight": self.image_weight,
            "text_weight": self.text_weight,
            "documents": [],
        }
        for doc in self._documents:
            entry = {
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "image_path": doc.get("image_path"),
            }
            meta["documents"].append(entry)

        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        images_dir = out / "images"
        images_dir.mkdir(exist_ok=True)
        for i, doc in enumerate(self._documents):
            if "_image" in doc:
                doc["_image"].save(images_dir / f"{i:05d}.jpg")

        logger.info("Saved index with {} docs to {}", len(self._documents), out)

    def load_index(self, path: Union[str, Path]) -> None:
        """Load a previously saved index from disk."""
        root = Path(path)
        self._index = faiss.read_index(str(root / "combined.index"))
        self._image_index = faiss.read_index(str(root / "image.index"))
        self._text_index = faiss.read_index(str(root / "text.index"))

        meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        self._dim = int(meta["dim"])
        self.image_weight = float(meta.get("image_weight", self.image_weight))
        self.text_weight = float(meta.get("text_weight", self.text_weight))

        self._documents = []
        images_dir = root / "images"
        for i, entry in enumerate(meta["documents"]):
            doc: dict[str, Any] = {
                "text": entry["text"],
                "metadata": entry.get("metadata", {}),
                "image_path": entry.get("image_path"),
            }
            img_file = images_dir / f"{i:05d}.jpg"
            if img_file.is_file():
                with Image.open(img_file) as img:
                    doc["_image"] = img.convert("RGB")
            self._documents.append(doc)

        logger.info("Loaded {} documents from {}", len(self._documents), root)

    @staticmethod
    def _to_vector(tensor: torch.Tensor) -> np.ndarray:
        """Convert ``[1, dim]`` or ``[dim]`` tensor to a 1-D float32 numpy vector."""
        return tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Return L2-normalized CLIP image embedding ``[1, dim]``."""
        emb = self.encoder._projected_image_features([image])
        return emb

    def _encode_text(self, text: str) -> torch.Tensor:
        """Return L2-normalized CLIP text embedding ``[1, dim]``."""
        emb = self.encoder._projected_text_features([text])
        return emb

    @staticmethod
    def _load_doc_image(image: ImageInput | None) -> Image.Image:
        """Load a document image from PIL or path."""
        if image is None:
            raise ValueError("Document must include an 'image' field")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as img:
            return img.convert("RGB")


def format_retrieval_context(
    docs: list[dict[str, Any]],
    max_chars: int = 2000,
) -> str:
    """Format retrieved documents as context for the LLM prompt.

    Args:
        docs: Output from ``MultimodalRetriever.retrieve``.
        max_chars: Truncate context to this many characters.

    Returns:
        Context block to prepend to the user question.
    """
    if not docs:
        return ""

    lines = [
        "Use the following reference documents to answer the question.",
        "If the documents are not relevant, say you are unsure.",
        "",
    ]
    for doc in docs:
        rank = doc.get("rank", "?")
        score = doc.get("score", 0.0)
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else ""
        header = f"[Doc {rank} | score={score:.3f}]"
        if meta_str:
            header += f" ({meta_str})"
        lines.append(header)
        lines.append(text)
        lines.append("")

    context = "\n".join(lines).strip()
    if len(context) > max_chars:
        context = context[: max_chars - 3] + "..."
    return context
