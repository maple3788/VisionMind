"""CLIP vision-text encoder wrapper."""

from __future__ import annotations

import logging
from typing import Any, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor

from src.encoders.vit_encoder import resolve_device

logger = logging.getLogger(__name__)


def _as_clip_embedding_tensor(features: Tensor | tuple[Any, ...] | Any) -> Tensor:
    """Extract a 2D embedding tensor from CLIP ``get_*_features`` return values.

    Newer ``transformers`` versions return ``BaseModelOutputWithPooling`` (or a
    tuple wrapping it) instead of a raw ``Tensor``.

    Args:
        features: Return value from ``CLIPModel.get_image_features`` or
            ``get_text_features``.

    Returns:
        Float tensor of shape ``[batch, embed_dim]``.
    """
    if isinstance(features, torch.Tensor):
        return features

    if isinstance(features, tuple):
        if len(features) == 0:
            raise TypeError("CLIP features tuple is empty")
        features = features[0]

    if isinstance(features, torch.Tensor):
        return features

    pooler = getattr(features, "pooler_output", None)
    if pooler is not None:
        return pooler

    last = getattr(features, "last_hidden_state", None)
    if last is not None:
        return last[:, 0, :]

    raise TypeError(
        f"Cannot extract embedding tensor from CLIP output type {type(features)!r}"
    )


class CLIPVisionEncoder:
    """Wraps a HuggingFace CLIP model for image and text encoding.

    Both vision and text towers are frozen by default. Image encoding returns
    full sequence features for downstream projectors; similarity uses CLIP's
    projected embedding space.

    Args:
        model_id: HuggingFace CLIP model identifier.
        device: Target device (``auto``, ``cuda``, ``mps``, ``cpu``).
        freeze: If True, disable gradients on all parameters.

    Example:
        >>> encoder = CLIPVisionEncoder(device="cpu")
        >>> feats = encoder.encode_image([pil_image])  # [1, seq_len, hidden]
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        device: Union[str, torch.device] = "auto",
        freeze: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)

        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            logger.info("CLIP weights frozen for %s", model_id)

        vision_config = self.model.config.vision_config
        self.hidden_size: int = vision_config.hidden_size
        self.patch_size: int = vision_config.patch_size
        image_size = vision_config.image_size
        num_side = image_size // self.patch_size
        self.num_patches: int = num_side**2

    def encode_image(self, images: list[Image.Image]) -> Tensor:
        """Encode images into vision token features.

        Args:
            images: List of PIL images.

        Returns:
            Tensor of shape ``[batch, num_patches + 1, hidden_size]``.
        """
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values=pixel_values)

        return outputs.last_hidden_state

    def encode_text(self, texts: list[str]) -> Tensor:
        """Encode text strings into feature vectors.

        Uses the final token hidden state from the text transformer (EOS position).

        Args:
            texts: List of text strings.

        Returns:
            Tensor of shape ``[batch, hidden_size]``.
        """
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Pool at the last non-padding token for each sequence.
        last_token_indices = attention_mask.sum(dim=-1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=self.device)
        return outputs.last_hidden_state[batch_indices, last_token_indices, :]

    def _projected_image_features(self, images: list[Image.Image]) -> Tensor:
        """Return L2-normalized CLIP image embeddings for similarity."""
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            raw = self.model.get_image_features(pixel_values=pixel_values)
            features = _as_clip_embedding_tensor(raw)

        return F.normalize(features, dim=-1)

    def _projected_text_features(self, texts: list[str]) -> Tensor:
        """Return L2-normalized CLIP text embeddings for similarity."""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            raw = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            features = _as_clip_embedding_tensor(raw)

        return F.normalize(features, dim=-1)

    def compute_similarity(
        self,
        images: list[Image.Image],
        texts: list[str],
    ) -> Tensor:
        """Compute cosine similarity between images and texts.

        Args:
            images: List of PIL images.
            texts: List of text strings.

        Returns:
            Similarity matrix of shape ``[num_images, num_texts]`` with values
            in ``[-1, 1]``.
        """
        image_features = self._projected_image_features(images)
        text_features = self._projected_text_features(texts)
        return image_features @ text_features.T
