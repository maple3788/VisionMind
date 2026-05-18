"""Vision Transformer (ViT) encoder wrapper."""

from __future__ import annotations

import logging
from typing import Union

import torch
from PIL import Image
from torch import Tensor
from transformers import ViTImageProcessor, ViTModel

logger = logging.getLogger(__name__)


def resolve_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Resolve ``auto`` to the best available compute device.

    Args:
        device: Device string or ``torch.device``. Use ``auto`` for CUDA/MPS/CPU.

    Returns:
        Resolved ``torch.device``.
    """
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ViTEncoder:
    """Wraps a HuggingFace ViT model for vision feature extraction.

    Extracts patch-level and CLS-level features from images. By default, weights
    are frozen for inference-only use.

    Args:
        model_id: HuggingFace model identifier.
        device: Target device (``auto``, ``cuda``, ``mps``, ``cpu``).
        freeze: If True, disable gradients on all parameters.

    Example:
        >>> encoder = ViTEncoder("google/vit-base-patch16-224", device="cpu")
        >>> features = encoder.encode([pil_image])  # [1, 197, 768]
    """

    def __init__(
        self,
        model_id: str = "google/vit-base-patch16-224",
        device: Union[str, torch.device] = "auto",
        freeze: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)

        self.processor = ViTImageProcessor.from_pretrained(model_id)
        self.model = ViTModel.from_pretrained(model_id).to(self.device)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            logger.info("ViT weights frozen for %s", model_id)

        self.hidden_size: int = self.model.config.hidden_size
        self.patch_size: int = self.model.config.patch_size
        num_side = self.model.config.image_size // self.patch_size
        self.num_patches: int = num_side**2

    def _preprocess(self, images: list[Image.Image]) -> Tensor:
        """Convert PIL images to pixel tensors on the target device."""
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def encode(self, images: list[Image.Image]) -> Tensor:
        """Encode images into token-level feature tensors.

        Args:
            images: List of PIL images.

        Returns:
            Tensor of shape ``[batch, num_patches + 1, hidden_size]``. Index 0 is
            the CLS token; indices ``1:`` are patch tokens.
        """
        pixel_values = self._preprocess(images)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        return outputs.last_hidden_state

    def get_cls_feature(self, images: list[Image.Image]) -> Tensor:
        """Return the CLS token embedding for each image.

        Args:
            images: List of PIL images.

        Returns:
            Tensor of shape ``[batch, hidden_size]``.
        """
        return self.encode(images)[:, 0, :]

    def get_patch_features(self, images: list[Image.Image]) -> Tensor:
        """Return patch token embeddings, excluding the CLS token.

        Args:
            images: List of PIL images.

        Returns:
            Tensor of shape ``[batch, num_patches, hidden_size]``.
        """
        return self.encode(images)[:, 1:, :]
