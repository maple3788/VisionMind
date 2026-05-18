"""Multimodal LLM backbone: vision encoder + projector + language model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration

from src.encoders.clip_encoder import CLIPVisionEncoder
from src.encoders.vit_encoder import ViTEncoder, resolve_device
from src.projectors.linear_projector import LinearProjector
from src.projectors.mlp_projector import MLPProjector
from src.projectors.qformer import QFormer

HistoryTurn = dict[str, str]


def create_projector(
    projector_cfg: Any,
    in_dim: int,
    llm_dim: int,
    qformer_cfg: Any | None = None,
) -> nn.Module:
    """Build a projector module from config.

    Args:
        projector_cfg: Projector section of the model config.
        in_dim: Vision encoder hidden size.
        llm_dim: LLM hidden size (must match ``text_config.hidden_size``).
        qformer_cfg: Optional Q-Former section when ``type=qformer``.

    Returns:
        Initialized projector module.
    """
    ptype = str(projector_cfg.type).lower()
    if ptype == "linear":
        return LinearProjector(in_dim=in_dim, out_dim=llm_dim)
    if ptype == "mlp":
        return MLPProjector(
            in_dim=in_dim,
            hidden_dim=int(projector_cfg.hidden_dim),
            out_dim=llm_dim,
            num_layers=int(projector_cfg.num_layers),
            dropout=float(getattr(projector_cfg, "dropout", 0.0)),
        )
    if ptype == "qformer":
        if qformer_cfg is None:
            raise ValueError("qformer config section required when projector.type=qformer")
        return QFormer(
            num_queries=int(qformer_cfg.num_queries),
            encoder_dim=in_dim,
            llm_dim=llm_dim,
            num_heads=int(qformer_cfg.num_heads),
            num_layers=int(qformer_cfg.num_layers),
            dropout=float(qformer_cfg.dropout),
        )
    raise ValueError(f"Unknown projector type: {ptype}")


class MultimodalLLM:
    """Connects a vision encoder, projector, and Qwen2-VL language model.

    Supports two inference paths:

    * **Native** — Qwen2-VL's built-in vision tower and processor (baseline).
    * **Custom** — CLIP/ViT + projector; visual tokens prepended as
      ``inputs_embeds`` to the language model (learning pipeline).

    Args:
        encoder: Vision encoder instance (``CLIPVisionEncoder`` or ``ViTEncoder``).
        projector: Projector mapping encoder dim → LLM hidden dim.
        llm_model_id: HuggingFace model id for Qwen2-VL.
        device: Compute device (``auto``, ``cuda``, ``mps``, ``cpu``).
        load_in_4bit: Use 4-bit quantization when CUDA is available.
        encoder_on_cpu: Keep encoder/projector on CPU to save GPU memory.
        temperature: Sampling temperature for generation.
    """

    def __init__(
        self,
        encoder: Union[CLIPVisionEncoder, ViTEncoder],
        projector: nn.Module,
        llm_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Union[str, torch.device] = "auto",
        load_in_4bit: bool = True,
        encoder_on_cpu: bool = False,
        temperature: float = 0.7,
    ) -> None:
        self.encoder = encoder
        self.projector = projector
        self.llm_model_id = llm_model_id
        self.device = resolve_device(device)
        self.load_in_4bit = load_in_4bit
        self.encoder_on_cpu = encoder_on_cpu
        self.temperature = temperature

        self._encoder_device = torch.device("cpu") if encoder_on_cpu else self.device
        self._projector_device = self._encoder_device

        llm_config = AutoConfig.from_pretrained(llm_model_id)
        text_config = llm_config.get_text_config()
        self.llm_hidden_size: int = int(text_config.hidden_size)

        self._validate_projector_dim()

        logger.info("Loading LLM: {}", llm_model_id)
        self.llm, self.processor = self._load_llm_and_processor()
        self.tokenizer = self.processor.tokenizer
        self.language_model = self.llm.model.language_model

        self.projector = self.projector.to(self._projector_device)
        logger.info(
            "MultimodalLLM ready | device={} | llm_hidden={} | encoder_on_cpu={}",
            self.device,
            self.llm_hidden_size,
            encoder_on_cpu,
        )

    def _projector_out_dim(self) -> int:
        """Return the projector output dimension."""
        if isinstance(self.projector, QFormer):
            return int(self.projector.llm_dim)
        if isinstance(self.projector, LinearProjector):
            return int(self.projector.linear.out_features)
        if isinstance(self.projector, MLPProjector):
            return int(self.projector.out_norm.normalized_shape[0])
        raise TypeError(f"Unsupported projector type: {type(self.projector)}")

    def _validate_projector_dim(self) -> None:
        """Ensure projector output dim matches the LLM hidden size."""
        out_dim = self._projector_out_dim()
        if out_dim != self.llm_hidden_size:
            raise ValueError(
                f"Projector out dim ({out_dim}) must match LLM hidden size "
                f"({self.llm_hidden_size}). Update config/model_config.yaml."
            )

    def _load_llm_and_processor(
        self,
    ) -> tuple[Qwen2VLForConditionalGeneration, Any]:
        """Load Qwen2-VL and processor with device-appropriate precision."""
        use_4bit = self.load_in_4bit and self.device.type == "cuda"

        if use_4bit:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            llm = Qwen2VLForConditionalGeneration.from_pretrained(
                self.llm_model_id,
                quantization_config=quant_config,
                device_map="auto",
            )
        else:
            dtype = (
                torch.float16
                if self.device.type in ("cuda", "mps")
                else torch.float32
            )
            llm = Qwen2VLForConditionalGeneration.from_pretrained(
                self.llm_model_id,
                dtype=dtype,
            ).to(self.device)

        processor = AutoProcessor.from_pretrained(self.llm_model_id)
        return llm, processor

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        device: Union[str, torch.device] = "auto",
        encoder_on_cpu: bool = False,
    ) -> MultimodalLLM:
        """Build ``MultimodalLLM`` from ``config/model_config.yaml``.

        Args:
            config_path: Path to YAML config file.
            device: Target device for the LLM.
            encoder_on_cpu: Offload encoder/projector to CPU.

        Returns:
            Configured ``MultimodalLLM`` instance.
        """
        cfg = OmegaConf.load(config_path)
        enc_device = "cpu" if encoder_on_cpu else device

        encoder = CLIPVisionEncoder(
            model_id=cfg.vision_encoder.model_id,
            device=enc_device,
            freeze=bool(cfg.vision_encoder.freeze),
        )

        llm_config = AutoConfig.from_pretrained(cfg.llm.model_id)
        llm_dim = int(llm_config.get_text_config().hidden_size)
        in_dim = int(cfg.vision_encoder.output_dim)

        projector = create_projector(
            cfg.projector,
            in_dim=in_dim,
            llm_dim=llm_dim,
            qformer_cfg=getattr(cfg, "qformer", None),
        )

        return cls(
            encoder=encoder,
            projector=projector,
            llm_model_id=cfg.llm.model_id,
            device=device,
            load_in_4bit=bool(cfg.llm.load_in_4bit),
            encoder_on_cpu=encoder_on_cpu,
            temperature=float(cfg.llm.temperature),
        )

    def _encode_vision(self, image: Image.Image) -> Tensor:
        """Run encoder (+ Q-Former if applicable) to get LLM-space visual tokens."""
        if isinstance(self.projector, QFormer):
            if hasattr(self.encoder, "encode_image"):
                encoder_out = self.encoder.encode_image([image])
            else:
                encoder_out = self.encoder.encode([image])
            encoder_out = encoder_out.to(self._projector_device)
            with torch.no_grad():
                return self.projector(encoder_out)

        if hasattr(self.encoder, "get_patch_features"):
            vision_tokens = self.encoder.get_patch_features([image])
        elif hasattr(self.encoder, "encode_image"):
            vision_tokens = self.encoder.encode_image([image])[:, 1:, :]
        else:
            vision_tokens = self.encoder.encode([image])[:, 1:, :]

        vision_tokens = vision_tokens.to(self._projector_device)
        with torch.no_grad():
            return self.projector(vision_tokens)

    def prepare_inputs(
        self,
        image: Optional[Image.Image],
        prompt: str,
    ) -> dict[str, Tensor | int]:
        """Merge visual and text modalities into ``inputs_embeds``.

        Layout: ``[visual_tokens] + [text_tokens]``

        Args:
            image: Optional PIL image. If None, text-only.
            prompt: User question or instruction.

        Returns:
            Dict with ``inputs_embeds``, ``attention_mask``, token counts, and
            ``text_input_ids`` for debugging.
        """
        embed_layer = self.language_model.get_input_embeddings()
        text_encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        text_ids = text_encoding["input_ids"].to(self.device)
        text_mask = text_encoding["attention_mask"].to(self.device)
        text_embeds = embed_layer(text_ids)

        if image is not None:
            visual_embeds = self._encode_vision(image).to(self.device)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            visual_mask = torch.ones(
                visual_embeds.shape[:2], dtype=text_mask.dtype, device=self.device
            )
            attention_mask = torch.cat([visual_mask, text_mask], dim=1)
            num_visual = int(visual_embeds.shape[1])
        else:
            inputs_embeds = text_embeds
            attention_mask = text_mask
            num_visual = 0

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "text_input_ids": text_ids,
            "num_visual_tokens": num_visual,
            "num_text_tokens": int(text_ids.shape[1]),
            "total_tokens": int(inputs_embeds.shape[1]),
        }

    def generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: int = 512,
        stream: bool = False,
        use_native: bool = False,
    ) -> str | Iterator[str]:
        """Generate an answer for an image question (custom pipeline by default).

        Args:
            image: Optional input image.
            prompt: User question.
            max_new_tokens: Maximum tokens to generate.
            stream: If True, yield decoded text chunks (custom path only).
            use_native: If True, use Qwen2-VL native vision pipeline.

        Returns:
            Decoded answer string, or iterator when ``stream=True``.
        """
        if use_native:
            if image is None:
                raise ValueError("Native Qwen2-VL path requires an image.")
            return self.generate_native(image, prompt, max_new_tokens=max_new_tokens)

        if stream:
            return self._generate_custom_stream(image, prompt, max_new_tokens)

        start = time.perf_counter()
        inputs = self.prepare_inputs(image, prompt)
        prefix_len = int(inputs["total_tokens"])

        logger.info(
            "Custom generate | visual_tokens={} text_tokens={} total={}",
            inputs["num_visual_tokens"],
            inputs["num_text_tokens"],
            prefix_len,
        )

        with torch.no_grad():
            output_ids = self.language_model.generate(
                inputs_embeds=inputs["inputs_embeds"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        elapsed = time.perf_counter() - start
        if output_ids.shape[1] > prefix_len:
            answer_ids = output_ids[0, prefix_len:]
        else:
            answer_ids = output_ids[0]
        new_tokens = max(int(answer_ids.shape[0]), 1)
        logger.info(
            "Generation done | latency={:.2f}s | ~{:.1f} tok/s",
            elapsed,
            new_tokens / elapsed,
        )

        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    def _generate_custom_stream(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: int,
    ) -> Iterator[str]:
        """Streaming is not fully supported for inputs_embeds; yield full answer."""
        logger.warning("Streaming falls back to single-shot decode for custom path.")
        yield str(self.generate(image, prompt, max_new_tokens=max_new_tokens, stream=False))

    def generate_native(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Baseline inference with Qwen2-VL's built-in vision encoder.

        Args:
            image: Input PIL image.
            prompt: User question.
            max_new_tokens: Maximum new tokens to generate.

        Returns:
            Decoded model response.
        """
        start = time.perf_counter()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        elapsed = time.perf_counter() - start
        logger.info(
            "Native generate | latency={:.2f}s | new_tokens={}",
            elapsed,
            generated.shape[0],
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def chat(
        self,
        history: list[HistoryTurn],
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 512,
        use_native: bool = False,
    ) -> str:
        """Multi-turn chat using conversation history.

        Args:
            history: List of ``{"role": "user"|"assistant", "content": str}``.
            image: Optional image (used on first turn with vision).
            max_new_tokens: Max tokens to generate.
            use_native: Use native Qwen2-VL path.

        Returns:
            Assistant reply string.
        """
        prompt = self._format_history(history)
        return str(
            self.generate(
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                use_native=use_native,
            )
        )

    @staticmethod
    def _format_history(history: list[HistoryTurn]) -> str:
        """Format chat history into a single prompt string."""
        lines: list[str] = []
        for turn in history:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("Assistant:")
        return "\n".join(lines)
