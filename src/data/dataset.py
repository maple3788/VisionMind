"""Multimodal QA dataset and splitting utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.preprocessing import (
    DEFAULT_SYSTEM_PROMPT,
    augment_image,
    build_instruction_messages,
)


class DatasetSplitter:
    """Split a single JSONL file into train/val/test (80/10/10)."""

    def __init__(
        self,
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> None:
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def split_file(
        self,
        source_jsonl: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write ``train.jsonl``, ``val.jsonl``, and ``test.jsonl``.

        Args:
            source_jsonl: Combined examples file.
            output_dir: Directory for split outputs.

        Returns:
            Dict mapping split name to output path.
        """
        source = Path(source_jsonl)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        lines = [ln.strip() for ln in source.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rng = random.Random(self.seed)
        rng.shuffle(lines)

        n = len(lines)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        splits = {
            "train": lines[:n_train],
            "val": lines[n_train : n_train + n_val],
            "test": lines[n_train + n_val :],
        }

        paths: dict[str, Path] = {}
        for name, rows in splits.items():
            path = out / f"{name}.jsonl"
            path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            paths[name] = path

        return paths


class MultimodalQADataset(Dataset):
    """JSONL multimodal QA dataset for instruction fine-tuning.

    Each line: ``{"image": "rel/or/abs path", "question": "...", "answer": "..."}``

    Args:
        data_dir: Root directory with ``{split}.jsonl`` and images.
        split: One of ``train``, ``val``, ``test``.
        processor: Qwen2-VL processor (tokenizer + vision preprocessing).
        max_length: Max sequence length for tokenization.
        augment_config: Augmentation dict (train split only).
        system_prompt: Optional system message override.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        processor: Any,
        max_length: int = 512,
        augment_config: Optional[dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.processor = processor
        self.max_length = max_length
        self.augment_config = augment_config or {}
        self.system_prompt = system_prompt
        self.tokenizer = processor.tokenizer

        jsonl_path = self.data_dir / f"{split}.jsonl"
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"Missing split file: {jsonl_path}")

        self.examples: list[dict[str, str]] = []
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        if not self.examples:
            raise ValueError(f"No examples in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def _resolve_image_path(self, image_ref: str) -> Path:
        path = Path(image_ref)
        if path.is_absolute():
            return path
        return self.data_dir / path

    def _load_image(self, image_ref: str) -> Image.Image:
        path = self._resolve_image_path(image_ref)
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.split == "train" and self.augment_config:
            image = augment_image(image, self.augment_config)
        return image

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.examples[index]
        image = self._load_image(row["image"])
        question = row["question"]
        answer = row["answer"]

        system = self.system_prompt or DEFAULT_SYSTEM_PROMPT
        user_content: list[dict[str, Any]] = [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]
        user_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        user_text = self.processor.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_messages = build_instruction_messages(
            question=question,
            answer=answer,
            image=image,
            system_prompt=self.system_prompt,
        )
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        user_inputs = self.processor(
            text=[user_text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )
        full_inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = int(user_inputs["input_ids"].shape[1])
        labels[:prompt_len] = -100

        sample: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question": question,
            "answer": answer,
            "image_path": str(row["image"]),
        }
        for key in ("pixel_values", "image_grid_thw", "video_grid_thw"):
            if key in full_inputs:
                sample[key] = full_inputs[key].squeeze(0) if full_inputs[key].dim() > 0 else full_inputs[key]

        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
        """Pad variable-length sequences for batch training.

        Args:
            batch: List of samples from ``__getitem__``.
            pad_token_id: Token id used for padding.

        Returns:
            Batched tensors including ``input_ids``, ``attention_mask``, ``labels``.
        """
        max_len = max(item["input_ids"].shape[0] for item in batch)

        def pad_1d(tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
            if tensor.shape[0] == max_len:
                return tensor
            pad_size = max_len - tensor.shape[0]
            return torch.cat(
                [tensor, torch.full((pad_size,), pad_value, dtype=tensor.dtype)]
            )

        input_ids = torch.stack(
            [pad_1d(item["input_ids"], pad_token_id) for item in batch]
        )
        attention_mask = torch.stack(
            [pad_1d(item["attention_mask"], 0) for item in batch]
        )
        labels = torch.stack([pad_1d(item["labels"], -100) for item in batch])

        collated: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "questions": [item["question"] for item in batch],
            "answers": [item["answer"] for item in batch],
            "image_paths": [item["image_path"] for item in batch],
        }

        if "pixel_values" in batch[0]:
            collated["pixel_values"] = torch.cat(
                [item["pixel_values"].unsqueeze(0) for item in batch], dim=0
            )
        for key in ("image_grid_thw", "video_grid_thw"):
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])

        return collated


def make_collate_fn(pad_token_id: int = 0) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Return a collate function bound to a pad token id."""
    return lambda batch: MultimodalQADataset.collate_fn(batch, pad_token_id=pad_token_id)
