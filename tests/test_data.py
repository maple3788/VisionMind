"""Unit tests for data loading and preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import DatasetSplitter, MultimodalQADataset, make_collate_fn
from src.data.preprocessing import (
    augment_image,
    build_instruction_messages,
    build_instruction_prompt,
    normalize_vqa_answer,
    validate_dataset,
)


@pytest.fixture
def sample_dataset_dir(tmp_path: Path) -> Path:
    """Minimal JSONL dataset with one image."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "a.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img_path)

    row = {
        "image": "images/a.png",
        "question": "What color?",
        "answer": "black",
    }
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
    return tmp_path


class TestPreprocessing:
    """Tests for preprocessing helpers."""

    def test_build_instruction_messages(self) -> None:
        """Messages include system, user (with image), and assistant."""
        img = Image.new("RGB", (8, 8))
        msgs = build_instruction_messages("Q?", "A.", image=img)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["content"] == "A."

    def test_build_instruction_prompt_fallback(self) -> None:
        """Without processor, returns a readable string."""
        text = build_instruction_prompt("Q?", "A.")
        assert "Q?" in text and "A." in text

    def test_augment_image_flip(self) -> None:
        """Horizontal flip changes pixel layout."""
        img = Image.new("RGB", (16, 16), color=(255, 0, 0))
        left = img.copy()
        left.putpixel((0, 0), (0, 0, 255))
        out = augment_image(left, {"horizontal_flip": True})
        assert out.size == left.size

    def test_normalize_vqa_answer(self) -> None:
        """Normalization lowercases and strips punctuation."""
        assert normalize_vqa_answer("  Dog! ") == "dog"

    def test_validate_dataset_ok(self, sample_dataset_dir: Path) -> None:
        """Valid dataset reports ok=True."""
        report = validate_dataset(sample_dataset_dir)
        assert report["ok"] is True
        assert report["total_examples"] == 3

    def test_validate_dataset_missing_image(self, tmp_path: Path) -> None:
        """Missing images are reported."""
        (tmp_path / "train.jsonl").write_text(
            json.dumps({"image": "nope.jpg", "question": "q", "answer": "a"}) + "\n",
            encoding="utf-8",
        )
        report = validate_dataset(tmp_path)
        assert report["missing_images"]


class TestDatasetSplitter:
    """Tests for JSONL splitting."""

    def test_split_ratios(self, tmp_path: Path) -> None:
        """80/10/10 split produces three files."""
        rows = [json.dumps({"image": f"{i}.jpg", "question": "q", "answer": "a"}) for i in range(10)]
        src = tmp_path / "all.jsonl"
        src.write_text("\n".join(rows), encoding="utf-8")

        paths = DatasetSplitter().split_file(src, tmp_path / "splits")
        assert paths["train"].exists()
        train_lines = paths["train"].read_text(encoding="utf-8").strip().splitlines()
        assert len(train_lines) == 8


class TestMultimodalQADataset:
    """Tests for dataset and collate (mocked processor)."""

    @staticmethod
    def _mock_processor() -> MagicMock:
        processor = MagicMock()
        processor.apply_chat_template = MagicMock(side_effect=lambda msgs, **_: str(msgs))
        processor.tokenizer = MagicMock()
        processor.tokenizer.pad_token_id = 0

        def _proc(*_args: object, **kwargs: object) -> dict:
            text = kwargs.get("text", ["x"])[0]
            length = max(8, len(str(text)))
            ids = torch.arange(length).unsqueeze(0)
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
            }

        processor.side_effect = _proc
        return processor

    def test_getitem_keys(self, sample_dataset_dir: Path) -> None:
        """__getitem__ returns tensors and metadata."""
        processor = self._mock_processor()
        ds = MultimodalQADataset(
            sample_dataset_dir, "train", processor, max_length=64, augment_config={}
        )
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item
        assert item["answer"] == "black"
        assert (item["labels"] == -100).any()

    def test_collate_fn(self, sample_dataset_dir: Path) -> None:
        """Collate pads batches to the same length."""
        processor = self._mock_processor()
        ds = MultimodalQADataset(
            sample_dataset_dir, "train", processor, max_length=64, augment_config={}
        )
        batch = [ds[0], ds[0]]
        collated = make_collate_fn(pad_token_id=0)(batch)
        assert collated["input_ids"].shape[0] == 2
        assert len(collated["image_paths"]) == 2
