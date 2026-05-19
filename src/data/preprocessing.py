"""Dataset preprocessing utilities for multimodal QA fine-tuning."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageEnhance, ImageOps

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. Answer questions about images "
    "accurately and concisely."
)


def build_instruction_messages(
    question: str,
    answer: str,
    image: Optional[Image.Image] = None,
    system_prompt: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Build Qwen2-VL chat messages for instruction tuning.

    Args:
        question: User question about the image.
        answer: Ground-truth assistant reply (training target).
        image: Optional PIL image for the user turn.
        system_prompt: Optional system instruction.

    Returns:
        Message list suitable for ``processor.apply_chat_template``.
    """
    system = system_prompt or DEFAULT_SYSTEM_PROMPT
    user_content: list[dict[str, Any]] | str
    if image is not None:
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]
    else:
        user_content = question

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]


def build_instruction_prompt(
    question: str,
    answer: str,
    system_prompt: Optional[str] = None,
    processor: Any | None = None,
    image: Optional[Image.Image] = None,
) -> str:
    """Render instruction-tuning text via the model chat template.

    Args:
        question: User question.
        answer: Target answer.
        system_prompt: Optional system message.
        processor: HuggingFace processor with ``apply_chat_template``.
        image: Optional image for multimodal template.

    Returns:
        Formatted prompt string.
    """
    messages = build_instruction_messages(
        question=question,
        answer=answer,
        image=image,
        system_prompt=system_prompt,
    )
    if processor is None:
        return (
            f"System: {system_prompt or DEFAULT_SYSTEM_PROMPT}\n"
            f"User: {question}\n"
            f"Assistant: {answer}"
        )
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def augment_image(image: Image.Image, augment_config: dict[str, Any]) -> Image.Image:
    """Apply random image augmentations for training.

    Supported keys in ``augment_config``:
        ``random_crop`` (bool), ``crop_scale`` (float),
        ``horizontal_flip`` (bool), ``color_jitter`` (bool),
        ``brightness`` (float), ``contrast`` (float), ``saturation`` (float).

    Args:
        image: Input RGB image.
        augment_config: Augmentation flags and strengths.

    Returns:
        Augmented PIL image (RGB).
    """
    img = image.convert("RGB")

    if augment_config.get("random_crop", False):
        scale = float(augment_config.get("crop_scale", 0.9))
        w, h = img.size
        crop_w = max(1, int(w * scale))
        crop_h = max(1, int(h * scale))
        left = random.randint(0, max(0, w - crop_w))
        top = random.randint(0, max(0, h - crop_h))
        img = img.crop((left, top, left + crop_w, top + crop_h))
        img = img.resize((w, h), Image.Resampling.BILINEAR)

    if augment_config.get("horizontal_flip", False) and random.random() < 0.5:
        img = ImageOps.mirror(img)

    if augment_config.get("color_jitter", False):
        brightness = float(augment_config.get("brightness", 0.2))
        contrast = float(augment_config.get("contrast", 0.2))
        saturation = float(augment_config.get("saturation", 0.2))
        img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-brightness, brightness))
        img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-contrast, contrast))
        img = ImageEnhance.Color(img).enhance(1.0 + random.uniform(-saturation, saturation))

    return img


def normalize_vqa_answer(text: str) -> str:
    """Normalize VQA answers for exact-match evaluation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_dataset(data_dir: str | Path) -> dict[str, Any]:
    """Validate JSONL multimodal QA files and image references.

    Args:
        data_dir: Directory containing ``*.jsonl`` split files and images.

    Returns:
        Report dict with counts and error lists.
    """
    root = Path(data_dir)
    report: dict[str, Any] = {
        "data_dir": str(root),
        "files": [],
        "total_examples": 0,
        "missing_images": [],
        "invalid_lines": [],
        "missing_fields": [],
    }

    if not root.is_dir():
        report["error"] = f"Not a directory: {root}"
        return report

    jsonl_files = sorted(root.glob("*.jsonl"))
    if not jsonl_files:
        report["error"] = "No .jsonl files found"
        return report

    required = {"image", "question", "answer"}

    for path in jsonl_files:
        file_report: dict[str, Any] = {
            "file": path.name,
            "examples": 0,
            "errors": 0,
        }
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    report["invalid_lines"].append(
                        {"file": path.name, "line": line_no, "error": str(exc)}
                    )
                    file_report["errors"] += 1
                    continue

                if not required.issubset(row.keys()):
                    report["missing_fields"].append(
                        {"file": path.name, "line": line_no, "keys": list(row.keys())}
                    )
                    file_report["errors"] += 1
                    continue

                img_path = root / row["image"] if not Path(row["image"]).is_absolute() else Path(row["image"])
                if not img_path.is_file():
                    report["missing_images"].append(
                        {"file": path.name, "line": line_no, "image": str(img_path)}
                    )
                    file_report["errors"] += 1
                    continue

                file_report["examples"] += 1
                report["total_examples"] += 1

        report["files"].append(file_report)

    report["ok"] = (
        not report.get("error")
        and not report["missing_images"]
        and not report["invalid_lines"]
        and not report["missing_fields"]
    )
    return report
