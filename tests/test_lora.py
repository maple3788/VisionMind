"""Unit tests for LoRA fine-tuning (mocked — no model downloads)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import PretrainedConfig

from src.llm.lora_finetune import LoRATrainer, apply_lora, count_trainable_params, merge_and_save


class TinyLM(nn.Module):
    """Minimal causal LM stub for PEFT tests."""

    def __init__(self, hidden: int = 32, vocab: int = 128) -> None:
        super().__init__()
        self.config = PretrainedConfig(model_type="llama", hidden_size=hidden)
        self.embed = nn.Embedding(vocab, hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> MagicMock:
        x = self.embed(input_ids)
        x = self.q_proj(x) + self.v_proj(x)
        logits = self.out(x)
        labels = kwargs.get("labels")
        if isinstance(labels, torch.Tensor):
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss_val = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss_val = logits.mean()
        out = MagicMock()
        out.loss = loss_val
        out.logits = logits
        return out

    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        """PEFT expects this hook on causal LMs."""
        return {"input_ids": input_ids}


class FakeMultimodalLLM:
    """Minimal MultimodalLLM stand-in for LoRA tests."""

    def __init__(self) -> None:
        self.language_model = TinyLM()
        self.projector = nn.Linear(8, 32)
        self.encoder = nn.Linear(8, 8)
        self.llm = MagicMock(
            side_effect=lambda **batch: self.language_model(**batch)
        )
        self.llm.train = MagicMock()
        self.llm.eval = MagicMock()
        self.llm.gradient_checkpointing_enable = MagicMock()
        self.device = torch.device("cpu")
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1
        self.generate_native = MagicMock(return_value="dog")


def _mock_multimodal_llm() -> FakeMultimodalLLM:
    """Build a fake MultimodalLLM with a real tiny LM inside."""
    return FakeMultimodalLLM()


class TestApplyLora:
    """Tests for LoRA attachment."""

    def test_apply_lora_reduces_trainable_params(self) -> None:
        """LoRA adds trainable adapters while base stays mostly frozen."""
        model = _mock_multimodal_llm()
        lora_cfg = {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
        apply_lora(model, lora_cfg, train_projector=True)
        trainable, total = count_trainable_params(model.language_model)
        assert trainable > 0
        assert model.projector.weight.requires_grad

    def test_projector_trainable_when_enabled(self) -> None:
        """Projector weights are trainable when train_projector=True."""
        model = _mock_multimodal_llm()
        apply_lora(model, {"r": 4, "alpha": 8, "target_modules": ["q_proj"]}, train_projector=True)
        assert model.projector.weight.requires_grad


class TestMergeAndSave:
    """Tests for checkpoint merge/save."""

    def test_merge_and_save_writes_files(self, tmp_path: Path) -> None:
        """merge_and_save writes projector and adapter paths."""
        model = _mock_multimodal_llm()
        lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"])
        model.language_model = get_peft_model(model.language_model, lora_cfg)

        out = merge_and_save(model, tmp_path)
        assert (out / "projector.pt").is_file()
        assert (out / "lora_adapter").is_dir()


class TestLoRATrainer:
    """Tests for training loop with mocked data."""

    def test_evaluate_token_accuracy(self) -> None:
        """evaluate(use_generation=False) returns vqa_accuracy key."""
        model = _mock_multimodal_llm()
        apply_lora(model, {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]})

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, 3, 4]]),
            "questions": ["q"],
            "answers": ["a"],
            "image_paths": ["img.png"],
        }
        val_loader = [batch]
        train_loader = [batch]

        train_ds = MagicMock()
        train_ds.data_dir = Path(".")
        val_ds = MagicMock()
        val_ds.data_dir = Path(".")

        cfg = {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "num_epochs": 1,
            "save_steps": 1000,
            "checkpoint_dir": "checkpoints/test",
        }

        train_ds.__len__ = MagicMock(return_value=1)
        val_ds.__len__ = MagicMock(return_value=1)

        with patch("src.llm.lora_finetune.Accelerator") as mock_acc_cls:
            acc = MagicMock()
            acc.is_local_main_process = True

            def _prepare(*args: object) -> tuple:
                return args

            acc.prepare.side_effect = _prepare
            acc.accumulate.return_value.__enter__ = MagicMock(return_value=None)
            acc.accumulate.return_value.__exit__ = MagicMock(return_value=None)
            acc.backward = MagicMock()
            acc.unwrap_model = MagicMock(side_effect=lambda m: m)
            mock_acc_cls.return_value = acc

            with patch("src.llm.lora_finetune.DataLoader") as mock_loader:
                mock_loader.return_value = train_loader
                trainer = LoRATrainer(model, train_ds, val_ds, cfg)
            trainer.train_loader = train_loader
            trainer.val_loader = val_loader
            metrics = trainer.evaluate(use_generation=False, max_samples=1)
        assert "vqa_accuracy" in metrics
        assert "loss" in metrics
