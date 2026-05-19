"""LoRA fine-tuning for MultimodalLLM (Phase 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch
from accelerate import Accelerator
from loguru import logger
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.preprocessing import normalize_vqa_answer
from src.llm.backbone import MultimodalLLM


def apply_lora(
    model: MultimodalLLM,
    lora_config: dict[str, Any] | Any,
    train_projector: bool = True,
) -> MultimodalLLM:
    """Attach LoRA adapters to the language model; optionally train the projector.

    Args:
        model: Initialized ``MultimodalLLM``.
        lora_config: LoRA hyperparameters (``r``, ``alpha``, ``dropout``, ``target_modules``).
        train_projector: If True, projector weights remain trainable.

    Returns:
        The same ``MultimodalLLM`` with a PEFT-wrapped ``language_model``.
    """
    if not isinstance(lora_config, dict):
        lora_config = OmegaConf.to_container(lora_config, resolve=True)

    target_modules = list(lora_config.get("target_modules", ["q_proj", "v_proj"]))
    peft_config = LoraConfig(
        r=int(lora_config.get("r", 8)),
        lora_alpha=int(lora_config.get("alpha", 16)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.language_model = get_peft_model(model.language_model, peft_config)

    for param in model.encoder.parameters():
        param.requires_grad = False

    visual = getattr(model.llm, "visual", None)
    if visual is not None:
        for param in visual.parameters():
            param.requires_grad = False

    if train_projector:
        for param in model.projector.parameters():
            param.requires_grad = True
    else:
        for param in model.projector.parameters():
            param.requires_grad = False

    trainable_lm, total_lm = count_trainable_params(model.language_model)
    trainable_proj = sum(
        p.numel() for p in model.projector.parameters() if p.requires_grad
    )
    total_proj = sum(p.numel() for p in model.projector.parameters())
    trainable = trainable_lm + trainable_proj
    total = total_lm + total_proj
    logger.info(
        "LoRA applied | trainable={:,} / {:,} ({:.2f}%)",
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )
    return model


def count_trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def merge_and_save(model: MultimodalLLM, output_dir: Union[str, Path]) -> Path:
    """Merge LoRA weights into the base LM and save adapter + projector.

    Args:
        model: ``MultimodalLLM`` with PEFT-wrapped ``language_model``.
        output_dir: Directory for merged weights and projector checkpoint.

    Returns:
        Output directory path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lm = model.language_model
    if isinstance(lm, PeftModel):
        lm.save_pretrained(out / "lora_adapter")
        model.language_model = lm.merge_and_unload()
        logger.info("Saved and merged LoRA adapter at {}", out / "lora_adapter")
    else:
        logger.warning("language_model is not a PeftModel; skipping adapter merge")

    torch.save(model.projector.state_dict(), out / "projector.pt")
    logger.info("Saved projector weights to {}", out / "projector.pt")
    return out


class LoRATrainer:
    """Fine-tune LoRA adapters (+ projector) on a multimodal QA dataset.

    Uses HuggingFace ``accelerate`` for mixed-device training and optional MLflow
    metric logging.

    Args:
        model: ``MultimodalLLM`` after ``apply_lora``.
        train_dataset: Training ``MultimodalQADataset``.
        val_dataset: Validation dataset.
        train_config: Training hyperparameters (YAML-loaded dict or OmegaConf).
    """

    def __init__(
        self,
        model: MultimodalLLM,
        train_dataset: Any,
        val_dataset: Any,
        train_config: dict[str, Any] | Any,
        collate_fn: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if not isinstance(train_config, dict):
            train_config = OmegaConf.to_container(train_config, resolve=True)
        self.cfg = train_config

        self.accelerator = Accelerator(
            gradient_accumulation_steps=int(
                self.cfg.get("gradient_accumulation_steps", 1)
            )
        )
        pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id or 0
        self.collate_fn = collate_fn or (
            lambda batch: __import__(
                "src.data.dataset", fromlist=["MultimodalQADataset"]
            ).MultimodalQADataset.collate_fn(batch, pad_token_id=pad_id)
        )

        batch_size = int(self.cfg.get("batch_size", 4))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        trainable_params = [
            p for p in model.language_model.parameters() if p.requires_grad
        ] + [p for p in model.projector.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(self.cfg.get("learning_rate", 2e-4)),
        )

        if hasattr(self.model.llm, "gradient_checkpointing_enable"):
            self.model.llm.gradient_checkpointing_enable()
        self.train_loader, self.val_loader, self.optimizer = self.accelerator.prepare(
            self.train_loader, self.val_loader, self.optimizer
        )

        self._mlflow = None
        if self.cfg.get("use_mlflow", False):
            try:
                import mlflow

                self._mlflow = mlflow
            except ImportError:
                logger.warning("MLflow not installed; disabling experiment tracking")

    def train(self) -> list[dict[str, float]]:
        """Run the training loop.

        Returns:
            List of per-step metric dicts (loss).
        """
        epochs = int(self.cfg.get("num_epochs", 1))
        save_steps = int(self.cfg.get("save_steps", 100))
        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints/lora"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history: list[dict[str, float]] = []
        global_step = 0
        self.model.llm.train()
        self.model.projector.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                disable=not self.accelerator.is_local_main_process,
            )
            for batch in pbar:
                with self.accelerator.accumulate(self.model.llm):
                    loss = self._training_step(batch)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                global_step += 1
                num_batches += 1
                epoch_loss += float(loss.detach())
                metrics = {"loss": float(loss.detach()), "epoch": epoch, "step": global_step}
                history.append(metrics)
                self.log_metrics(global_step, metrics)
                pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

                if global_step % save_steps == 0:
                    self.save_checkpoint(checkpoint_dir / f"step_{global_step}")

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info("Epoch {} finished | avg_loss={:.4f}", epoch + 1, avg_loss)

        self.save_checkpoint(checkpoint_dir / "final")
        return history

    def _training_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """Forward pass on the full Qwen2-VL model (native multimodal path)."""
        device = self.model.device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k not in ("questions", "answers", "image_path")
        }
        outputs = self.model.llm(**batch)
        if outputs.loss is None and hasattr(outputs, "logits"):
            raise RuntimeError("Model forward did not return loss; pass labels in the batch.")
        return outputs.loss

    @torch.no_grad()
    def evaluate(
        self,
        max_samples: Optional[int] = None,
        max_new_tokens: int = 64,
        use_generation: bool = True,
    ) -> dict[str, float]:
        """Compute validation loss and VQA exact-match accuracy.

        Args:
            max_samples: Optional cap on evaluation examples.
            max_new_tokens: Generation length for accuracy metric.
            use_generation: If True, score with ``generate_native``; else token match on labels.

        Returns:
            Dict with ``loss`` and ``vqa_accuracy``.
        """
        self.model.llm.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for batch in self.val_loader:
            n_batches += 1
            device = self.model.device
            skip_keys = ("questions", "answers", "image_paths")
            model_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k not in skip_keys
            }
            lm = self.model.language_model
            outputs = lm(**model_batch)
            total_loss += float(outputs.loss.detach())

            if use_generation:
                from PIL import Image

                for q, gold, img_path in zip(
                    batch["questions"],
                    batch["answers"],
                    batch.get("image_paths", []),
                ):
                    if max_samples is not None and total >= max_samples:
                        break
                    full_path = self.val_dataset.data_dir / img_path
                    with Image.open(full_path) as img:
                        pred = self.model.generate_native(
                            img.convert("RGB"), q, max_new_tokens=max_new_tokens
                        )
                    if normalize_vqa_answer(pred) == normalize_vqa_answer(gold):
                        correct += 1
                    total += 1
            else:
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                labels = model_batch["labels"]
                for i in range(labels.shape[0]):
                    mask = labels[i] != -100
                    if mask.any() and torch.equal(preds[i][mask], labels[i][mask]):
                        correct += 1
                    total += 1

            if max_samples is not None and total >= max_samples:
                break

        self.model.llm.train()
        self.model.projector.train()
        accuracy = correct / max(total, 1)
        return {
            "loss": total_loss / max(n_batches, 1),
            "vqa_accuracy": accuracy,
            "num_evaluated": float(total),
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save LoRA adapter and projector weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model.language_model)
        if isinstance(unwrapped, PeftModel):
            unwrapped.save_pretrained(path / "lora")
        torch.save(self.model.projector.state_dict(), path / "projector.pt")
        logger.info("Checkpoint saved to {}", path)

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log metrics to MLflow when enabled."""
        if self._mlflow is None:
            return
        if self._mlflow.active_run() is None:
            self._mlflow.start_run(run_name=self.cfg.get("run_name", "lora-finetune"))
        for key, value in metrics.items():
            self._mlflow.log_metric(key, value, step=step)
