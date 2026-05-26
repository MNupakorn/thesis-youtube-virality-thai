"""Fine-tune Thai transformers (WangchanBERTa full FT, Typhoon / OpenThaiGPT via QLoRA)
   for binary virality classification on titles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import setup_logger


def _has_cuda() -> bool:
    """fp16/bf16 mixed precision in HuggingFace Trainer requires CUDA. MPS and CPU
    silently produce NaN logits for some architectures, so we gate the flags."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False

log = setup_logger("models.transformer_finetune")


def build_hf_dataset(df: pd.DataFrame, text_col: str = "title", label_col: str = "label_viral"):
    """Convert a pandas DataFrame to a 🤗 Datasets DatasetDict keyed by split."""
    from datasets import Dataset, DatasetDict

    splits = {}
    for sp in df["split"].unique():
        sub = df[df["split"] == sp]
        splits[sp] = Dataset.from_dict(
            {
                "text": sub[text_col].fillna("").astype(str).tolist(),
                "labels": sub[label_col].astype(int).tolist(),
                "video_id": sub["video_id"].astype(str).tolist(),
            }
        )
    return DatasetDict(splits)


class FocalLoss:
    """Binary focal loss for classification with class imbalance.

    L = -alpha_t * (1 - p_t) ** gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        self.gamma = gamma
        self.alpha = alpha  # weight of positive class; None = no rebalancing

    def __call__(self, logits, labels):
        import torch
        import torch.nn.functional as F

        ce = F.cross_entropy(logits, labels, reduction="none")
        p_t = torch.exp(-ce)
        focal = ((1 - p_t) ** self.gamma) * ce
        if self.alpha is not None:
            at = torch.where(labels == 1, self.alpha, 1.0 - self.alpha)
            focal = focal * at.to(focal.device)
        return focal.mean()


def _build_trainer_full_ft(
    model_name: str,
    train_ds,
    val_ds,
    output_dir: str | Path,
    num_labels: int,
    cfg: dict,
    class_weight_pos: float | None = None,
):
    """Standard full fine-tuning (used for WangchanBERTa)."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg["max_length"],
            padding=False,
        )

    train_ds = train_ds.map(_tok, batched=True, remove_columns=["text", "video_id"])
    val_ds = val_ds.map(_tok, batched=True, remove_columns=["text", "video_id"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg.get("grad_accumulation", 1),
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg.get("warmup_ratio", 0.0),
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("eval_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_roc_auc",
        greater_is_better=True,
        logging_steps=50,
        fp16=cfg.get("precision", "fp16") == "fp16" and _has_cuda(),
        bf16=cfg.get("precision") == "bf16" and _has_cuda(),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        seed=cfg.get("seed", 42),
        report_to=["none"],
    )

    loss_kind = cfg.get("loss", "cross_entropy")
    focal_gamma = cfg.get("focal_gamma", 2.0)
    pos_alpha = class_weight_pos

    class _Trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if loss_kind == "focal":
                loss = FocalLoss(gamma=focal_gamma, alpha=pos_alpha)(logits, labels)
            elif loss_kind == "weighted_ce" and pos_alpha is not None:
                w = torch.tensor([1.0 - pos_alpha, pos_alpha], device=logits.device)
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=w)
            else:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def _metrics(eval_pred):
        from sklearn.metrics import roc_auc_score

        logits, labels = eval_pred
        probs = (np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True))[:, 1]
        return {"roc_auc": float(roc_auc_score(labels, probs))}

    trainer = _Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get("early_stopping_patience", 2))],
    )
    return trainer, tokenizer


def _build_trainer_qlora(
    model_name: str,
    train_ds,
    val_ds,
    output_dir: str | Path,
    num_labels: int,
    cfg: dict,
    class_weight_pos: float | None = None,
):
    """QLoRA fine-tuning for Llama/Qwen-style decoder LLMs (Typhoon, OpenThaiGPT)."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    bnb_cfg = cfg.get("bnb", {})
    bnb = BitsAndBytesConfig(
        load_in_4bit=bnb_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, bnb_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=bnb_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bnb_cfg.get("bnb_4bit_use_double_quant", True),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    peft_cfg = cfg.get("peft", {})
    lora = LoraConfig(
        r=peft_cfg.get("r", 16),
        lora_alpha=peft_cfg.get("lora_alpha", 32),
        lora_dropout=peft_cfg.get("lora_dropout", 0.05),
        bias=peft_cfg.get("bias", "none"),
        target_modules=peft_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora)

    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg["max_length"],
            padding=False,
        )

    train_ds = train_ds.map(_tok, batched=True, remove_columns=["text", "video_id"])
    val_ds = val_ds.map(_tok, batched=True, remove_columns=["text", "video_id"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg.get("grad_accumulation", 1),
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg.get("warmup_ratio", 0.0),
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("eval_strategy", "epoch"),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_roc_auc",
        greater_is_better=True,
        logging_steps=20,
        bf16=cfg.get("precision", "bf16") == "bf16" and _has_cuda(),
        fp16=cfg.get("precision") == "fp16" and _has_cuda(),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        optim=cfg.get("optimizer", "paged_adamw_8bit"),
        seed=cfg.get("seed", 42),
        report_to=["none"],
    )

    loss_kind = cfg.get("loss", "focal")
    focal_gamma = cfg.get("focal_gamma", 2.0)
    pos_alpha = class_weight_pos

    class _Trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if loss_kind == "focal":
                loss = FocalLoss(gamma=focal_gamma, alpha=pos_alpha)(logits, labels)
            else:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def _metrics(eval_pred):
        from sklearn.metrics import roc_auc_score

        logits, labels = eval_pred
        probs = (np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True))[:, 1]
        return {"roc_auc": float(roc_auc_score(labels, probs))}

    trainer = _Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get("early_stopping_patience", 2))],
    )
    return trainer, tokenizer


def fine_tune(
    model_alias: str,
    df: pd.DataFrame,
    cfg: dict,
    output_dir: str | Path,
    class_weight_pos: float | None = None,
) -> dict[str, Any]:
    """High-level dispatcher.

    All three encoder models (WangchanBERTa, PhayaThaiBERT, XLM-RoBERTa-large)
    use full fine-tuning with the same Trainer wrapper — only batch size,
    gradient checkpointing, and learning rate change between them.
    The QLoRA path remains available for any future decoder LLM via the alias
    'typhoon-2.5' / 'openthaigpt' (kept for reproducibility of the original
    thesis design).
    """
    ds = build_hf_dataset(df)
    pretrained = cfg["pretrained_name"]
    num_labels = cfg.get("num_labels", 2)

    encoder_aliases = {"wangchanberta", "phayathaibert", "xlm-roberta-large"}
    if model_alias in encoder_aliases:
        trainer, tokenizer = _build_trainer_full_ft(
            pretrained, ds["train"], ds["val"], output_dir, num_labels, cfg, class_weight_pos
        )
    elif model_alias in ("typhoon-2.5", "openthaigpt"):
        trainer, tokenizer = _build_trainer_qlora(
            pretrained, ds["train"], ds["val"], output_dir, num_labels, cfg, class_weight_pos
        )
    else:
        raise ValueError(f"unknown model_alias: {model_alias}")

    trainer.train()

    # predict on each split for downstream evaluation
    preds: dict[str, np.ndarray] = {}
    for sp in ("train", "val", "test"):
        if sp not in ds:
            continue
        sub_ds = ds[sp].map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=cfg["max_length"], padding=False),
            batched=True,
            remove_columns=["text", "video_id"],
        )
        out = trainer.predict(sub_ds)
        logits = out.predictions
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        preds[sp] = probs[:, 1]

    return {"trainer": trainer, "tokenizer": tokenizer, "predictions": preds, "datasets": ds}
