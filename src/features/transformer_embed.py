"""Transformer-derived sentence embeddings (CLS or mean-pool) for hybrid model fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import get_device, setup_logger

log = setup_logger("features.transformer_embed")


def _set_eval_mode(model) -> None:
    """Put a HuggingFace model into inference mode (bypass dropout / etc.)."""
    getattr(model, "eval")()


class TitleEmbedder:
    """Compute fixed-size embeddings for a list of titles."""

    def __init__(
        self,
        model_name: str = "airesearch/wangchanberta-base-att-spm-uncased",
        pooling: str = "cls",
        max_length: int = 64,
        batch_size: int = 64,
        device: str = "auto",
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.device_str = get_device(device)

        log.info(f"loading embedder {model_name} on {self.device_str}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        _set_eval_mode(self.model)

        import torch

        self.torch = torch
        self.model.to(self.device_str)

    def _pool(self, last_hidden_state, attention_mask):
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def encode(self, texts: list[str]) -> np.ndarray:
        all_embs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="embed"):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device_str)
            with self.torch.no_grad():
                out = self.model(**enc)
            pooled = self._pool(out.last_hidden_state, enc["attention_mask"])
            all_embs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embs, axis=0)


def compute_or_load_embeddings(
    df: pd.DataFrame,
    cache_path: str | Path,
    model_name: str = "airesearch/wangchanberta-base-att-spm-uncased",
    pooling: str = "cls",
    max_length: int = 64,
    batch_size: int = 64,
    device: str = "auto",
    title_col: str = "title",
    id_col: str = "video_id",
) -> tuple[pd.Series, np.ndarray]:
    """Return (video_ids, embedding_matrix) aligned to df row order."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path = cache_path.with_suffix(".ids.parquet")

    if cache_path.exists() and ids_path.exists():
        log.info(f"loading cached embeddings from {cache_path}")
        embs = np.load(cache_path)
        ids = pd.read_parquet(ids_path)[id_col]
        cache_set = set(ids.tolist())
        missing = df[~df[id_col].isin(cache_set)]
    else:
        embs = np.empty((0, 0), dtype=np.float32)
        ids = pd.Series([], dtype=str, name=id_col)
        missing = df

    if len(missing) > 0:
        embedder = TitleEmbedder(
            model_name=model_name,
            pooling=pooling,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        sub = missing.drop_duplicates(id_col)
        new_embs = embedder.encode(sub[title_col].fillna("").astype(str).tolist())
        new_ids = sub[id_col]

        if embs.size == 0:
            embs = new_embs
            ids = new_ids
        else:
            embs = np.concatenate([embs, new_embs], axis=0)
            ids = pd.concat([ids, new_ids], ignore_index=True)

        np.save(cache_path, embs)
        pd.DataFrame({id_col: ids}).to_parquet(ids_path, index=False)
        log.info(f"saved embeddings ({embs.shape}) -> {cache_path}")

    id_to_row = {vid: i for i, vid in enumerate(ids.tolist())}
    sel = df[id_col].map(id_to_row).to_numpy()
    return df[id_col], embs[sel]
