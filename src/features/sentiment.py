"""Thai sentiment as a feature.

Uses a state-of-the-art Thai sentiment classifier as a frozen feature extractor.
Each title gets a probability vector over {pos, neu, neg, q} (Wisesight-3K labels).
Cached to parquet keyed by video_id so we don't re-infer on every run.

Primary model: ``poom-sci/WangchanBERTa-finetuned-sentiment``
              (WangchanBERTa fine-tuned on Wisesight-3K).
Fallback:     re-train a sentiment head from scratch if needed (out of scope).

We also expose:
- ``sent_arousal`` = 1 - P(neutral): roughly "how non-neutral" a title is, which
  captures Berger & Milkman (2012)'s "high arousal -> virality" hypothesis.
- ``sent_valence`` = P(pos) - P(neg): signed polarity in [-1, 1].
- ``sent_polar``   = P(pos) + P(neg): emotional charge regardless of sign.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import get_device, setup_logger

log = setup_logger("features.sentiment")

DEFAULT_MODEL = "poom-sci/WangchanBERTa-finetuned-sentiment"


class ThaiSentimentExtractor:
    """Thin wrapper around a HuggingFace classifier returning full prob distribution."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 128,
        batch_size: int = 64,
        device: str = "auto",
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device_str = get_device(device)

        log.info(f"loading sentiment model {model_name} on {self.device_str}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        import torch

        self.torch = torch
        self.model.to(self.device_str)

        cfg_id2label = getattr(self.model.config, "id2label", None)
        if isinstance(cfg_id2label, dict) and len(cfg_id2label) > 0:
            self.id2label = {int(k): str(v).lower() for k, v in cfg_id2label.items()}
        else:
            self.id2label = {0: "pos", 1: "neu", 2: "neg", 3: "q"}

    def _predict_batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device_str)
        with self.torch.no_grad():
            logits = self.model(**enc).logits
            probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict_proba(self, titles: list[str]) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(titles), self.batch_size), desc="sentiment"):
            batch = titles[i : i + self.batch_size]
            out.append(self._predict_batch(batch))
        return np.concatenate(out, axis=0)


def _to_sentiment_dataframe(probs: np.ndarray, id2label: dict[int, str]) -> pd.DataFrame:
    cols = {f"sent_p_{id2label[i]}": probs[:, i] for i in range(probs.shape[1])}
    out = pd.DataFrame(cols)
    p_pos = out.get("sent_p_pos", pd.Series(np.zeros(len(out))))
    p_neu = out.get("sent_p_neu", pd.Series(np.zeros(len(out))))
    p_neg = out.get("sent_p_neg", pd.Series(np.zeros(len(out))))
    out["sent_arousal"] = 1.0 - p_neu
    out["sent_valence"] = p_pos - p_neg
    out["sent_polar"] = p_pos + p_neg
    out["sent_top1"] = probs.argmax(axis=1)
    return out


def compute_or_load_sentiment(
    df: pd.DataFrame,
    cache_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    max_length: int = 128,
    batch_size: int = 64,
    device: str = "auto",
    title_col: str = "title",
    id_col: str = "video_id",
) -> pd.DataFrame:
    """Compute sentiment features for every row in df with on-disk parquet caching."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
        missing_ids = df.loc[~df[id_col].isin(cache[id_col]), id_col].unique().tolist()
        log.info(f"sentiment cache hit: {len(cache)} rows, {len(missing_ids)} new to compute")
    else:
        cache = pd.DataFrame()
        missing_ids = df[id_col].unique().tolist()

    if missing_ids:
        sub = df[df[id_col].isin(missing_ids)].drop_duplicates(id_col)
        extractor = ThaiSentimentExtractor(
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        titles = sub[title_col].fillna("").astype(str).tolist()
        probs = extractor.predict_proba(titles)
        new_feats = _to_sentiment_dataframe(probs, extractor.id2label)
        new_feats[id_col] = sub[id_col].values
        cache = pd.concat([cache, new_feats], ignore_index=True) if not cache.empty else new_feats
        cache.drop_duplicates(id_col, keep="last", inplace=True)
        cache.to_parquet(cache_path, index=False)
        log.info(f"sentiment cache saved -> {cache_path}  ({len(cache)} total rows)")

    return df.merge(cache, on=id_col, how="left")


SENTIMENT_FEATURE_COLUMNS = [
    "sent_p_pos",
    "sent_p_neu",
    "sent_p_neg",
    "sent_p_q",
    "sent_arousal",
    "sent_valence",
    "sent_polar",
]
