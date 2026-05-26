"""Attention rollout for transformer encoders.

Implements Abnar & Zuidema (2020): for each layer, A_l = 0.5 * (att_l + I) ; the
rolled attention is the cumulative product across layers. We extract the [CLS]
token's row to attribute importance to each input token.

Outputs:
- ``attention_rollout_per_example.parquet`` with one row per (example, token).
- ``attention_html/example_XXX.html`` highlighting tokens by rollout weight.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from src.utils import setup_logger

log = setup_logger("explain.attention")


def _rollout(attentions: list[np.ndarray]) -> np.ndarray:
    """attentions: list of [n_heads, T, T] per layer. Returns [T, T] rollout."""
    eye = np.eye(attentions[0].shape[-1])
    rolled = eye
    for att in attentions:
        att_mean = att.mean(axis=0)
        a = 0.5 * att_mean + 0.5 * eye
        a = a / a.sum(axis=-1, keepdims=True)
        rolled = a @ rolled
    return rolled


def _highlight_html(tokens: list[str], weights: np.ndarray, label: int, prob: float) -> str:
    if weights.max() > 0:
        norm = weights / weights.max()
    else:
        norm = weights
    spans = []
    for tok, w in zip(tokens, norm, strict=True):
        intensity = float(w)
        bg = f"rgba(220,38,38,{intensity:.2f})"
        spans.append(
            f'<span style="background:{bg};padding:2px 4px;margin:1px;'
            f'border-radius:3px" title="{intensity:.3f}">{html.escape(tok)}</span>'
        )
    body = " ".join(spans)
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>attn pred={label} p={prob:.3f}</title></head>"
        f'<body style="font-family:sans-serif;padding:16px">'
        f"<p>predicted_class={label} &nbsp; viral_prob={prob:.3f}</p>"
        f'<p style="font-size:18px;line-height:2">{body}</p></body></html>'
    )


def explain_with_attention(
    model_dir: str | Path,
    titles: list[str],
    labels: list[int] | None,
    out_dir: str | Path,
    n_samples: int = 20,
    max_length: int = 96,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_dir = out_dir / "attention_html"
    html_dir.mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir, output_attentions=True).to(
        device
    )
    mdl.train(False)

    rng = np.random.default_rng(seed)
    if len(titles) > n_samples:
        idx = rng.choice(len(titles), size=n_samples, replace=False).tolist()
    else:
        idx = list(range(len(titles)))

    rows = []
    for k, i in enumerate(idx):
        text = titles[i]
        with torch.no_grad():
            enc = tok([text], truncation=True, max_length=max_length, return_tensors="pt").to(
                device
            )
            out = mdl(**enc, output_attentions=True)
            probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
            attentions = [a[0].cpu().numpy() for a in out.attentions]

        rolled = _rollout(attentions)
        cls_attn = rolled[0]  # [CLS] row
        tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
        keep = min(len(tokens), len(cls_attn))
        weights = cls_attn[:keep]
        tokens = tokens[:keep]

        pred = int(probs.argmax())
        (html_dir / f"example_{k:03d}_idx{i}.html").write_text(
            _highlight_html(tokens, weights, pred, float(probs[1]))
        )
        for t, w in zip(tokens, weights, strict=True):
            rows.append(
                {
                    "example": k,
                    "title_idx": int(i),
                    "token": t,
                    "rollout_weight": float(w),
                    "predicted": pred,
                    "viral_prob": float(probs[1]),
                    "label": int(labels[i]) if labels is not None else -1,
                }
            )

    per_token = pd.DataFrame(rows)
    per_token.to_parquet(out_dir / "attention_rollout_per_example.parquet", index=False)
    log.info(f"attention rollout: {len(idx)} examples -> {out_dir}")
    return {"per_token": per_token}
