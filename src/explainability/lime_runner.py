"""LIME explanations for a fine-tuned transformer text classifier.

Given a HuggingFace classifier checkpoint and a list of titles, produce per-title
HTML explanations and an aggregate top-token CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils import setup_logger

log = setup_logger("explain.lime")


def _build_predictor(model_dir: str | Path, device: str = "cpu", max_length: int = 96) -> Any:
    """Return a function ``predict_proba(list[str]) -> np.ndarray[N, 2]`` for LIME."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    mdl.train(False)

    def predict_proba(texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            enc = tok(
                texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    return predict_proba


def explain_with_lime(
    model_dir: str | Path,
    titles: list[str],
    out_dir: str | Path,
    n_samples_to_explain: int = 30,
    num_features: int = 10,
    num_perturbations: int = 1000,
    max_length: int = 96,
    seed: int = 42,
) -> dict:
    """Run LIME on a sample of titles. Save per-example HTML + aggregate CSV."""
    import pandas as pd
    from lime.lime_text import LimeTextExplainer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predict_proba = _build_predictor(model_dir, max_length=max_length)
    explainer = LimeTextExplainer(class_names=["non_viral", "viral"], random_state=seed)

    rng = np.random.default_rng(seed)
    if len(titles) > n_samples_to_explain:
        idx = rng.choice(len(titles), size=n_samples_to_explain, replace=False).tolist()
    else:
        idx = list(range(len(titles)))

    aggregate: dict[str, float] = {}
    rows = []
    html_dir = out_dir / "lime_html"
    html_dir.mkdir(exist_ok=True)
    for k, i in enumerate(idx):
        text = titles[i]
        try:
            exp = explainer.explain_instance(
                text,
                predict_proba,
                num_features=num_features,
                num_samples=num_perturbations,
                labels=(1,),
            )
            (html_dir / f"example_{k:03d}_idx{i}.html").write_text(exp.as_html())
            for tok, weight in exp.as_list(label=1):
                rows.append(
                    {"example": k, "title_idx": int(i), "token": tok, "weight": float(weight)}
                )
                aggregate[tok] = aggregate.get(tok, 0.0) + abs(float(weight))
        except Exception as e:
            log.warning(f"lime failed on idx={i}: {e}")

    per_example = pd.DataFrame(rows)
    per_example.to_csv(out_dir / "lime_per_example.csv", index=False)
    agg = (
        pd.DataFrame({"token": list(aggregate), "sum_abs_weight": list(aggregate.values())})
        .sort_values("sum_abs_weight", ascending=False)
        .reset_index(drop=True)
    )
    agg.to_csv(out_dir / "lime_aggregate_tokens.csv", index=False)
    log.info(f"lime: explained {len(idx)} titles -> {out_dir}")
    return {"per_example": per_example, "aggregate": agg}
