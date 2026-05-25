---
name: thesis-operating-system
description: >
  Project-level operating rules for the Thai YouTube Virality Prediction thesis at KMITL.
  Use when the user plans, builds, or reviews: dataset collection/cleaning, label definition
  (ViralityIndex), train/val/test splitting (channel-grouped + time-aware), feature engineering
  (structural, TF-IDF, sentiment, embeddings), encoder fine-tuning (WangchanBERTa, PhayaThaiBERT,
  XLM-RoBERTa-large), hybrid MLP+GBM heads, evaluation (ROC-AUC, F1, McNemar, Cochran's Q,
  bootstrap CIs, calibration), explainability (SHAP, LIME, attention), Kaggle/Colab cloud GPU
  orchestration, MLflow tracking, or LaTeX thesis writing. Also use when the user asks scope
  questions like "should we add a decoder LLM?" or "should we change the label?".
  Do NOT use for: ERP/web/landing-page work, content writing unrelated to the thesis, or
  generic ML questions not grounded in this dataset.
user-invocable: true
---

# Thesis Operating System

This skill is the working agreement for the Thai YouTube Virality Prediction thesis. It is loaded into every session that touches model architecture, dataset semantics, evaluation methodology, or thesis content.

Always read `CLAUDE.md` first. This skill extends those rules with operating discipline.

---

## 1. Identity

**Title:** *A Comparative Study of Thai Transformer Encoder Models for YouTube Video Virality Prediction.*
**Department:** Applied Statistics & Data Analytics, KMITL, AY 2025.

This is a **comparative classification study**, not a generation task. The output is a probability that a video will be viral. There is no chatbot, no summarizer, no decoder LLM in the comparison.

**Models under comparison (fixed):**

| Alias | HF Name | Params |
|---|---|---|
| `wangchanberta` | `airesearch/wangchanberta-base-att-spm-uncased` | 105 M |
| `phayathaibert` | `clicknext/phayathaibert` | 278 M |
| `xlm-roberta-large` | `FacebookAI/xlm-roberta-large` | 560 M |

A QLoRA path for `typhoon-2.5` / `openthaigpt` is kept in the code for reproducibility of the original proposal but is **not part of the comparison**.

---

## 2. Operating Workflow

Every task follows this sequence. Do not skip.

1. **Inspect.** Read affected files. Understand current state before proposing change.
2. **Confirm scope.** If the task introduces a new model, a new label definition, a new split scheme, or removes a model from comparison — stop and write a decision entry first.
3. **Plan in thin slices.** Data → feature → model → eval → explain. Not all at once.
4. **Implement.** Match existing patterns in `src/`. Use `configs/*.yaml` for hyperparameters.
5. **Run with MLflow.** Every training/eval run logs config, git SHA, dataset hash, metrics, artifacts.
6. **Save predictions, not just metrics.** `reports/predictions/{model}_{split}.parquet` is the contract with the evaluator.
7. **Update memory.** `.claude/memory/experiments.md` + `.claude/memory/changelog.md` after every concrete result.
8. **Quality gate.** `pytest`, `ruff`, `black --check`, MLflow run logged → only then declare done.

---

## 3. Frozen Decisions (do not silently revisit)

These are written in `.claude/memory/decisions.md`. If they must change, write a new dated entry and announce.

- **Task type:** binary classification, top-decile-per-channel virality.
- **Label:** `ViralityIndex = z(log_views_per_channel) + z(engagement_rate_per_channel)`. Top decile per channel = positive.
- **Splits:** channel-grouped train/val (no channel in both) + time-aware test (latest 15% by `published_at`).
- **Imbalance handling:** focal loss + undersample on train only. Val/test untouched.
- **Comparison models:** 3 encoders listed above. Decoders out of scope.
- **Sentiment model:** `poom-sci/WangchanBERTa-finetuned-sentiment` (Wisesight-3K), frozen feature extractor, 4-dim distribution + arousal/valence/polar.
- **Hybrid backbone:** frozen WangchanBERTa CLS (768-dim) + sentiment (7-dim) + structural (16-dim) + numerical metadata.
- **Stats tests:** pairwise McNemar + Cochran's Q + bootstrap 95% CI (n=1000).

---

## 4. The Three Cardinal Risks

These are the only ways this thesis can produce a wrong conclusion. Guard against each on every change.

### 4.1 Data leakage
Same channel in train and val. Future titles influencing past splits. Sentiment or embeddings computed on the union and then split. **Mitigation:** all feature caching uses `video_id` as key; splits are done once in `splits.py` and never recomputed downstream.

### 4.2 Test-set contamination
HPO touching test. Calibration fitted on test. Stats tests run on val by mistake. **Mitigation:** `eval.yaml` is the only file that reads from `reports/predictions/*_test.parquet`. HPO and calibration consume `*_val.parquet`.

### 4.3 Cherry-picking
Reporting only the best seed. Hiding the val/test gap. Choosing a comparator that flatters the proposed model. **Mitigation:** every run logged to MLflow, predictions saved, experiments file is append-only.

---

## 5. Communication

- When proposing an experiment, state: hypothesis, what changes from the previous run, what's held constant, expected effect, how it will be measured.
- When reporting results, lead with the headline metric + 95% CI. Then context (what changed). Then artifacts (run id, predictions path).
- When a result surprises you, write it down in `.claude/memory/experiments.md` even if you don't understand it yet.

---

## 6. What This Skill Will Not Do

- Will not introduce decoder LLMs into the comparison.
- Will not change the label or the split scheme without a decision entry.
- Will not run HPO on the test set.
- Will not skip MLflow logging "just for this one quick run".
- Will not delete or overwrite a previous run's predictions.
