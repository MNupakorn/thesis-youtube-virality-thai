---
name: thesis-writing
description: >
  Use when drafting, structuring, or polishing the LaTeX thesis. Covers chapter outline,
  figure/table conventions, citation style, how to phrase the comparative claim, what to
  emphasize in results vs. discussion, and how to handle negative/honest findings (e.g.
  the val/test gap). Triggers: "write the methods chapter", "draft the abstract", "polish
  the discussion", "what should the conclusion say?".
user-invocable: false
---

# Thesis Writing

The code produces numbers. This skill turns those numbers into a defensible thesis chapter.

## 1. Chapter Outline (8 chapters)

1. **Introduction** — problem (Thai content creators, virality prediction value), research questions, contributions.
2. **Background & Related Work** — Thai NLP, transformer encoders for classification, social-media virality literature, sentiment-as-feature literature.
3. **Dataset** — collection (YouTube API v3), cleaning, label definition (ViralityIndex), splits (channel-grouped + time-aware), descriptive stats, ethics.
4. **Methods** — features (structural, TF-IDF, sentiment, embeddings), baselines, three encoders (full FT setup), hybrid (MLP + LightGBM heads), imbalance handling (focal + undersample), HPO (Optuna), calibration.
5. **Experimental Setup** — hardware (M1 local + Kaggle T4), seeds, hyperparameters, MLflow tracking, reproducibility statement.
6. **Results** — baselines table, three-encoder comparison table with bootstrap CIs, McNemar + Cochran's Q, per-category breakdown, calibration, hybrid vs. best encoder.
7. **Discussion** — what the results imply, the val/test gap (honest acknowledgement of distribution shift), error analysis themes, comparison to related work, limitations.
8. **Conclusion & Future Work** — main findings (1 paragraph), contributions (3 bullets), future work (decoder LLM comparison, multimodal with thumbnail, deployment study).

## 2. The Comparative Claim

Frame the research question as: **"Among modern Thai-capable transformer encoders, which best predicts viral YouTube videos from titles + metadata + sentiment?"**

Not "which is the SOTA Thai NLP model" — that's a different (and harder) question.

The honest answer the thesis can defend:
- All three encoders are reasonable choices.
- Statistical tests (McNemar + Cochran's Q) determine if observed differences are real.
- The thesis's contribution is the *first head-to-head* on this task with proper rigor (channel-grouped + time-aware + bootstrap + McNemar).

## 3. Tables (auto-generated, do not retype)

- `reports/tables/baselines_metrics.csv` → Table 1
- `reports/tables/encoder_metrics.csv` → Table 2 (THE headline table)
- `reports/tables/stats_tests.csv` → Table 3
- `reports/tables/calibration.csv` → Table 4
- `reports/tables/per_category_metrics.csv` → Table 5
- `reports/tables/{model}_top50_fp.csv` / `_fn.csv` → Appendix

LaTeX: use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`). Generate via `pandas.DataFrame.to_latex(..., float_format="%.3f")` so numbers come straight from the CSVs.

## 4. Figures

SVG → PDF (LaTeX-friendly). Always:
- ROC curves: all 3 encoders + best baseline on one plot, test set only.
- PR curves: same.
- Calibration / reliability diagrams: one per model.
- Confusion matrices: 4×4 grid (3 encoders + best baseline).
- Per-category bar: ROC-AUC per category per model.

## 5. How to Phrase the Val/Test Gap

This is a real finding — do not hide it. Suggested phrasing:

> "The hybrid model achieved validation ROC-AUC of 0.70 but test ROC-AUC of 0.56, indicating distribution shift between the channel-grouped validation set and the time-aware test set. We interpret this as evidence that virality patterns evolve over the 17-month window, and that channel-based generalization is easier than time-based generalization. This finding is consistent with [prior work] and motivates time-aware evaluation in social-media ML."

## 6. Negative or Surprising Findings to Report Honestly

- Sentiment features alone have ROC-AUC ≈ 0.50 (no signal solo, modest contribution in combination). Report it.
- The largest model (XLM-R-large, 560M) may not be the best. If so, say so — and discuss compute-efficiency tradeoffs.
- Bootstrap CIs may overlap across all three encoders, meaning no winner with statistical significance. That's a valid finding — it implies model choice doesn't dominate; data + features do.

## 7. Citation Style

- IEEE numeric or APA (consult advisor). Default IEEE for technical thesis.
- BibTeX: keep one `references.bib` per chapter or one global; do not duplicate entries.
- Cite the model papers (WangchanBERTa, PhayaThaiBERT, XLM-R), the sentiment dataset (Wisesight-3K), focal loss, McNemar's test, Cochran's Q, Optuna, MLflow.

## 8. What Goes in Appendix vs. Body

| Body | Appendix |
|---|---|
| Headline metrics table | All 3 seeds × all 3 models full grid |
| ROC + calibration figures | Reliability diagrams per category |
| 2–3 example FPs and FNs | Top-50 FP/FN tables |
| Architecture diagram | Full pyproject.toml + env freeze |
| Stats tests results | Bootstrap distribution histograms |

## 9. The Abstract

Must include, in order: task (1 sentence), dataset size (1), method (1–2), key result with CIs (1), comparative finding from stats tests (1), limitation/discussion (1). Aim for 200–250 words. Thai version + English version.

## 10. Drafting Order

1. Draft Methods + Experimental Setup first (they're closest to the code).
2. Draft Results from the CSVs.
3. Draft Discussion only after all results are final — never write conclusions before numbers exist.
4. Draft Introduction + Abstract last.
5. Polish chapter-by-chapter top to bottom in a separate pass.
