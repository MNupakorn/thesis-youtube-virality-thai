---
name: evaluation-rigor
description: >
  Use when computing or reporting model metrics, comparing models, building confidence
  intervals, running statistical tests (McNemar, Cochran's Q), calibration (Platt, isotonic,
  ECE, reliability diagrams), per-category breakdowns, or error analysis. Triggers:
  "compare models", "is the difference significant?", "calibrate", "ECE", "report results".
user-invocable: false
---

# Evaluation Rigor

Headline numbers without rigor get you a B. Rigor gets the A+.

## 1. The Headline Metric

ROC-AUC on the test set, with bootstrap 95% CI (n=1000). Always reported alongside F1-pos, recall-pos, precision-pos.

Format: `0.673 [0.643, 0.700]`.

PR-AUC is also reported because the dataset is imbalanced (~9% positive on test).

## 2. Stats Tests Across the 3 Encoders

All tests run on the **test** set, all on the **same examples**, using the predictions parquet files.

### 2.1 Pairwise McNemar (3 pairs: WCB×PTB, WCB×XLMR, PTB×XLMR)
- Threshold both probabilities at 0.5 → binary predictions.
- Build the 2×2 contingency of agreements/disagreements.
- McNemar's exact test (use `statsmodels.stats.contingency_tables.mcnemar(..., exact=True)` for small disagreement counts; midp for larger).
- Report χ² statistic + p-value + Holm-Bonferroni adjusted p across the 3 pairs.

### 2.2 Cochran's Q (all three at once)
- Tests if the three classifiers have the same error rate on the same items.
- `statsmodels.stats.contingency_tables.cochrans_q`.
- Report Q statistic, df=2, p-value.

### 2.3 Output
Single CSV: `reports/tables/stats_tests.csv`:
```
test,model_a,model_b,statistic,p_raw,p_adj,verdict
mcnemar,wangchanberta,phayathaibert,12.3,0.001,0.003,significant
mcnemar,wangchanberta,xlm-roberta-large,4.5,0.034,0.068,not significant
mcnemar,phayathaibert,xlm-roberta-large,2.1,0.147,0.147,not significant
cochran_q,all,,15.4,0.0004,,significant
```

## 3. Bootstrap CIs

Use BCa (bias-corrected accelerated) bootstrap when feasible; percentile bootstrap as fallback.

```python
from src.evaluation.metrics import bootstrap_ci
auc, lo, hi = bootstrap_ci(y_test, prob_test, "roc_auc", n=1000, seed=42, method="percentile")
```

Same procedure for F1 / precision / recall — but bootstrap on the **same indices** for paired comparison.

## 4. Calibration

For each transformer (and each baseline reported in the thesis):

1. Fit Platt scaling on val: `LogisticRegression().fit(logit(prob_val).reshape(-1,1), y_val)`.
2. Fit isotonic on val: `IsotonicRegression(out_of_bounds="clip").fit(prob_val, y_val)`.
3. Apply each to test probabilities.
4. Compute ECE (10 equal-width bins) before / after Platt / after isotonic.
5. Reliability diagram → `reports/figures/calibration_{model}.svg`.

Report ECE in `reports/tables/calibration.csv`:
```
model,ece_raw,ece_platt,ece_isotonic
wangchanberta,0.064,0.041,0.038
...
```

## 5. Per-Category Breakdown

Categories: Gaming (15,852), Entertainment (5,703), Music (2,174). Smaller categories grouped as "Other".

Per category, report ROC-AUC + n + positive count. Output `reports/tables/per_category_metrics.csv`.

If any category drops the headline by more than 0.05 AUC, document it in the discussion.

## 6. Error Analysis

For each model, save:
- `reports/tables/{model}_top50_fp.csv` — false positives ranked by descending prob (model said viral, was not)
- `reports/tables/{model}_top50_fn.csv` — false negatives ranked by ascending prob (model said not, was viral)

Columns: `video_id, title, channel, published_at, views, label, prob`.

The thesis's discussion section needs this to write about *why* the models err.

## 7. Common Mistakes

- **Computing CIs from a single seed.** CIs are over the test set, not over seeds. Seed variance is a separate analysis (run 3 seeds, report mean ± std on top of the bootstrap CI).
- **Calibration on test.** Always val-fit, test-evaluate.
- **Comparing models trained on different splits.** All comparisons require identical splits (`data/processed/dataset_with_labels.parquet` sha256 must match).
- **Reporting accuracy.** With 9% positive rate, accuracy is misleading. Always lead with ROC-AUC and F1-pos.

## 8. The "Is the Difference Significant?" Question

When user asks: a 0.02 ROC-AUC gap between two models — is it real?

Answer in this order:
1. **Bootstrap on the test set:** if the 95% CIs overlap by more than 50% of either CI's width, treat as inconclusive.
2. **McNemar:** does the disagreement structure favor one model with p<0.05 (Holm-adjusted)?
3. **Cohen's effect size on errors:** is it large, medium, or small?

Only call a difference "significant" when McNemar p<0.05 (adjusted) AND non-overlapping bootstrap CIs.
