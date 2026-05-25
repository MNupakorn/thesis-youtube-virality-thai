# Experiments Log

Append-only. One entry per training or evaluation run. MLflow is the source of truth for numbers; this is the human-readable index.

---

## Format

```
### YYYY-MM-DD HH:MM · {model} · {feature_set} · {loss}

- **MLflow run:** <run_id>  (or "local file:./mlruns/<exp_id>/<run_id>")
- **Git SHA:** <short sha>
- **Dataset hash:** <12-char sha256>
- **Where:** local M1 / Kaggle T4 / Kaggle P100 / Colab
- **Config:** configs/train.yaml → <section>
- **Metrics (test):** ROC-AUC {x [lo, hi]} · F1-pos {x} · recall-pos {x} · precision-pos {x}
- **Metrics (val):** ROC-AUC {x}
- **Predictions:** reports/predictions/{model}_{train,val,test}.parquet
- **Artifacts:** reports/artifacts/{model}/
- **Notes:** one paragraph — anything surprising, anomalies, anything to investigate.
```

---

## Baselines (pre-architecture-setup)

### 2026-05-25 · lightgbm · structured+tfidf · class_weight=balanced

- **MLflow run:** (logged locally; check `mlruns/`)
- **Where:** local M1
- **Metrics (test):** ROC-AUC 0.673 [0.643, 0.700] · F1-pos 0.199 · recall-pos 0.240
- **Notes:** Best baseline. Sets the floor that the encoder FTs must beat.

### 2026-05-25 · xgboost · structured+tfidf

- **Metrics (test):** ROC-AUC 0.653 [0.621, 0.680] · F1-pos 0.191 · recall-pos 0.250

### 2026-05-25 · hybrid · cls+sent+struct · focal

- **Metrics (val):** ROC-AUC 0.70
- **Metrics (test):** ROC-AUC 0.56
- **Notes:** Large val/test gap. Distribution shift between channel-grouped val and time-aware test is the likely cause. Document in the thesis's Discussion section — do not hide.

---

## Encoder Fine-Tunes (pending)

### Pending: wangchanberta · title · focal — Kaggle T4
### Pending: phayathaibert · title · focal — Kaggle T4
### Pending: xlm-roberta-large · title · focal — Kaggle T4 (grad ckpt + batch 8 + accum 4)
