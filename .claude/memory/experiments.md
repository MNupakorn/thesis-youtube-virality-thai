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

### 2026-05-26 · wangchanberta · title · focal · M1 MPS local fallback (3 epoch)

- **Why local:** Kaggle free GPU pool was congested >6 h, kernel `mknpk01/thesis-wangchanberta-20260525` never left QUEUED state past the 6 h driver cap. Fell back to local M1 fine-tune.
- **Config:** `configs/train_m1.yaml` (epochs=3, batch_size=16, grad_accum=2, max_length=48, gradient_checkpointing=true, fp32 — fp16 produced NaN logits on MPS).
- **Best checkpoint:** `reports/artifacts/models/wangchanberta/checkpoint-218` (epoch 1, val ROC-AUC 0.5491).
- **Test metrics:** ROC-AUC 0.5748 [0.5416, 0.6044], f1_pos 0.1754, threshold 0.461. **Below the LightGBM baseline (0.673).**
- **Caveat:** Reduced training (3 epoch / max_len 48) is sub-optimal. Real comparison needs the full Kaggle config (5 epoch / max_len 64, fp16). Treat this as a pipeline smoke test, not the final number.
- **Predictions:** `reports/artifacts/predictions/transformers/wangchanberta.parquet`.
- **Explainability:** SHAP (LightGBM head) + LIME (30 titles) + attention rollout (20 titles) all generated.

### Pending: phayathaibert (278M) · cloud-only

- **M1 status:** OOM on backward at 9 GB MPS pool with batch=8 + grad_accum=4 + max_len=48 + grad_ckpt. Local route closed.
- **Recommended infra:** HF Jobs `--flavor a10g-small`.

### Pending: xlm-roberta-large (560M) · cloud-only

- **M1 status:** Not attempted; would OOM (>10 GB needed for fp32).
- **Recommended infra:** HF Jobs `--flavor a10g-large` or A100.
