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

### 2026-05-26 · wangchanberta · title · focal · HF Jobs t4-small (full config 5 epoch)

- **Job ID:** `6a156bd8f17429a271eee137` on HF Jobs (`t4-small`, ~$0.40/h)
- **Config:** `configs/train.yaml` (full: epochs=5, batch=32, max_len=64, fp16)
- **Test metrics:** ROC-AUC 0.6402 [0.6100, 0.6699], f1_pos 0.197, threshold 0.422
- **Output repo:** `MGodK/thesis-output-wangchanberta-20260526`
- **Predictions:** `reports/artifacts/predictions/transformers/wangchanberta.parquet`
- **Note:** Replaces the M1-reduced 3-epoch run from earlier today.

### 2026-05-26 · phayathaibert · title · focal · HF Jobs t4-small (full config 4 epoch)

- **Job ID:** `6a156bda404eb93b204f23d2` on HF Jobs (`t4-small`)
- **Config:** `configs/train.yaml` (epochs=4, batch=16, grad_accum=2, max_len=96, fp16)
- **Best checkpoint:** `checkpoint-872` (val_roc_auc 0.5938)
- **Test metrics:** ROC-AUC 0.6451 [0.6143, 0.6757], f1_pos 0.203, threshold 0.461
- **Output repo:** `MGodK/thesis-output-phayathaibert-20260526`
- **Best of the three encoders.**

### 2026-05-26 · xlm-roberta-large · title · focal · HF Jobs a10g-small (full config 4 epoch)

- **Job ID:** `6a156bddf17429a271eee139` on HF Jobs (`a10g-small`, ~$1.00/h)
- **Config:** `configs/train.yaml` (epochs=4, batch=8, grad_accum=4, gradient_checkpointing, fp16)
- **Test metrics:** ROC-AUC 0.5450 [0.5110, 0.5775], f1_pos 0.166, threshold 0.500
- **Output repo:** `MGodK/thesis-output-xlm-roberta-large-20260526`
- **Note:** Significantly worse than the Thai-specific encoders. Multilingual model under-fits the Thai title distribution at this sample size.

### 2026-05-26 · 3-encoder comparison (the thesis headline)

- **Cochran's Q:** stat=612.69, df=2, p ≈ 9.0e-134 → encoders differ.
- **Pairwise McNemar (b/w-corrected):**
  - PhayaThaiBERT vs WangchanBERTa: 450/303, p ≈ 1.0e-7 (Phaya wins)
  - PhayaThaiBERT vs XLM-R-large: 1032/261, p ≈ 1.0e-101 (Phaya wins)
  - WangchanBERTa vs XLM-R-large: 930/306, p ≈ 2.9e-70 (Wangchan wins)
- **Ordering on test set:** PhayaThaiBERT > WangchanBERTa > XLM-R-large.
- **Headline finding:** all 3 encoders are still beaten by LightGBM with structured + TF-IDF features (0.673 vs 0.645 best encoder). Suggests Thai-YouTube virality is driven more by metadata/structural signals than by title text alone.

### 2026-05-26 · Stacking ensemble (LR meta) + ablations · the A+ push

**Best ensemble:** `stacking_lr_calibrated` — LR over 15 base models, Platt-calibrated.
- **Test ROC-AUC: 0.6914 [0.6625, 0.7174]** ← new project-best, +1.86 pt over LightGBM (0.6728), +4.6 pt over best encoder PhayaThaiBERT (0.6451)
- **ECE before / after Platt:** 0.337 → 0.016 (95 % calibration improvement)
- **Threshold (val-tuned):** 0.1472 for calibrated probas

**Drop-one ablation** (`reports/tables/stacking_drop_one_ablation.csv`):
- Largest contributors: `baselines/lightgbm_structured_plus_tfidf` (-0.0047), `transformers/phayathaibert` (-0.0041), `baselines/logistic_regression_structured_plus_tfidf` (-0.0030).
- `transformers/xlm-roberta-large` contributes ~0 — confirms its under-performance.
- Dropping `xgboost_structured_plus_tfidf` or `xgboost_tfidf` slightly *improves* AUC — they're slight harm.

**Sub-population breakdown** (`reports/tables/stacking_by_channel_size.csv`):
- Large channels (>1M subs, n=2105): AUC 0.7459 [0.7028, 0.7862]
- Mid channels (100k-1M, n=1388): AUC 0.6273 [0.5819, 0.6686]
- Striking ~12 pt gap — viral signal much stronger in big channels (more data per channel, established patterns).

**Pruned ensemble** (top-7 important models, C=0.3): AUC 0.6901 — slightly *worse* than full 15-model. Negative-coef models help via regularization.

**Multi-field PhayaThaiBERT ablation** (`reports/artifacts/predictions/ablation/phayathaibert_multifield.parquet`):
- Config: text_cols=[title, description, channel_title], max_length=192. HF Job `6a157b055c8d10ffa1101a70`.
- Test ROC-AUC: 0.5455 — *much worse* than title-only PhayaThaiBERT (0.6451).
- Honest finding: Thai YouTube descriptions are mostly boilerplate / CTA / cross-promo; longer text adds noise, not signal. Discussion-section material.

**Channel-prior baseline** (`reports/artifacts/predictions/baselines/channel_prior.parquet`):
- Per-channel mean(label_viral) computed from train videos; applied to test (59/60 train channels appear in test). Anti-correlated with test labels (alone AUC=0.293) due to undersampling-driven bias + popularity decay.
- Adds ~0 to stacking AUC — train structural features (`log_subscriber_count`, etc.) already capture this.

### 2026-05-26 · PhayaThaiBERT HPO sweep (3 trials on HF Jobs t4-small)

**All trials underperform the default config — null result, kept honestly.**

| Trial | Config diff | Test ROC-AUC | Δ vs default |
|---|---|---|---|
| default | lr=1.0e-5, ep=4 | 0.6451 | — |
| lr5e6_ep6 | lr=5.0e-6, ep=6 | 0.6414 | -0.0037 |
| lr2e5_ep4 | lr=2.0e-5, ep=4 | 0.6437 | -0.0014 |
| lr1e5_ep6 | lr=1.0e-5, ep=6, warmup=0.06 | 0.6448 | -0.0003 |

**Job IDs:** 6a15851f5c8d10ffa1101b25, 6a1585225c8d10ffa1101b29, 6a1585255c8d10ffa1101b2b.
**Output repos:** `MGodK/thesis-output-phaya-{trial}-20260526`.
**Predictions saved:** `reports/artifacts/predictions/ablation/hpo/phaya_*.parquet`.

**Interpretation:** the default config (4 epochs, lr=1e-5, warmup_ratio=0.10, batch=16/grad_accum=2) is at or near the local optimum for this task. Longer training overfits; aggressive LR causes instability; longer training with smaller warmup gains nothing. Reinforces the claim that the encoder upper-bound on title-only Thai-YouTube virality classification is around 0.645 ROC-AUC.
