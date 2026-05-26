# Frozen Decisions

Append-only log. Never edit a prior entry; supersede with a new dated entry that explicitly links back.

---

## 2026-05-26 · Task type: binary classification, encoders only

**Decision:** The thesis is a comparative study of three Thai-capable transformer **encoders** for binary virality classification. Decoder LLMs (Typhoon 2.5, OpenThaiGPT) — originally proposed in the thesis brief — are **excluded from the comparison**.

**Models in comparison (fixed):**
- `airesearch/wangchanberta-base-att-spm-uncased` (105M)
- `clicknext/phayathaibert` (278M)
- `FacebookAI/xlm-roberta-large` (560M)

**Why:** Classification is the right shape for sequence-classification heads, where encoders are more parameter-efficient than decoders. Comparing encoder × encoder controls architecture family and isolates the variable that matters (training corpus, size, multilingual vs. Thai-only).

**Cost of revisiting:** Would invalidate every metric and stats test. Re-trains all three models from scratch.

---

## 2026-05-26 · Label definition: ViralityIndex top-decile per channel

**Decision:** Positive label = top 10% of `ViralityIndex` *within each channel*, where:
- `ViralityIndex = z(log(views+1)) + z(engagement_rate)` computed per channel.
- `engagement_rate = (likes + comments) / max(views, 1)`.

**Resulting global positive rate:** ≈ 10.16%.

**Why per-channel:** raw view counts correlate with channel size; per-channel z-scores ask "viral *for this channel*?" — the deployment-relevant question. Avoids a model that just predicts channel popularity.

**Cost of revisiting:** invalidates labels for all 23,431 rows; re-trains everything.

---

## 2026-05-26 · Splits: channel-grouped (train/val) + time-aware (test)

**Decision:**
- `test` = videos with `published_at` in the latest 15% of the time range.
- `train ∪ val` = the remaining 85%, channel-grouped (no channel appears in both train and val).
- After undersampling (3:1 negative:positive, train-only): train 6,964 · val 3,143 · test 3,510.

**Why time-aware test:** deployment predicts the virality of *future* videos using past data. Random shuffle leaks time-correlated patterns (seasonality, trends, channel growth phases).

**Why channel-grouped train/val:** prevents the model from memorizing channel-specific style and getting credit for it as language signal.

**Cost of revisiting:** invalidates all comparisons; predictions parquet files become non-comparable.

---

## 2026-05-26 · Imbalance handling: train-only

**Decision:**
- Train: focal loss (γ=2.0) + 3:1 undersampling.
- Val & test: untouched (natural ~10% / ~9% positive rates).

**Why:** val/test must reflect the natural distribution to give honest generalization estimates. Imbalance handling on val/test inflates metrics.

---

## 2026-05-26 · Sentiment as a frozen feature extractor

**Decision:** Use `poom-sci/WangchanBERTa-finetuned-sentiment` (Wisesight-3K) as a **frozen** feature extractor. Each title produces a 4-dim distribution (pos, neu, neg, q) + 3 derived (arousal = 1 − P(neu), valence = P(pos) − P(neg), polar = P(pos) + P(neg)).

**Why frozen:** the goal is to study *whether sentiment is a useful feature*, not to fine-tune a new sentiment model. Frozen extraction is reproducible and cheap.

---

## 2026-05-26 · Statistical comparison: McNemar pairwise + Cochran's Q

**Decision:** Compare the 3 encoders with pairwise McNemar's test (Holm-Bonferroni adjusted across 3 pairs) and Cochran's Q across all three at once. Bootstrap 95% CIs (n=1000) on every headline metric.

**Why:** McNemar is the textbook paired-classifier comparison on the same test set; Cochran's Q extends it to k>2. Bootstrap CIs quantify uncertainty without parametric assumptions.

---

## 2026-05-26 · Cloud GPU: Kaggle Kernels as primary

**Decision:** Kaggle Kernels via `kaggle` CLI (`mknpk01`) is the primary cloud-GPU path. Colab is the manual fallback. HF Jobs / RunPod / Modal are only considered if Kaggle weekly quota (30h) is exhausted.

**Why:** free; fully scriptable (no OAuth flow blocking automation); T4 16GB is sufficient for all three models with the configs in `configs/train.yaml`.

---

## How to Supersede

If a decision changes, add a new entry below with a new date, and at the top of the new entry write:

> **Supersedes:** [date · short title]
> **Reason:** [one paragraph]

Never delete the old entry. The audit trail is the point.

## 2026-05-26 · Known bug: train_hybrid.py silently skips GBM block when MLP runs first

**Decision:** Acknowledged but **not patched**. Workaround documented.

**Symptom:** Running `scripts/train_hybrid.py` end-to-end: MLP block finishes, prints `hybrid_mlp test: ...`, saves `mlp_state_dict.joblib` + `feature_names.joblib`, then the process exits cleanly (exit 0) **without ever entering the GBM block**, despite `also_train_gbm_head: true`.

**Diagnosis:** Likely a segfault inside `train_gbm_head` when invoked inside an MLflow run context immediately after MPS-backed PyTorch work in a preceding run. Reproduces deterministically on macOS arm64 (M1) + transformers v5 + lightgbm 4.x + mlflow file backend. Running `train_gbm_head` in isolation (no preceding MPS work) completes in ~10 s.

**Workaround (current):** GBM head trained out-of-band with a one-liner that calls `train_gbm_head` + `save_hybrid_artifacts` directly. `X_test.npy` / `y_test.npy` saved by a separate snippet. All downstream consumers (SHAP, evaluate) are unblocked.

**Proper fix later:** split MLP and GBM into two scripts, or explicitly clear MPS state (`torch.mps.empty_cache(); del mlp_state`) between blocks before re-running the hybrid pipeline on revised data.

---
