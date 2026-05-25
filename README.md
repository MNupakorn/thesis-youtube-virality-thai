# Thai YouTube Virality Prediction — Comparative Study of Thai Transformer Encoders

> **A Comparative Study of Thai Transformer Encoder Models for YouTube Video Virality Prediction**
> WangchanBERTa, PhayaThaiBERT, XLM-RoBERTa-large — with sentiment-augmented features and a multi-modal hybrid extension.

**Department of Applied Statistics and Data Analytics, School of Science, KMITL · Academic Year 2025**

---

## 1. What is this project?

We predict whether a Thai YouTube video will go viral from its **title + channel/content metadata + sentiment signal**. Three Thai-capable transformer **encoders** are fine-tuned and compared (encoders are the right shape for sequence classification — decoder LLMs like Typhoon and OpenThaiGPT were originally proposed in the thesis but we swap them for cleaner, stronger encoder-only architectures of comparable scale):

| Model | Architecture | Params | Strengths for this task |
|---|---|---|---|
| **WangchanBERTa** `airesearch/wangchanberta-base-att-spm-uncased` | RoBERTa | 105 M | Thai SentencePiece, strong on Thai text classification benchmarks |
| **PhayaThaiBERT** `clicknext/phayathaibert` | RoBERTa | 278 M | Thai-specific, newer + much larger Thai corpus than WangchanBERTa, often the SOTA on modern Thai NLP tasks |
| **XLM-RoBERTa-large** `FacebookAI/xlm-roberta-large` | RoBERTa | 560 M | Multilingual, ideal for code-switched Thai+English titles common on YouTube |

All three are encoder-only → classification head with **full fine-tuning** (no QLoRA needed; they fit comfortably on a T4 16 GB).

### Improvements over the baseline thesis design

1. **Proper 3-way split** — Train / Validation / Test (≈ 70 / 15 / 15) with **channel-grouped** stratification on train/val (prevents leakage of the same channel into both) AND **time-based** test set (latest 15% by `published_at` mimics deployment — predicting virality of *future* videos using past data).
2. **Sentiment as a feature** — a state-of-the-art Thai sentiment classifier (`poom-sci/WangchanBERTa-finetuned-sentiment`, trained on Wisesight-3K) is used as a **frozen feature extractor**: each title gets a 4-dim sentiment distribution (pos / neu / neg / q) + arousal (1 − P(neu)) + valence (P(pos) − P(neg)) + polar (P(pos) + P(neg)).
3. **Multi-modal hybrid model (proposed)** — fuses (i) WangchanBERTa CLS embedding, (ii) sentiment distribution, (iii) engineered title features, (iv) channel + temporal + video numerical features → trains both an MLP head (focal loss) **and** a LightGBM head (great for SHAP) on the fused vector.
4. **Engineered text features** — title length (char/token), emoji count, caps ratio, `?` / `!` counts, has_number, curiosity-gap word indicators (Thai + English clickbait lexicon), hyperbolic adjective ratio, code-switching indicator (16 features in total).
5. **Class imbalance** — combination of focal loss + class-weighted loss + undersampling. Imbalance handling **applied to train only**; val/test kept at natural distribution (~9 % viral).
6. **Hyperparameter tuning** — Optuna TPE on the validation set (toggleable in `configs/train.yaml → hpo.enabled`).
7. **Statistical model comparison** — **McNemar's test** (pairwise) + **Cochran's Q test** (all three encoders on the same test set) + bootstrap 95 % CIs (n=1000) for every metric.
8. **Calibration** — Platt + isotonic, fitted on val, evaluated on test, ECE + reliability diagrams.
9. **Explainability** — SHAP (LightGBM on hybrid features), LIME on titles, attention rollout on WangchanBERTa, per-category breakdown, error analysis.
10. **Reproducibility** — every run seeded (`PYTHONHASHSEED`, NumPy, PyTorch, transformers); versions pinned via `uv`; configs in YAML; experiments tracked via **MLflow**.

---

## 2. Project layout

```
.
├── configs/                # YAML: data.yaml, features.yaml, train.yaml, eval.yaml
├── data/                   # raw / interim / processed (gitignored)
├── notebooks/              # 04_transformer_finetune.ipynb (Colab-ready)
├── reports/                # figures + tables + predictions (auto-generated)
├── scripts/                # CLI entry points
├── src/
│   ├── data_collection/    # YouTube Data API v3 collector
│   ├── data_processing/    # clean.py, labels.py (ViralityIndex), splits.py
│   ├── features/           # structural.py, sentiment.py, tfidf.py, transformer_embed.py
│   ├── models/             # baselines.py, transformer_finetune.py, hybrid.py
│   ├── evaluation/         # metrics.py, calibration.py, stats_tests.py
│   └── explainability/     # shap_runner.py
└── tests/                  # pytest unit tests (currently 9 passing)
```

---

## 3. Dataset (current state)

The user-collected dataset under `data/raw/youtube_thai_videos.parquet` contains:

| stat | value |
|---|---|
| rows | 23 431 (after cleaning) |
| unique channels | 82 |
| categories | Gaming (15 852) · Entertainment (5 703) · Music (2 174) |
| time window | 2025-01-01 → 2026-05-23 (≈ 17 months) |
| view-count range | 0 – 175 M |
| missing video_id | 298 (dropped) |

The `ViralityIndex` (z-log-views + z-engagement, per-channel) gives a binary label with **10.16 % positive rate** (top decile per channel). After channel-grouped time-split:

| split | rows | channels | positive rate |
|---|---|---|---|
| train (after undersample 3:1) | 6 964 | 60 | 25.0 % |
| val | 3 143 | 13 | 10.4 % |
| test (latest 15 %) | 3 510 | 73 | 8.8 % |

---

## 4. Results so far (real data)

### 4.1 Baselines (locally on M1, ≈ 5 min for the whole sweep)

| Model | Features | ROC-AUC | F1 (pos) | Recall (pos) |
|---|---|---|---|---|
| **LightGBM** | **structured + TF-IDF** | **0.673 [0.643, 0.700]** | 0.199 | 0.240 |
| XGBoost | structured + TF-IDF | 0.653 [0.621, 0.680] | 0.191 | 0.250 |
| LightGBM | structured | 0.653 [0.623, 0.683] | 0.217 | 0.636 |
| XGBoost | structured | 0.631 | 0.206 | 0.429 |
| LR | TF-IDF | 0.612 | 0.183 | 0.906 |
| LightGBM | TF-IDF only | 0.608 | 0.193 | 0.776 |
| LR | structured | 0.517 | 0.079 | 0.062 |

> Reading: ROC-AUC ~0.67 on a *channel-grouped time-aware* test set is a realistic ceiling for a tabular+TF-IDF baseline. The whole point of adding transformer + sentiment features is to push this above 0.75.

Full numbers: `reports/tables/baselines_metrics.csv`.

### 4.2 Next results (pending GPU run)
- Sentiment-augmented baselines (`feature_set: all`)
- WangchanBERTa fine-tune (locally on M1 MPS or T4)
- PhayaThaiBERT fine-tune (T4 / A100)
- XLM-RoBERTa-large fine-tune (T4 / A100)
- Multi-modal hybrid (MLP + LightGBM heads on fused features)
- Statistical tests (McNemar + Cochran's Q) across the 3 transformers
- SHAP global + local explanations

---

## 5. Where to train? (Cloud GPU recommendations)

Heaviest model is XLM-RoBERTa-large (560 M). It does **not** need an A100 — a single T4 with FP16 + gradient checkpointing + batch size 8 + grad-accumulation 4 is enough. But faster is nicer:

| Provider | GPU | Cost | Notes |
|---|---|---|---|
| **Google Colab Free** | T4 16 GB | $0 | Best free option. Use `notebooks/04_transformer_finetune.ipynb`. WangchanBERTa ~25 min, PhayaThaiBERT ~1 h, XLM-R-large ~2 h. Session limit ~12 h. |
| **Kaggle** | T4 ×2 / P100 | $0 | 30 GPU-h/week free; easier persistence via Kaggle datasets. |
| **Google Colab Pro+** | A100 40 GB | ≈ $50/mo | 3 – 4× faster than T4, no session limits. **Easiest paid option.** |
| **Lambda Labs** | A100 / H100 | ~$1 – 2 /h | Pay-as-you-go, very fast, no quotas. |
| **RunPod** | A100 spot | ~$0.8 – 1.3 /h | Cheapest A100 in cloud; SSH-style workflow. |
| **Modal Labs** | A100 / H100 | ~$1.2 /h | Pay-per-second, great for short jobs; Python-native. |
| **Hugging Face Jobs** | A100 / H100 | ~$1 – 2 /h | Zero setup, just `hf jobs run`. Recommended if you don't want to deal with infra. |

**Recommendation for this thesis**: start free on Colab T4 (works for all 3 models). If iteration speed becomes painful, **Colab Pro+ (A100) or HF Jobs** is the cleanest paid upgrade — same scripts, same results, 3 – 4× faster.

---

## 6. Quickstart

### 6.1 Local setup (Mac / Linux)

```bash
cd thesis-youtube-virality-thai
uv venv --python 3.11
uv pip install -e ".[dev]"
source .venv/bin/activate
```

### 6.2 Data already loaded?

The user-provided `dataset.xlsx` was converted to `data/raw/youtube_thai_videos.parquet`. To re-collect (or expand), drop your YouTube Data API v3 key in `.env` and run:

```bash
echo "YOUTUBE_API_KEY=..." > .env
python scripts/collect_data.py --config configs/data.yaml
```

### 6.3 Pipeline

```bash
# 1. Prepare (clean + label + split + structural features)
python scripts/prepare_data.py --skip-sentiment --skip-embeddings        # fast path
python scripts/prepare_data.py                                            # full (computes sentiment + embeddings)

# 2. Baselines (locally — ~5 min)
python scripts/train_baselines.py

# 3. WangchanBERTa locally on M1 (MPS) — ~30–60 min
python scripts/train_transformer.py --model wangchanberta

# 4. PhayaThaiBERT + XLM-R-large — use the Colab notebook (need GPU)
#    notebooks/04_transformer_finetune.ipynb

# 5. Hybrid model (needs sentiment + embeddings cached from step 1 full)
python scripts/train_hybrid.py

# 6. Evaluate everything + statistical tests
python scripts/evaluate.py

# 7. SHAP explanations
python scripts/explain.py
```

---

## 7. Authors & License

- **Students**: Traipoohrin Suebuswansarn (65050330), Teeranon Chailangka (65050417), Phanudech Phuthoson (65050685)
- **Advisor**: Asst. Prof. Dr. Pornpimol Chaiwuttisak
- **License**: MIT — see `LICENSE`
