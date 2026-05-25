# Thai YouTube Virality Prediction — Comparative Study of Thai Transformer LMs

> **A Comparative Study of Thai Transformer Language Models for YouTube Video Virality Prediction**
> WangchanBERTa, Typhoon 2.5, OpenThaiGPT — with sentiment-augmented features and a multi-modal hybrid extension.

**Department of Applied Statistics and Data Analytics, School of Science, KMITL · Academic Year 2025**

---

## 1. What is this project?

We predict whether a Thai YouTube video will go viral from its **title + channel/content metadata + sentiment signal**. Three Thai transformer language models are fine-tuned and compared:

| Model | Architecture | Params | Role |
|---|---|---|---|
| **WangchanBERTa** | RoBERTa (encoder) | 105M | Text classifier (full FT) |
| **Typhoon 2.5** | Llama/Qwen decoder | 3B | Text classifier (QLoRA) |
| **OpenThaiGPT** | Llama/Qwen decoder | 7B/72B | Text classifier (QLoRA) |

### Extensions over the baseline thesis design

1. **Proper 3-way split** — Train / Validation / Test (70 / 15 / 15) with **channel-grouped** stratification (prevents same-channel leakage) AND **time-based** test set (videos published in the final 15% of the window).
2. **Sentiment as a feature** — a state-of-the-art Thai sentiment classifier (`airesearch/wangchanberta-base-att-spm-uncased` fine-tuned on Wisesight-3K) is used as a **feature extractor**: each title gets a 4-dim sentiment distribution + arousal/valence proxies.
3. **Multi-modal hybrid model (proposed)** — text embedding (WangchanBERTa CLS) ⊕ sentiment distribution ⊕ engineered title features ⊕ channel/temporal numerical features → MLP head + LightGBM on the same fused vector for comparison.
4. **Engineered text features** — title length (char/token), emoji count, caps ratio, `?` / `!` counts, has_number, curiosity-gap word indicators (Thai + English clickbait lexicon), hyperbolic adjective ratio, code-switching indicator.
5. **Class imbalance** — combination of focal loss + class-weighted loss + undersampling (training only); validation/test kept at natural distribution.
6. **Hyperparameter tuning** — Optuna TPE on the validation set; never touches test.
7. **Statistical model comparison** — **McNemar's test** (pairwise) + **Cochran's Q test** (all three models on the same test set) + bootstrap 95% CIs for all metrics.
8. **Explainability** — SHAP (LightGBM on hybrid features), LIME on titles, attention-rollout on WangchanBERTa, per-category breakdown, error analysis.
9. **Reproducibility** — every run is seeded (`PYTHONHASHSEED`, NumPy, PyTorch, transformers), versions pinned via `uv`, configs in YAML, all experiments tracked via **MLflow**.

---

## 2. Project layout

```
.
├── configs/                # YAML configs (data, features, train, eval)
├── data/                   # raw / interim / processed (gitignored)
├── notebooks/              # 01_eda, 02_labels, 03_baselines, 04_transformer, 05_hybrid, 06_explain
├── reports/                # figures + tables for the thesis (auto-generated)
├── scripts/                # CLI entry-points (collect, prepare, train, evaluate, explain)
├── src/
│   ├── data_collection/    # YouTube Data API v3 collector (will be added once user shares script)
│   ├── data_processing/    # cleaning, label engineering (ViralityIndex)
│   ├── features/           # sentiment, emotion, structural, fusion
│   ├── models/             # baselines, transformer FT, hybrid head
│   ├── evaluation/         # metrics, McNemar's / Cochran's Q, calibration
│   └── explainability/     # SHAP, LIME, attention
└── tests/                  # pytest unit tests
```

---

## 3. Quickstart

### 3.1 Local setup (Mac / Linux)

```bash
# Install uv if you don't have it
# curl -LsSf https://astral.sh/uv/install.sh | sh

cd thesis-youtube-virality-thai
uv venv --python 3.11
uv pip install -e ".[dev]"
source .venv/bin/activate
```

### 3.2 Data collection

The original dataset CSV that was uploaded with this thesis was corrupted at the byte level (every UTF-16 high byte stripped, including Thai chars and CSV delimiters — recovery is not possible without re-collection). Once you drop the original collector script into `src/data_collection/`, run:

```bash
# Set your YouTube Data API v3 key
echo "YOUTUBE_API_KEY=your_key_here" > .env

# Collect
python scripts/collect_data.py --config configs/data.yaml
```

The output lands in `data/raw/youtube_thai_videos.parquet`.

### 3.3 Pipeline

```bash
# 1. Prepare features + ViralityIndex labels + splits
python scripts/prepare_data.py --config configs/data.yaml

# 2. Train baselines (LR + LightGBM + XGBoost on structured + TF-IDF)
python scripts/train_baselines.py --config configs/train.yaml

# 3. Fine-tune WangchanBERTa  (~30 min on T4)
python scripts/train_transformer.py --model wangchanberta --config configs/train.yaml

# 4. Fine-tune Typhoon 2.5 / OpenThaiGPT  (QLoRA, run on Colab/Kaggle T4 or A100)
python scripts/train_transformer.py --model typhoon-2.5 --config configs/train.yaml
python scripts/train_transformer.py --model openthaigpt --config configs/train.yaml

# 5. Train the hybrid multi-modal model
python scripts/train_hybrid.py --config configs/train.yaml

# 6. Evaluate all models + statistical tests + calibration
python scripts/evaluate.py --config configs/eval.yaml

# 7. Generate explanations
python scripts/explain.py --config configs/eval.yaml
```

---

## 4. Where to train?

The 3 transformer models do **not** fit on an 8 GB M1 MacBook Air. Recommended:

| Model | Local M1 (8 GB) | Colab Free (T4 16 GB) | Colab Pro / Kaggle (T4×2 / A100) |
|---|---|---|---|
| Baselines (LR / GBM) | ✅ | ✅ | ✅ |
| WangchanBERTa (105 M, full FT) | ⚠️ slow | ✅ ~25 min | ✅ ~10 min |
| Typhoon 2.5 (3 B, QLoRA) | ❌ | ✅ ~2 h | ✅ ~45 min |
| OpenThaiGPT 7 B (QLoRA) | ❌ | ⚠️ tight, 3+ h | ✅ ~1.5 h |
| OpenThaiGPT 72 B | ❌ | ❌ | needs A100 80 GB |

`notebooks/04_transformer_finetune.ipynb` is Colab-ready: it pulls this repo, sets up the env, and runs the fine-tune. For OpenThaiGPT 72B specifically, **Hugging Face Jobs** (paid by the minute) is the cleanest path.

---

## 5. Evaluation protocol

For every model on the held-out test set we report:

- **Primary**: Accuracy, Precision, Recall, F1, **AUC-ROC**, **AUC-PR** (more honest under imbalance)
- **Per-class**: macro/weighted F1, per-class recall
- **Calibration**: ECE, reliability diagram (after Platt / isotonic scaling on val set)
- **Robustness**: per-category breakdown (14 YouTube categories), per-channel-size strata
- **Statistical**: McNemar's test pairwise + Cochran's Q across the 3 transformers + bootstrap 95% CIs (n=1000)

---

## 6. Reproducibility

- All randomness seeded at: `random`, `numpy`, `torch`, `transformers.set_seed`, plus `PYTHONHASHSEED=42`
- Library versions pinned in `pyproject.toml` / `uv.lock`
- Every training run logs to MLflow: params, metrics, artifacts, git commit hash
- Configs are YAML; CLI scripts accept `--config path/to.yaml`
- A `Makefile` provides one-shot commands (`make data`, `make train-all`, `make eval`, `make report`)

---

## 7. Authors & License

- **Students**: Traipoohrin Suebuswansarn (65050330), Teeranon Chailangka (65050417), Phanudech Phuthoson (65050685)
- **Advisor**: Asst. Prof. Dr. Pornpimol Chaiwuttisak
- **License**: MIT (code) — see `LICENSE`
