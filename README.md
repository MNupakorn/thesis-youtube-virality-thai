# การพยากรณ์ความไวรัลของวิดีโอ YouTube ภาษาไทย

> **A Comparative Study of Thai Transformer Encoder Models for YouTube Video Virality Prediction**
>
> เปรียบเทียบ WangchanBERTa, PhayaThaiBERT และ XLM-RoBERTa-large ในการพยากรณ์ความไวรัลของวิดีโอ YouTube ภาษาไทย

**ปัญหาพิเศษ — ภาควิชาสถิติประยุกต์และการวิเคราะห์ข้อมูล คณะวิทยาศาสตร์**
**สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง · ปีการศึกษา 2568**

**คณะผู้จัดทำ**
- นายตรัยภูรินท์ สืบสุวรรณสาร (รหัสนักศึกษา 65050330)
- นายธีรนนท์ ไชยลังกา (รหัสนักศึกษา 65050417)
- นายภาณุเดช ภู่โทสนธิ์ (รหัสนักศึกษา 65050685)

**อาจารย์ที่ปรึกษา:** ผศ. ดร. พรพิมล ชัยวุฒิศักดิ์

---

## สารบัญ

1. [โปรเจกต์นี้คืออะไร](#1-โปรเจกต์นี้คืออะไร)
2. [ผลลัพธ์หลัก (Headline Results)](#2-ผลลัพธ์หลัก-headline-results)
3. [Quick Start (เริ่มต้นใช้งาน)](#3-quick-start-เริ่มต้นใช้งาน)
4. [โครงสร้างโปรเจกต์](#4-โครงสร้างโปรเจกต์)
5. [ภาพรวมขั้นตอน (Pipeline)](#5-ภาพรวมขั้นตอน-pipeline)
6. [คำอธิบาย Scripts ทั้งหมด](#6-คำอธิบาย-scripts-ทั้งหมด)
7. [คำอธิบายโมดูลใน src/](#7-คำอธิบายโมดูลใน-src)
8. [คำอธิบาย Configs](#8-คำอธิบาย-configs)
9. [Cloud GPU (Hugging Face Jobs)](#9-cloud-gpu-hugging-face-jobs)
10. [Outputs และ Reports](#10-outputs-และ-reports)
11. [การทำซ้ำ (Reproducibility)](#11-การทำซ้ำ-reproducibility)
12. [License และเครดิต](#12-license-และเครดิต)

---

## 1. โปรเจกต์นี้คืออะไร

โปรเจกต์นี้พัฒนาระบบ **พยากรณ์ความไวรัล** (virality prediction) ของวิดีโอ YouTube ภาษาไทย เป็นปัญหา **การจำแนกประเภททวิภาค** (binary classification) ที่ตอบคำถามว่า "วิดีโอที่กำลังจะปล่อยจะกลายเป็นไวรัลหรือไม่" โดยใช้ข้อมูลที่สังเกตได้ก่อนเผยแพร่เท่านั้น คือ

- **ชื่อวิดีโอ (title)** — ข้อความภาษาไทย
- **คำอธิบาย (description) + แท็ก (tags)**
- **Meta-data ของช่อง** — ผู้ติดตาม จำนวนวิดีโอ ความถี่อัปโหลด อายุช่อง
- **ฟีเจอร์เชิงเวลา** — ชั่วโมง วัน วันหยุด เทศกาล

### โจทย์ทางวิชาการ

1. **WangchanBERTa, PhayaThaiBERT, XLM-RoBERTa-large** ตัวไหนทำงานดีที่สุดในงานนี้?
2. การ **ประกอบหลายแบบจำลอง** (stacking ensemble) ดีกว่าแบบจำลองเดี่ยวอย่างมีนัยสำคัญทางสถิติหรือไม่?
3. **รูปแบบความผิดพลาด** ของแต่ละแบบจำลองต่างกันอย่างไร? (McNemar / Cochran's Q)
4. แบบจำลอง **คาลิเบรต** ความน่าจะเป็นได้ดีแค่ไหน? (ECE / reliability diagram)
5. แบบจำลอง **ใช้สัญญาณอะไร** ในการตัดสินใจ? (SHAP / LIME / Attention rollout)

---

## 2. ผลลัพธ์หลัก (Headline Results)

| Metric | ค่า |
|---|---|
| **Top model:** Stacking ensemble ที่ปรับเทียบ Platt | `stacking_lr_calibrated` |
| **Test ROC-AUC** | **0.6914** [0.6625, 0.7174] (bootstrap 95% CI) |
| **ECE** (15 bins) | **0.016** (ปรับเทียบดีมาก) |
| **Best single model** | LightGBM Structured+Embeddings — 0.6728 |
| **Best transformer** | PhayaThaiBERT — 0.6451 |
| **Cochran's Q** (3 encoders) | $Q = 612.69$, $p = 9 \times 10^{-134}$ |
| **Stacking vs LightGBM** McNemar | $p = 6.9 \times 10^{-53}$ |
| **5-fold robustness** mean ± SD | 0.6936 ± 0.0982 |
| **Channel-bootstrap CI** (500 iter) | [0.640, 0.753] |

### ขนาดชุดข้อมูล

- **23,431 วิดีโอ** จาก **82 ช่อง** ภาษาไทย
- อัตราชั้นบวก (positive rate) ตามธรรมชาติ: **10.16%**
- การแบ่ง: **train 6,964** (undersampled 3:1) / **val 3,143** (natural) / **test 3,510** (latest 15% by time)
- **Time-aware test set** = วิดีโอที่เผยแพร่ใน 15% ท้ายสุดของช่วงเวลา

---

## 3. Quick Start (เริ่มต้นใช้งาน)

### ติดตั้ง

```bash
# ติดตั้ง uv (Python package manager) ถ้ายังไม่มี
curl -LsSf https://astral.sh/uv/install.sh | sh

# สร้าง venv และติดตั้ง deps
make install
# หรือ: uv venv --python 3.11 && uv pip install -e ".[dev]"
```

### ตั้งค่า API key

สร้างไฟล์ `.env.local` (อยู่ใน `.gitignore` แล้ว) :

```bash
YOUTUBE_API_KEY=AIza...your-key
HUGGINGFACE_TOKEN=hf_...optional-for-cloud-training
```

### รันทั้ง pipeline

```bash
# 1) ดึงข้อมูลจาก YouTube Data API v3 (1-2 ชั่วโมง)
make data

# 2) ทำความสะอาด + คำนวณ Virality Index + แบ่งข้อมูล + สกัดฟีเจอร์
make prepare

# 3) ฝึก baselines (LR/LightGBM/XGBoost) - บน CPU ใช้เวลา ~10 นาที
make train-baselines

# 4) ฝึกทรานส์ฟอร์เมอร์บน Cloud GPU (HF Jobs ~$0.50 รวม)
make train-hf-all-detached

# 5) ฝึก hybrid และ stacking ensemble
make train-hybrid
uv run python scripts/train_stacking.py
uv run python scripts/stacking_drop_one.py

# 6) ประเมินผล (ROC, McNemar, Cochran's Q, calibration)
make eval

# 7) สร้าง explainability artifacts (SHAP / LIME / attention)
make explain

# 8) สร้างกราฟวิทยานิพนธ์ขั้นสุดท้าย
uv run python scripts/thesis_figures.py
```

### ดูคำสั่งทั้งหมด

```bash
make help
```

---

## 4. โครงสร้างโปรเจกต์

```
thesis-youtube-virality-thai/
├── README.md                    # ไฟล์นี้
├── CLAUDE.md                    # กฎการพัฒนา (สำหรับ AI agent)
├── MEMORY.md                    # บันทึกสถานะของโปรเจกต์
├── Makefile                     # คำสั่งหลัก: make {data,prepare,train-*,eval,explain}
├── pyproject.toml               # การประกาศ deps ผ่าน uv
├── uv.lock                      # ล็อกเวอร์ชัน deps แบบเฉพาะ
├── .env.local                   # secret keys (gitignored)
│
├── configs/                     # YAML ทุกไฟล์ของ HP และ pipeline
│   ├── data.yaml                # นโยบายเก็บข้อมูล + การแบ่งชุด
│   ├── features.yaml            # การกำหนดฟีเจอร์
│   ├── train.yaml               # HP ของทุกแบบจำลอง
│   ├── train_m1.yaml            # config สำหรับ Apple M1 (smaller batch)
│   ├── eval.yaml                # การประเมินผล + bootstrap + stats tests
│   └── train_phaya_*.yaml       # HPO trials ของ PhayaThaiBERT
│
├── src/                         # ไลบรารีหลัก (ใช้ซ้ำได้, มี unit tests)
│   ├── utils.py                 # logger, set_seed, load_yaml, ensure_dir
│   ├── data_collection/         # ดึงข้อมูลจาก YouTube API
│   ├── data_processing/         # clean / labels / splits
│   ├── features/                # structural / sentiment / tfidf / embeddings
│   ├── models/                  # baselines / hybrid / transformer_finetune
│   ├── evaluation/              # metrics / calibration / stats_tests
│   └── explainability/          # SHAP / LIME / attention rollout
│
├── scripts/                     # CLI entry points (รันได้ตรง ๆ)
│   ├── collect_data.py          # ดึงข้อมูล
│   ├── prepare_data.py          # ทำความสะอาด + label + split + features
│   ├── train_baselines.py       # ฝึก LR/LightGBM/XGBoost
│   ├── train_transformer.py     # ฝึก WangchanBERTa/PhayaThaiBERT/XLM-R
│   ├── train_hybrid.py          # ฝึก MLP + LightGBM head
│   ├── train_stacking.py        # ฝึก meta-LR สำหรับ stacking ensemble
│   ├── stacking_drop_one.py     # ablation: ตัดแบบจำลองออกทีละตัว
│   ├── evaluate.py              # ROC/PR/McNemar/Cochran's Q/calibration
│   ├── explain.py               # SHAP/LIME/attention rollout
│   ├── thesis_figures.py        # สร้างกราฟวิทยานิพนธ์ขั้นสุดท้าย
│   └── cloud/                   # การ orchestrate cloud GPU
│       ├── build_kaggle_dataset.py     # อัปเดต Kaggle private dataset
│       ├── kaggle_kernel_template.py   # เทมเพลต kernel script
│       ├── run_on_kaggle.py            # push + poll Kaggle kernel
│       ├── build_hf_dataset.py         # อัปข้อมูลขึ้น HF private dataset
│       └── run_on_hf_jobs.py           # submit + poll HF Jobs
│
├── data/                        # gitignored
│   ├── raw/                     # YouTube API dumps (parquet)
│   ├── interim/                 # cache ของฟีเจอร์
│   └── processed/               # canonical input: dataset_with_labels.parquet
│
├── reports/                     # output: ตาราง + กราฟ + predictions
│   ├── tables/                  # CSV (metrics, McNemar, Cochran's Q, ablations)
│   ├── figures/                 # SVG/PNG (ROC, PR, calibration, SHAP)
│   └── artifacts/
│       ├── predictions/         # parquet ของทุกแบบจำลอง (video_id, split, y_true, y_proba)
│       └── models/              # model checkpoints
│
├── mlruns/                      # MLflow tracking (file backend)
├── tests/                       # pytest unit tests
├── notebooks/                   # Colab fallback (.ipynb)
└── docs/thesis/                 # LaTeX วิทยานิพนธ์ (XeLaTeX, KMITL format)
```

---

## 5. ภาพรวมขั้นตอน (Pipeline)

โครงงานทั้งหมดเป็น pipeline 8 ระยะที่ทำงานจากข้อมูลดิบไปจนถึงรายงาน:

```
[1] ดึงข้อมูล              →  scripts/collect_data.py        →  data/raw/*.parquet
[2] ทำความสะอาด+label      →  scripts/prepare_data.py        →  data/processed/dataset_with_labels.parquet
[3] สกัดฟีเจอร์            →  src/features/*                 →  data/interim/*
[4] ฝึก baselines           →  scripts/train_baselines.py     →  reports/artifacts/predictions/baselines/
[5] ฝึก transformers       →  scripts/train_transformer.py   →  reports/artifacts/predictions/transformers/
[6] ฝึก hybrid + stacking  →  scripts/train_hybrid.py
                              scripts/train_stacking.py     →  reports/artifacts/predictions/{hybrid,stacking}/
[7] ประเมินผล              →  scripts/evaluate.py            →  reports/tables/*.csv
[8] อธิบายแบบจำลอง         →  scripts/explain.py             →  reports/figures/{shap,lime,attention}/
```

ทุกขั้นตอน (5)–(8) อ่าน predictions parquet ของ (4)–(6) เป็นอินพุต **ไม่ต้อง re-train** เมื่อรันซ้ำ เพื่อให้การวิเคราะห์ทำซ้ำได้แบบไม่กิน GPU ใหม่

---

## 6. คำอธิบาย Scripts ทั้งหมด

### `scripts/collect_data.py` — เก็บข้อมูลจาก YouTube API

**หน้าที่:** เรียก YouTube Data API v3 (`videos.list`, `channels.list`) เก็บ meta-data ของวิดีโอจากช่องเป้าหมาย แล้ว save เป็น parquet

**วิธีใช้:**
```bash
make data
# หรือ
uv run python scripts/collect_data.py --config configs/data.yaml
```

**อินพุต:** `configs/data.yaml` (รายการช่อง, snapshot date, quota policy)
**เอาต์พุต:** `data/raw/youtube_thai_videos.parquet` (~23,431 rows)

**สิ่งที่ทำ:**
1. Authenticate กับ YouTube API ผ่าน `YOUTUBE_API_KEY` ใน `.env.local`
2. ดึงรายการวิดีโอของแต่ละช่อง 50 รายการ/request (page จนหมด)
3. ดึง meta-data ของแต่ละช่อง (subscribers, video_count, view_count)
4. เคารพ rate limit (10,000 units/day) ผ่าน exponential backoff
5. Save เป็น parquet พร้อม timestamp ของ snapshot

---

### `scripts/prepare_data.py` — ทำความสะอาด + label + split + features

**หน้าที่:** เปลี่ยนข้อมูลดิบเป็น **input มาตรฐาน** ของระบบ

**วิธีใช้:**
```bash
make prepare
```

**สิ่งที่ทำตามลำดับ:**
1. **Clean** (`src/data_processing/clean.py`)
   - ตัดวิดีโอที่ `view_count` < 100 หรือ NULL
   - ตัดชื่อที่ว่างเปล่าหรือไม่ใช่ภาษาไทย
   - แปลง `duration` ISO 8601 → วินาที
   - คำนวณอายุของวิดีโอเป็นวัน
   - Deduplicate ตาม `video_id`

2. **Label** (`src/data_processing/labels.py`)
   - คำนวณ **Per-Channel Virality Index** ($\mathit{VI}$) ตามสูตรในบทที่ 3 ของวิทยานิพนธ์:
     - $\mathit{VI}_i = 0.5 z_{eng,i} + 0.3 z_{like,i} + 0.2 z_{cmt,i}$
     - โดย $z$ คือ z-score ของอัตราส่วน (engagement/like/comment) **ภายในช่อง**
   - กำหนด `label_viral = 1` ถ้า $\mathit{VI}$ อยู่ใน **top decile per channel**

3. **Split** (`src/data_processing/splits.py`)
   - Test = 15% หลังสุดตาม `published_at` (time-aware)
   - Train/Val = channel-grouped (ไม่ให้ช่องปนข้ามกลุ่ม)
   - **Undersample เฉพาะ train fold** ในอัตรา 3:1 (negative:positive)

4. **Features** (เก็บ cache)
   - Structural 27 ตัว → `src/features/structural.py`
   - Sentiment 6 ตัว → `data/interim/sentiment_cache.parquet`
   - Title embeddings 768 มิติ → `data/interim/title_embeddings.npy`

**เอาต์พุต:** `data/processed/dataset_with_labels.parquet` (canonical input ของทุก downstream)

---

### `scripts/train_baselines.py` — ฝึก Logistic Regression / LightGBM / XGBoost

**หน้าที่:** ฝึก baseline 3 อัลกอริทึม × 4 ชุดฟีเจอร์ = 12 runs

**วิธีใช้:**
```bash
make train-baselines
# หรือ
uv run python scripts/train_baselines.py --config configs/train.yaml
```

**ชุดฟีเจอร์ที่ทดสอบ:**
- `structured` — 27 structural features เท่านั้น
- `tfidf` — character-N-gram TF-IDF (20,000 dims)
- `structured_plus_tfidf` — ผสม structural + 768-dim title embeddings (legacy key — ดูหมายเหตุใน CLAUDE.md)
- `channel_prior` — baseline เปรียบเทียบที่พยากรณ์ค่าเฉลี่ยของช่อง

**เอาต์พุต:**
- `reports/artifacts/predictions/baselines/{model}_{feature_set}.parquet`
- ทุก parquet มีคอลัมน์ `(video_id, split, y_true, y_proba)`
- MLflow run พร้อม config + git SHA + dataset hash

**ทุก run จะ log ลง MLflow** (`mlruns/`) สามารถดูด้วย `uv run mlflow ui`

---

### `scripts/train_transformer.py` — ฝึกทรานส์ฟอร์เมอร์ภาษาไทย

**หน้าที่:** Fine-tune แบบเต็มของ WangchanBERTa, PhayaThaiBERT, หรือ XLM-RoBERTa-large

**วิธีใช้:**
```bash
# Local (M1 หรือ Linux ที่มี GPU)
make train-wangchan      # 105M params, ~10 นาที บน T4
make train-phaya         # 278M params, ~12 นาที บน A10G
make train-xlmr          # 560M params, ~14 นาที บน A10G

# หรือ
uv run python scripts/train_transformer.py --model phayathaibert --config configs/train.yaml
```

**Wrapper เดียวสำหรับทั้งสามแบบจำลอง** ตามกฎของ CLAUDE.md เพื่อความยุติธรรมในการเปรียบเทียบ

**สิ่งที่ทำ:**
1. โหลดแบบจำลองที่ฝึกล่วงหน้าจาก Hugging Face Hub
2. Tokenize ชื่อวิดีโอ (max_length=128)
3. ฝึก 3 epoch ด้วย Adam ($\eta = 2 \times 10^{-5}$), warmup_ratio=0.06, weight_decay=0.01
4. Early stopping ที่ patience=2 (เกณฑ์ eval_roc_auc)
5. fp16 mixed precision (auto-disable เมื่อไม่มี CUDA)
6. ทำนายบน train/val/test → save เป็น parquet

**เอาต์พุต:** `reports/artifacts/predictions/transformers/{model}.parquet`

---

### `scripts/train_hybrid.py` — Multi-modal Hybrid Model

**หน้าที่:** ฝึก **MLP head** และ **LightGBM head** บน feature vector ที่รวม

```
input = [title_embedding (768)] + [structural (27)] + [sentiment (6)]
      = 801 มิติ
```

**วิธีใช้:**
```bash
make train-hybrid
```

**สิ่งที่ทำ:**
- **MLP head**: 801 → 256 → 64 → 2, ReLU + dropout 0.3, Adam, focal loss
- **LightGBM head**: ตัวเดียวกันแต่ tree-based, log สำคัญสำหรับ SHAP

**เอาต์พุต:** `reports/artifacts/predictions/hybrid/{hybrid_mlp,hybrid_gbm}.parquet`

**หมายเหตุ:** มี known issue บน Apple M1 ที่ GBM block ถูก skip เงียบ ๆ หลัง MLP — workaround ใน `.claude/memory/decisions.md`

---

### `scripts/train_stacking.py` — Stacking Ensemble

**หน้าที่:** ฝึก **meta-LR** ที่เรียนวิธีรวมความน่าจะเป็นจากแบบจำลองฐาน 15 ตัว

**วิธีใช้:**
```bash
uv run python scripts/train_stacking.py --head both
# --head ∈ {lr, lgbm, both}
```

**สิ่งที่ทำ:**
1. อ่าน `*.parquet` ทุกไฟล์ใน `reports/artifacts/predictions/`
2. สร้างเมทริกซ์ความน่าจะเป็น $X^{val} \in \mathbb{R}^{n_{val} \times 15}$
3. Fit Logistic Regression meta-learner บน validation fold
4. ทำนายบน test → save parquet
5. (Optional) Fit LightGBM head เป็น alternative meta-learner

**เอาต์พุต:**
- `reports/artifacts/predictions/stacking/stacking_lr.parquet`
- `reports/artifacts/predictions/stacking/stacking_lgbm.parquet`
- `reports/artifacts/predictions/stacking/stacking_feature_importance.csv`

---

### `scripts/stacking_drop_one.py` — Drop-One Ablation

**หน้าที่:** ตัด base model ออกทีละตัวจาก stacking → วัดว่า ROC-AUC ลดลงเท่าไหร่ = ระบุแบบจำลองที่สำคัญที่สุด

**วิธีใช้:**
```bash
uv run python scripts/stacking_drop_one.py
```

**เอาต์พุต:** `reports/tables/stacking_drop_one_ablation.csv`

จากผลใน vitanat: **LightGBM-best** (delta = $-0.0047$) และ **PhayaThaiBERT** (delta = $-0.0041$) สำคัญที่สุด

---

### `scripts/evaluate.py` — การประเมินผลแบบครบวงจร

**หน้าที่:** อ่าน predictions parquet ของทุกแบบจำลอง → คำนวณ metric + statistical tests + calibration

**วิธีใช้:**
```bash
make eval
```

**สิ่งที่ทำ:**
1. **Metrics ครบ** — accuracy, precision/recall/F1 (positive + macro), MCC, ROC-AUC, PR-AUC พร้อม **bootstrap 95% CI** (n=1000)
2. **Threshold optimal** — scan $\theta \in \{0.01, ..., 0.99\}$ บน val, ใช้ค่าที่ optimise F1$_+$
3. **Confusion matrix** — TP/FP/TN/FN ทุกแบบจำลอง
4. **McNemar's test** — pairwise สำหรับทุกคู่ที่สนใจ
5. **Cochran's Q test** — multi-classifier (2 versions: 22-model, 3-encoder)
6. **Calibration** — Platt + Isotonic + ECE 15-bin
7. **Sub-population** — แยกผลตามขนาดช่อง + หมวดหมู่ YouTube
8. **Top-50 errors** — บันทึก false positives/negatives ที่มั่นใจสูงสุด

**เอาต์พุต** ใน `reports/tables/`:
- `all_models_metrics.csv`
- `mcnemar_pairwise.csv`, `mcnemar_pairwise_encoders.csv`
- `cochrans_q.csv`, `cochrans_q_encoders.csv`
- `calibration_*.csv` (3 วิธี × 3 model)
- `per_category_breakdown.csv`, `stacking_by_channel_size.csv`
- `errors_top_fp_*.csv`, `errors_top_fn_*.csv`

---

### `scripts/explain.py` — Explainability

**หน้าที่:** สร้าง SHAP, LIME, Attention rollout

**วิธีใช้:**
```bash
make explain
# หรือเฉพาะบางตัว:
uv run python scripts/explain.py --only shap
uv run python scripts/explain.py --only lime
uv run python scripts/explain.py --only attention
```

**สิ่งที่ทำ:**
- **SHAP** บน LightGBM-best ผ่าน `TreeExplainer` → mean $|SHAP|$ ต่อฟีเจอร์
- **LIME** บน PhayaThaiBERT 50 ตัวอย่าง (25 ถูก / 25 ผิด)
- **Attention rollout** (Abnar & Zuidema 2020) บน PhayaThaiBERT 20 ตัวอย่าง

**เอาต์พุต:** `reports/figures/{shap_summary.png, lime/, attention/}` + CSV ที่เกี่ยวข้อง

---

### `scripts/thesis_figures.py` — กราฟสำหรับวิทยานิพนธ์

**หน้าที่:** สร้างกราฟ **โปรดักชัน** (SVG + PNG) จาก CSV ที่ `evaluate.py` สร้างไว้

**วิธีใช้:**
```bash
uv run python scripts/thesis_figures.py
```

**กราฟที่สร้าง:**
- `leaderboard_with_ci.svg` — ranked bar chart พร้อม CI 95%
- `robustness_slices.svg` — per-category + per-size sub-population
- `reliability_stacking.svg` — calibration curve (raw vs Platt)
- `roc_pr_test.svg` — ROC + PR ของ 6 แบบจำลองต้น

---

## 7. คำอธิบายโมดูลใน `src/`

### `src/utils.py` — utilities ที่ใช้ร่วมกัน

```python
from src.utils import setup_logger, set_seed, load_yaml, ensure_dir
```

- `setup_logger(name)` — logger ที่ใช้ทั้งโปรเจกต์ (ห้าม `print()`)
- `set_seed(seed)` — seed numpy / torch / transformers / PYTHONHASHSEED
- `load_yaml(path)` — อ่าน YAML config
- `ensure_dir(path)` — สร้าง dir ถ้ายังไม่มี

---

### `src/data_processing/`

| ไฟล์ | หน้าที่ |
|---|---|
| `clean.py` | ลบ NULL/duplicates, แปลง duration, เช็คภาษา |
| `labels.py` | คำนวณ Per-Channel Virality Index + binary label |
| `splits.py` | channel-grouped (train/val) + time-aware (test) + undersample |

---

### `src/features/`

| ไฟล์ | หน้าที่ | จำนวนมิติ |
|---|---|---|
| `structural.py` | ฟีเจอร์เชิงโครงสร้าง 27 ตัว (title length, emoji, channel size, time-of-day cyclical) | 27 |
| `sentiment.py` | คะแนนอารมณ์ของชื่อจาก WangchanBERTa-Wisesight | 6 |
| `tfidf.py` | character-N-gram TF-IDF, ngram=(2,5), max_features=20000 | 20,000 |
| `transformer_embed.py` | mean-pooled embeddings จาก SBERT mpnet | 768 |

ทุก feature module มี cache (parquet หรือ npy) ที่ถูก hash กับ input เพื่อไม่ recompute เปล่า ๆ

---

### `src/models/`

| ไฟล์ | หน้าที่ |
|---|---|
| `baselines.py` | wrapper รวม LR / LightGBM / XGBoost ใน API เดียว |
| `transformer_finetune.py` | wrapper เดียวสำหรับ encoder fine-tuning ทั้งสามตัว + QLoRA path สำหรับ decoder LLMs (ไม่ใช้ในการเปรียบเทียบหลัก) |
| `hybrid.py` | MLP head + LightGBM head บน fused feature vector |

---

### `src/evaluation/`

| ไฟล์ | หน้าที่ |
|---|---|
| `metrics.py` | classification metrics, `bootstrap_ci()`, `find_best_threshold()` |
| `calibration.py` | Platt scaling, isotonic regression, `expected_calibration_error()`, `reliability_curve()` |
| `stats_tests.py` | `mcnemar()`, `cochran_q()` |

---

### `src/explainability/`

| ไฟล์ | หน้าที่ |
|---|---|
| `shap_runner.py` | TreeSHAP บน LightGBM, generate `shap_global_importance.csv` + summary plot |
| `lime_runner.py` | LIME บน PhayaThaiBERT, generate per-example HTML + aggregate token CSV |
| `attention_runner.py` | Attention rollout (Abnar & Zuidema) บน PhayaThaiBERT, save heatmaps |

---

## 8. คำอธิบาย Configs

ไฟล์ YAML ทุกตัวอยู่ใน `configs/` ตามกฎ "configs over flags" (CLAUDE.md) — **ห้ามฮาร์ดโค้ด HP ใน Python**

### `configs/data.yaml`
- กำหนดเงื่อนไขช่องเป้าหมาย (ผู้ติดตามขั้นต่ำ, หมวดหมู่)
- นโยบายการแบ่งข้อมูล (test_ratio, val_ratio, undersample ratio)
- snapshot_date

### `configs/features.yaml`
- เปิด/ปิดแต่ละกลุ่มฟีเจอร์
- HP ของ TF-IDF (ngram_range, min_df, max_features)
- ชื่อ HF model ของ sentiment + embedding

### `configs/train.yaml`
- HP ของ baselines (n_estimators, learning_rate, num_leaves)
- HP ของ transformers — มี shared section + per-model overrides
- HP ของ hybrid (MLP layers, dropout, focal loss gamma)

### `configs/eval.yaml`
- bootstrap n_iter
- threshold scan range
- calibration n_bins
- buckets ของ sub-population analysis

### `configs/train_phaya_*.yaml`
- HPO trials: ลอง learning rate 5e-6, 1e-5, 2e-5 + epochs 4, 6
- มี multifield variant ที่ใช้ title + description + channel

---

## 9. Cloud GPU (Hugging Face Jobs)

เลือก HF Jobs เป็น **canonical cloud path** เพราะจองได้ทันที ไม่ติด queue (Kaggle path ยังคงเก็บไว้เป็น fallback)

### ค่าใช้จ่าย

| Hardware | rate | wall-time per encoder | cost |
|---|---|---|---|
| t4-small (16 GB) | $0.40/h | ~14 นาที (Wangchan) | $0.09 |
| a10g-small (24 GB) | $1.00/h | ~12 นาที (Phaya) | $0.18 |
| a10g-small | $1.00/h | ~14 นาที (XLM-R) | $0.22 |
| **รวมทั้งโครงงาน** | | | **$0.49** |

### วิธีใช้

```bash
# 1) Login เข้า HF (one-time setup)
hf auth login

# 2) อัปข้อมูลขึ้น HF private dataset
make hf-dataset

# 3) Submit งานทั้ง 3 แบบ detached
make train-hf-all-detached

# 4) Poll สถานะของแต่ละงาน
uv run python scripts/cloud/run_on_hf_jobs.py --poll-only --job-id <ID>
```

ผลลัพธ์ของแต่ละ job ถูก save กลับมาเป็น HF private dataset แล้วดาวน์โหลดอัตโนมัติเข้า `reports/artifacts/predictions/transformers/`

---

## 10. Outputs และ Reports

### `reports/tables/` — ตาราง CSV (input ของ thesis tables)

| ไฟล์ | เนื้อหา |
|---|---|
| `all_models_metrics.csv` | metric รวมของทุกแบบจำลอง |
| `baselines_metrics.csv` | metric ของ baselines เท่านั้น |
| `mcnemar_pairwise.csv` | McNemar ทุกคู่ |
| `mcnemar_pairwise_encoders.csv` | McNemar เฉพาะ 3 encoders |
| `cochrans_q.csv` | Cochran's Q ของ 22 แบบจำลอง |
| `cochrans_q_encoders.csv` | Cochran's Q เฉพาะ 3 encoders |
| `calibration_*.csv` | reliability curve + ECE (3 วิธี × n model) |
| `per_category_breakdown.csv` | ROC-AUC ตามหมวดหมู่ YouTube |
| `stacking_by_channel_size.csv` | ROC-AUC ตามขนาดช่อง |
| `stacking_drop_one_ablation.csv` | drop-one importance |
| `sentiment_ablation.csv` | with/without sentiment ablation |
| `per_channel_auc.csv` | ROC-AUC ภายในแต่ละช่อง (48 ช่องที่เข้าเงื่อนไข) |
| `k_fold_robustness.csv` | 5-fold over channels mean ± SD |
| `errors_top_fp_*.csv`, `errors_top_fn_*.csv` | top-50 false positives/negatives |

### `reports/figures/` — กราฟ SVG/PNG

- `leaderboard_with_ci.svg` — leaderboard ของ 12 แบบจำลองต้น
- `robustness_slices.svg` — per-category + per-size
- `reliability_stacking.svg` — calibration ก่อน/หลัง Platt
- `roc_pr_test.svg` — ROC + PR ของ 6 แบบจำลองต้น
- `shap_summary.png` — SHAP summary plot
- `per_channel_auc_cdf.png` — empirical CDF ของ ROC-AUC รายช่อง
- `lime/lime_html/*.html` — local explanations
- `attention/attention_html/*.html` — attention heatmaps

### `reports/artifacts/predictions/` — Parquet predictions

```
predictions/
├── baselines/          # 9 ไฟล์ (3 algos × 3 feature sets)
├── transformers/       # wangchanberta.parquet, phayathaibert.parquet, xlm-roberta-large.parquet
├── hybrid/             # hybrid_mlp.parquet, hybrid_gbm.parquet
├── ablation/           # phayathaibert_multifield, hpo/*
└── stacking/           # stacking_lr.parquet, stacking_lr_calibrated.parquet, stacking_lr_pruned.parquet
```

ทุกไฟล์มีคอลัมน์ `(video_id, split, y_true, y_proba)` — โหลดได้ตรง ๆ ด้วย pandas

---

## 11. การทำซ้ำ (Reproducibility)

ทุก run ถูกบันทึกอย่างละเอียด:

### Seeds

```python
PYTHONHASHSEED=42
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
transformers.set_seed(42)
```

### MLflow tracking

ทุก run บันทึก:
- `params` — config ที่ flatten แล้ว
- `metric` — ทุก metric ทุก epoch + final
- `tag.git_sha` — commit hash ปัจจุบัน
- `tag.dataset_hash` — sha256 ของ `dataset_with_labels.parquet`
- `artifact` — predictions parquet + config snapshot

ดู MLflow UI ได้ที่:

```bash
uv run mlflow ui
# เปิด http://localhost:5000
```

### กฎการพัฒนา (CLAUDE.md)

โปรเจกต์มี **กฎเข้มงวด** เพื่อป้องกัน data leakage และให้การทำซ้ำได้:

1. **Channel-grouped + time-aware split** (ห้าม shuffle)
2. **Imbalance handling เฉพาะ train** (val/test รักษาธรรมชาติ ~9% positive)
3. **Configs over flags** (HP อยู่ใน YAML)
4. **One change per experiment** (ห้ามรวมหลายตัวแปรในรันเดียว)
5. **Save predictions, not just metrics** (parquet เป็น first-class artifact)
6. **Bootstrap CI 95% (n=1000) บนทุก headline metric**
7. **McNemar pairwise + Cochran's Q บนชุดทดสอบเดียวกัน**
8. **Calibration on val, evaluate on test**
9. **Honest reporting** — รายงาน negative findings (multi-field, $H_2$ rejection, null HPO)

---

## 12. License และเครดิต

ผลงานทั้งหมดของปัญหาพิเศษเล่มนี้ ทั้งวิทยานิพนธ์ รหัสฐาน ระเบียบวิธีวิจัย ผลการทดลอง และคำอธิบายต่าง ๆ **เป็นเครดิตของนิสิตทั้งสามคน**:

- **นายตรัยภูรินท์ สืบสุวรรณสาร** (รหัสนักศึกษา 65050330)
- **นายธีรนนท์ ไชยลังกา** (รหัสนักศึกษา 65050417)
- **นายภาณุเดช ภู่โทสนธิ์** (รหัสนักศึกษา 65050685)

**อาจารย์ที่ปรึกษา:** ผศ. ดร. พรพิมล ชัยวุฒิศักดิ์
ภาควิชาสถิติประยุกต์และการวิเคราะห์ข้อมูล คณะวิทยาศาสตร์
สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง · ปีการศึกษา 2568

### ขอขอบคุณ

ทีมงานเบื้องหลังเครื่องมือโอเพนซอร์สที่ใช้ในงานนี้:

- **AIResearch.in.th** — WangchanBERTa
- **CLICKNEXT** — PhayaThaiBERT
- **Meta AI** — XLM-RoBERTa
- **Hugging Face** — Transformers, Datasets, Hub, Jobs
- **PyThaiNLP** — Thai tokenization
- **scikit-learn**, **LightGBM**, **XGBoost**
- **MLflow**, **SHAP**, **LIME**

---

## ภาคผนวก: คำสั่งที่มักใช้

```bash
# ดู MLflow runs ทั้งหมด
uv run mlflow ui

# Lint + format check
make lint

# Unit tests
make test

# ลบ cache
make clean

# คอมไพล์วิทยานิพนธ์
cd docs/thesis
xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex
```

---

📖 **เอกสารวิทยานิพนธ์ฉบับเต็ม** อยู่ที่ `docs/thesis/` เขียนด้วย XeLaTeX ตามรูปแบบ KMITL พร้อมคอมไพล์ได้ทันที (ดู `docs/thesis/README.md` สำหรับขั้นตอน)
