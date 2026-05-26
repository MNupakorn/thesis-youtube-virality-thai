.PHONY: help install data prepare train-baselines train-wangchan train-phaya train-xlmr train-hybrid train-all train-cloud-wangchan train-cloud-phaya train-cloud-xlmr train-cloud-all kaggle-dataset hf-dataset train-hf-wangchan train-hf-phaya train-hf-xlmr train-hf-all-detached eval explain clean lint test

PY := uv run python
CONFIG_DATA := configs/data.yaml
CONFIG_FEATS := configs/features.yaml
CONFIG_TRAIN := configs/train.yaml
CONFIG_EVAL := configs/eval.yaml

help:
	@echo "Targets:"
	@echo "  install              Create uv env and install deps"
	@echo "  data                 Collect raw data from YouTube API (needs YOUTUBE_API_KEY)"
	@echo "  prepare              Clean + label + feature engineer + split"
	@echo "  train-baselines      Train LR/LightGBM/XGBoost baselines"
	@echo "  train-wangchan       Fine-tune WangchanBERTa locally (slow on M1)"
	@echo "  train-phaya          Fine-tune PhayaThaiBERT locally (slow)"
	@echo "  train-xlmr           Fine-tune XLM-RoBERTa-large locally (very slow)"
	@echo "  train-hybrid         Train multi-modal hybrid model"
	@echo "  train-all            Run baselines + WangchanBERTa + hybrid locally"
	@echo "  ---  Cloud (Kaggle Kernels) ---"
	@echo "  kaggle-dataset       Build / version the private Kaggle dataset"
	@echo "  train-cloud-wangchan Push & poll Kaggle kernel for WangchanBERTa"
	@echo "  train-cloud-phaya    Push & poll Kaggle kernel for PhayaThaiBERT"
	@echo "  train-cloud-xlmr     Push & poll Kaggle kernel for XLM-RoBERTa-large"
	@echo "  train-cloud-all      All three encoders, sequential (Kaggle limits concurrency)"
	@echo "  ---  Cloud (HF Jobs — pay-as-you-go, requires PRO + 'hf auth login') ---"
	@echo "  hf-dataset           Upload data files to HF private dataset"
	@echo "  train-hf-wangchan    Submit + poll WangchanBERTa job on HF (default t4-small)"
	@echo "  train-hf-phaya       Submit + poll PhayaThaiBERT job on HF"
	@echo "  train-hf-xlmr        Submit + poll XLM-RoBERTa-large job (a10g-small)"
	@echo "  train-hf-all-detached  Submit all 3 jobs detached; poll each with --poll-only"
	@echo "  ---"
	@echo "  eval                 Run full evaluation incl. McNemar / Cochran's Q"
	@echo "  explain              Generate SHAP / LIME / attention artifacts"
	@echo "  lint                 ruff + black --check"
	@echo "  test                 pytest"
	@echo "  clean                Remove __pycache__ / .pytest_cache / .ruff_cache"

install:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

data:
	$(PY) scripts/collect_data.py --config $(CONFIG_DATA)

prepare:
	$(PY) scripts/prepare_data.py --data-config $(CONFIG_DATA) --features-config $(CONFIG_FEATS)

train-baselines:
	$(PY) scripts/train_baselines.py --config $(CONFIG_TRAIN)

train-wangchan:
	$(PY) scripts/train_transformer.py --model wangchanberta --config $(CONFIG_TRAIN)

train-phaya:
	$(PY) scripts/train_transformer.py --model phayathaibert --config $(CONFIG_TRAIN)

train-xlmr:
	$(PY) scripts/train_transformer.py --model xlm-roberta-large --config $(CONFIG_TRAIN)

train-hybrid:
	$(PY) scripts/train_hybrid.py --config $(CONFIG_TRAIN)

train-all: train-baselines train-wangchan train-hybrid

kaggle-dataset:
	$(PY) scripts/cloud/build_kaggle_dataset.py

train-cloud-wangchan:
	$(PY) scripts/cloud/run_on_kaggle.py --model wangchanberta

train-cloud-phaya:
	$(PY) scripts/cloud/run_on_kaggle.py --model phayathaibert

train-cloud-xlmr:
	$(PY) scripts/cloud/run_on_kaggle.py --model xlm-roberta-large

train-cloud-all: train-cloud-wangchan train-cloud-phaya train-cloud-xlmr

# --- HF Jobs path (preferred when Kaggle pool is congested) ---
hf-dataset:
	$(PY) -m scripts.cloud.build_hf_dataset

train-hf-wangchan:
	$(PY) -m scripts.cloud.run_on_hf_jobs --model wangchanberta

train-hf-phaya:
	$(PY) -m scripts.cloud.run_on_hf_jobs --model phayathaibert

train-hf-xlmr:
	$(PY) -m scripts.cloud.run_on_hf_jobs --model xlm-roberta-large --flavor a10g-small

train-hf-all-detached:
	$(PY) -m scripts.cloud.run_on_hf_jobs --model wangchanberta --push-only
	$(PY) -m scripts.cloud.run_on_hf_jobs --model phayathaibert --push-only
	$(PY) -m scripts.cloud.run_on_hf_jobs --model xlm-roberta-large --push-only --flavor a10g-small

eval:
	$(PY) scripts/evaluate.py --config $(CONFIG_EVAL)

explain:
	$(PY) scripts/explain.py --config $(CONFIG_EVAL)

lint:
	uv run ruff check src/ scripts/ tests/
	uv run black --check src/ scripts/ tests/

test:
	uv run pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
