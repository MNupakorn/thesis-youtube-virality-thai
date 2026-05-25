.PHONY: help install data prepare train-baselines train-wangchan train-typhoon train-openthai train-hybrid train-all eval explain clean lint test

PY := uv run python
CONFIG_DATA := configs/data.yaml
CONFIG_FEATS := configs/features.yaml
CONFIG_TRAIN := configs/train.yaml
CONFIG_EVAL := configs/eval.yaml

help:
	@echo "Targets:"
	@echo "  install         Create uv env and install deps"
	@echo "  data            Collect raw data from YouTube API (needs YOUTUBE_API_KEY)"
	@echo "  prepare         Clean + label + feature engineer + split"
	@echo "  train-baselines Train LR/LightGBM/XGBoost baselines"
	@echo "  train-wangchan  Fine-tune WangchanBERTa"
	@echo "  train-typhoon   QLoRA fine-tune Typhoon 2.5  (cloud GPU)"
	@echo "  train-openthai  QLoRA fine-tune OpenThaiGPT  (cloud GPU)"
	@echo "  train-hybrid    Train multi-modal hybrid model"
	@echo "  train-all       Run baselines + WangchanBERTa + hybrid locally"
	@echo "  eval            Run full evaluation incl. McNemar / Cochran's Q"
	@echo "  explain         Generate SHAP / LIME / attention artifacts"
	@echo "  lint            ruff + black --check"
	@echo "  test            pytest"
	@echo "  clean           Remove __pycache__ / .pytest_cache / .ruff_cache"

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

train-typhoon:
	$(PY) scripts/train_transformer.py --model typhoon-2.5 --config $(CONFIG_TRAIN)

train-openthai:
	$(PY) scripts/train_transformer.py --model openthaigpt --config $(CONFIG_TRAIN)

train-hybrid:
	$(PY) scripts/train_hybrid.py --config $(CONFIG_TRAIN)

train-all: train-baselines train-wangchan train-hybrid

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
