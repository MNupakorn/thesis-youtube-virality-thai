# Active Task

**As of:** 2026-05-26

## Current Focus
Setting up the `.claude/` architecture (CLAUDE.md + rules + skills + agents + memory) for the thesis project. Mirrors the jinkintech-erp SKILL.md pattern but adapted to ML research.

## Next Concrete Step
1. Finish `scripts/cloud/` Kaggle pipeline:
   - `build_kaggle_dataset.py` — package data/processed + data/interim → private Kaggle dataset `mknpk01/thesis-virality-data`
   - `run_on_kaggle.py` — render kernel template, push, poll, download
   - `kaggle_kernel_template.py` — clone repo → `uv pip install` → run `scripts/train_transformer.py --model X`
   - Makefile targets: `train-cloud-wangchan`, `train-cloud-phaya`, `train-cloud-xlmr`, `train-cloud-all`
2. Push to GitHub.
3. Trigger first run (WangchanBERTa) as a smoke test.

## Blocking / Risks
- None. Kaggle credentials verified (`mknpk01`, token chmod 600). `kaggle` CLI installed in `.venv`.

## Resume Hint
If the session is compacted: re-read `MEMORY.md`, then `.claude/skills/cloud-gpu-orchestration/SKILL.md`, then `git status` to see how much of `scripts/cloud/` exists.
