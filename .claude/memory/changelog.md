# Changelog

Per-session diffs. One line per concrete change. Include commit SHA when pushed.

---

## 2026-05-26 — Claude architecture setup

- Add `CLAUDE.md` (project identity, models, rules, prohibitions).
- Add `MEMORY.md` (root index).
- Add `.claude/settings.json` (env: PYTHONHASHSEED=42, TOKENIZERS_PARALLELISM=false).
- Add `.claude/rules/{research-engineering,reproducibility,session-continuity}.md`.
- Add `.claude/skills/{thesis-operating-system,ml-experiment-discipline,data-pipeline,cloud-gpu-orchestration,evaluation-rigor,memory-system,thesis-writing}/SKILL.md`.
- Add `.claude/agents/{plan-orchestrator,data-engineer,model-trainer,evaluator,explainer,thesis-writer}.md`.
- Add `.claude/memory/{active,decisions,architecture,changelog,experiments}.md`.

---

## 2026-05-26 — Kaggle CLI installed

- `uv pip install kaggle` in `.venv` → Kaggle CLI 2.1.2.
- Verified auth as `mknpk01` (token `~/.kaggle/kaggle.json` chmod 600).
- `kaggle kernels list --user mknpk01` and `kaggle datasets list --user mknpk01` both succeed.
- No new code yet — pipeline scripts under construction.

---

## 2026-05-26 (earlier) — Notebook polish · commit `3cf31a9`

Pre-existing. See `git log`.

---

## How to Append

```
## YYYY-MM-DD — short title · commit <sha or "uncommitted">

- one line per concrete change
- another line
```

## 2026-05-26 · cloud-pipeline + eval/explain extensions + WangchanBERTa local fallback

- `9fa5d70` feat(explain): add LIME + attention rollout runners; fix evaluate Path concat
- `60fad4a` feat(eval): ROC/PR + reliability figures + top-50 FP/FN error analysis
- `ba9a1b5` feat(cloud): Kaggle Kernels orchestration for encoder fine-tuning
- `18c28ec` docs(claude): add CLAUDE.md, .claude/ architecture, MEMORY.md, uv.lock
- (uncommitted as of writing) auto-disable fp16/bf16 on non-CUDA in `transformer_finetune.py`; added `configs/train_m1.yaml` reduced config; ran WangchanBERTa locally on M1; encoder subset stats in `evaluate.py`.
