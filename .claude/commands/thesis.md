# /thesis — Smart Router for the Thai Virality Thesis

> Use `/thesis <request>` (or `/t`) to route any task to the right agent.

## Routing Logic

| User phrase | Route to |
|---|---|
| "add a feature", "regenerate labels", "fix leakage", "resplit" | `data-engineer` |
| "train", "fine-tune", "retrain", "tune HPs", "run on Kaggle" | `model-trainer` |
| "compare models", "is X significant?", "calibrate", "bootstrap", "results table" | `evaluator` |
| "why did the model predict X?", "feature importance", "attention map" | `explainer` |
| "draft chapter", "polish abstract", "write methods section" | `thesis-writer` |
| Anything multi-phase ("train all three and evaluate") | `plan-orchestrator` |
| Ambiguous / scope-changing | `plan-orchestrator` (analyze + confirm decision impact) |

## Before Routing

Read `MEMORY.md` + `.claude/memory/active.md` + `.claude/memory/decisions.md`. If the request would change a frozen decision (label, splits, model set), refuse and ask the user to either accept a new dated decision entry or rephrase.

## Announce

```
[🔬 thesis] Request: "{user_request}"
Routing → {agent}
```

## Examples

| Request | Route |
|---|---|
| `/thesis add an emoji-count feature` | data-engineer |
| `/thesis run WangchanBERTa on Kaggle` | model-trainer |
| `/thesis is the WCB vs PTB gap significant?` | evaluator |
| `/thesis show me top false positives` | evaluator |
| `/thesis explain why hybrid overfits` | explainer (+ evaluator for the diagnostic) |
| `/thesis run all three encoders and produce final tables` | plan-orchestrator |
| `/thesis draft the methods chapter` | thesis-writer |
| `/thesis let's switch to a decoder LLM` | **refuse** — point to `.claude/memory/decisions.md` |

## Quality Gate Before Closing Out

- `uv run pytest` passes
- MLflow run logged (if training/eval ran)
- Predictions parquet exists at `reports/predictions/{model}_{split}.parquet`
- Memory files updated per `.claude/rules/session-continuity.md`
