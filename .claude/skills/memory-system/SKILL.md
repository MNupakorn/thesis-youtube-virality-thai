---
name: memory-system
description: >
  Use at session start and end. Defines what to read first, what to write last, and where.
  Read on start: MEMORY.md, CLAUDE.md, .claude/memory/active.md, .claude/memory/decisions.md.
  Write on end: changelog.md + active.md, plus experiments.md if a model ran.
user-invocable: false
---

# Memory System

Zero-config session continuity. The repo has six memory files; you read four on start and write 2–4 on end.

## At Session Start (~3,000 tokens total)

Read in this order:

1. **`MEMORY.md`** (root) — single-line index, current state, next priority.
2. **`CLAUDE.md`** (root) — rules, commands, prohibitions, model list.
3. **`.claude/memory/active.md`** — current focused task + next concrete step.
4. **`.claude/memory/decisions.md`** — frozen choices that must not be silently revisited.

Skim `architecture.md`, `experiments.md`, `changelog.md` only when relevant to the task.

## At Session End (if anything changed)

Update in this order:

1. **`MEMORY.md`** — bump "Last updated", refresh "Latest Session" + "Next Priority".
2. **`.claude/memory/active.md`** — current focused task + next step (one paragraph).
3. **`.claude/memory/changelog.md`** — one line per concrete change with commit SHA if pushed.
4. **`.claude/memory/experiments.md`** — only if a model trained / evaluated.
5. **`.claude/memory/decisions.md`** — only if a frozen choice changed (append new dated entry, never edit old ones).
6. **`.claude/memory/architecture.md`** — only if module structure changed.

Then confirm to user: `Memory updated.`

## Memory File Roles

| File | Purpose | Update cadence |
|---|---|---|
| `MEMORY.md` | Index | Every session |
| `active.md` | What I'm currently working on | Every session |
| `decisions.md` | Frozen choices (append-only, dated) | Rarely |
| `experiments.md` | Run log (append-only) | Per training/eval run |
| `architecture.md` | Directory + module map | When structure changes |
| `changelog.md` | Per-session diff log | Every session |

## Rules

- Be factual. Do not rewrite the whole file. Append where possible.
- Never delete a `decisions.md` entry. Supersede with a new dated one and link back.
- If MLflow disagrees with `experiments.md`, MLflow wins; correct the memory.
- Keep `MEMORY.md` under 100 lines. Detail goes in the topic files.
- Use commit SHAs in `changelog.md` when work is pushed.
