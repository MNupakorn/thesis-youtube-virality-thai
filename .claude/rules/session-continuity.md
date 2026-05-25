# Session Continuity

## At Session Start
1. Read `MEMORY.md` (root) — single-line index of project state.
2. Read `CLAUDE.md` — rules, commands, prohibitions, model list.
3. Skim `.claude/memory/active.md` — what was being worked on.
4. Skim `.claude/memory/decisions.md` — frozen choices you must not silently revisit.
5. Do not ask the user to re-explain anything already in these files.

## During the Session
- Treat `.claude/memory/decisions.md` as binding. If a decision must change, propose the change and write a new entry — do not overwrite.
- Append to `.claude/memory/experiments.md` after every training/eval run with: timestamp, git sha, model, config, metrics, artifact path.
- Use MLflow as the source of truth for numerical results. The memory file is the index, MLflow is the data.

## At Session End (if code or data changed)
Update the following, in this order:

1. `MEMORY.md` — bump "Last updated", refresh "Latest Session" + "Next Priority"
2. `.claude/memory/active.md` — current focused task + next step
3. `.claude/memory/changelog.md` — one line per concrete change with commit SHA
4. `.claude/memory/experiments.md` — only if a model was trained or evaluated
5. `.claude/memory/decisions.md` — only if a frozen choice changed
6. `.claude/memory/architecture.md` — only if directory/module structure changed

Be factual. Do not rewrite the whole file. Append where possible.

## Recovery from Context Compaction
If the conversation was compacted and you are unsure of state:
- The truth is on disk. Run `git log -10 --oneline`, `ls reports/predictions/`, `mlflow runs list` (if MLflow CLI available).
- The memory files describe intent. The repo describes reality. When they disagree, the repo wins and memory gets corrected.
