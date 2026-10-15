# Completed research and preview handoff

All 960 main evaluation games and 160 fixed-depth searches are complete. Final guided score: 48.75% (95% interval 45.75–51.625%), inconclusive. New supervised release-gate score: 46.25%, below the required 50%. Preserve the deployed checkpoints. The gallery is a local preview; default production builds retain the existing interface.

## Push both branches

```sh
git push -u origin codex/move-ranking codex/showcase-four-models
```

1. Open `codex/move-ranking` → `main`, titled **Train and evaluate root move ranking**. This also includes the previously unpushed RL update diagnosis. Wait for CI, then merge with **Create a merge commit** to preserve the shared branch history.
2. Open `codex/showcase-four-models` → `main`, titled **Add an identity-checked four-model preview**. Merge only after the research PR and passing CI. If opening both PRs at once, initially target the showcase PR at `codex/move-ranking`, then retarget it to `main` after the first merge.

Descriptions are sections 11 and 12 of `docs/pr-descriptions.md`. No separate diagnosis push is needed. No Railway account or billing change is required. Existing automatic deployment can rebuild the unchanged public model collection, but these PRs do not activate the experimental gallery or replace its models.

## Evidence and verification

- `docs/experiments/move-ranking/index.html`: report with learning curves, classification, development/final comparisons and limitations.
- `docs/experiments/move-ranking/final-audit.json`: independent score totals for all 960 games.
- Local Python, frontend build, lint, browser-flow and real-checkpoint preview checks passed. Docker and GitHub CI results are pending remote execution, not claimed as locally verified.
- `docs/four-model-showcase.md`: local preview commands and release boundary.

Future improvements require a separately planned experiment. These results establish modest teacher-move imitation and some shallow search savings, not stronger chess.
