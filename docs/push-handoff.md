# Policy/value research and consolidated website handoff

> Historical handoff: branch, lineup, and authorization details below apply to its original date. See the [current overview](../README.md) and [operations archive](archive/README.md).

Implementation branch: `codex/policy-value-research`. Local implementation commit: `ded5e5a` (**Add policy value research**). Later commits record execution evidence and documentation. The user manages all remote pushes; no PR, visibility, account or hosting setting was changed here.

PRs #9 and #10 are already merged. This branch starts from their verified equivalent file tree; do not reopen those completed studies or push the deleted historical branches.

## Three-opponent Demo update — 2026-10-25

The public Demo now presents **Classical**, **Supervised**, and **Self-play** in that order. Each card explains its evaluator; the selected opponent explains evaluation, shared alpha-beta search and evidence. The existing released self-play checkpoint carries its original inconclusive evidence, not the later v3 regression result. Evidence links preserve the board/history and selected opponent. Mathematics and checkpoint details are expandable.

All three cards remain visible during loading/outages. Missing requested opponents stay unavailable without fallback. Opponent selection stays locked during a game, with New game guidance; comparison accepts any two available released opponents and uses identical position history, node and time limits. Existing four-model local previews and legacy research redirects remain supported.

Validation: **93 Python tests**, lint, CLI help and frontend build passed. **24 local browser flows passed**, including six opponent/color combinations, keyboard/mobile states, release-identity evidence, unavailable models, history-preserving research links, comparison and a real released-backend flow. The **four-model preview browser flow passed separately** against its existing local artifacts. Two production-container flows were skipped; remote CI, Docker and Railway deployment are not verified by this update. The first browser attempts lacked the browser executable and then hit macOS sandbox launch restrictions; after installing temporary test browsers and allowing the local Chromium launch, the applicable suite passed.

Both released model checksums were verified, and the release manifest is unchanged. No training, checkpoint promotion, remote push or deployment was performed. Push this branch using the command below; the focused local update commit is identified in the chat handoff.

## Scope and verification

- Demo and Research navigation; Play/Compare share the current board/history. Existing model explanations and archived game replay remain accessible. Old model links resolve to the corresponding study without substituting checkpoints.
- Research index and an explicit public report allowlist, independent of the failed move-ranking checkpoint release gate. All negative/inconclusive results remain visible.
- Research-only NumPy policy/value model, legal-history-preserving PUCT, baseline/0.1 anchored training, frozen actors, uniform completed-game weighting, exact CPU minibatch resume, source/config/code locks, development-only selection, fresh paired finals and audits/reports.
- **93 Python tests passed**, lint passed, CLI help and production frontend build passed after the scope change. **16 real/mocked local browser flows passed**, including the completed feasibility label, deferred comparison and public scope audit; one optional four-model preview flow was skipped. The staged four-model compatibility workflow previously passed separately for all four routes. Local workflows do not establish Docker, remote CI or Railway deployment success.
- Actual CPU pilot/main-study status is in `docs/experiments/policy-value/` and `runs/policy-value/`. Tiny fixture runs verify integration only. Do not describe the main study as complete until all six runs, selections, planned finals and audits are present.
- Released checkpoint hashes and default are unchanged. No MCTS player enters the public API and no model is promoted automatically.

## User-managed push

**Ready for the reduced feasibility release.** The user explicitly replaced the full-study requirement with an immediately pushable product on 2026-10-15. Scope and audit: `docs/policy-value-release.md` and `docs/experiments/policy-value/release-scope.json`.

All three feasibility budgets passed: 60 attempts, 59 rule-completed games, one truncated game. The 559.6-hour main comparison was stopped after eight generated games, with no training update, selection or final match. Its data and original protocol are preserved; it is deferred future work rather than an outstanding release task. No background computation remains running and no strength improvement is claimed.

The website, existing playable collection, historical reports, research implementation and completed feasibility report form the finished release. Push the committed branch:

```sh
git push -u origin codex/policy-value-research
```

Then open the branch against `main` and wait for Python, browser and container CI. Suggested title: **Unify the demo and add policy/value research**. Hosting continues to use the two existing Railway services and Dockerfiles, with no paid activation or new services.

## Local reproduction

The README contains the benchmark, pilot, full-run and report commands. The full runner refuses a launch before a passing pilot has written its forecast. Communicate that measured estimate before starting the main run. Avoid concurrent study runners. Do not edit frozen scientific code/configuration to force resume through a mismatch.

For the test-browser cache used in this workspace:

```sh
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/chessengine-browsers npm --prefix web run test:e2e
```

Use the normal Playwright install in a fresh environment. For real local integration tests, start the API and a fresh frontend preview, then supply both `PLAYWRIGHT_BASE_URL` and the actual `DEPLOYMENT_API_URL`, plus `RUN_LIVE_TESTS=1`. An incorrect API interception URL does not simulate an outage.
