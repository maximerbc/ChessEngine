# Policy/value research and consolidated website handoff

Implementation branch: `codex/policy-value-research`. Local implementation commit: `ded5e5a` (**Add policy value research**). Later commits record execution evidence and documentation. The user manages all remote pushes; no PR, visibility, account or hosting setting was changed here.

PRs #9 and #10 are already merged. This branch starts from their verified equivalent file tree; do not reopen those completed studies or push the deleted historical branches.

## Scope and verification

- Demo and Research navigation; Play/Compare share the current board/history. Existing model explanations and archived game replay remain accessible. Old model links resolve to the corresponding study without substituting checkpoints.
- Research index and an explicit public report allowlist, independent of the failed move-ranking checkpoint release gate. All negative/inconclusive results remain visible.
- Research-only NumPy policy/value model, legal-history-preserving PUCT, baseline/0.1 anchored training, frozen actors, uniform completed-game weighting, exact CPU minibatch resume, source/config/code locks, development-only selection, fresh paired finals and audits/reports.
- **91 Python tests passed**, lint passed, CLI help and production frontend build passed. **16 real/mocked local browser flows passed**, one optional four-model preview flow skipped in that suite. The staged four-model compatibility workflow was subsequently run separately and passed for all four routes. Local workflows do not establish Docker, remote CI or Railway deployment success.
- Actual CPU pilot/main-study status is in `docs/experiments/policy-value/` and `runs/policy-value/`. Tiny fixture runs verify integration only. Do not describe the main study as complete until all six runs, selections, planned finals and audits are present.
- Released checkpoint hashes and default are unchanged. No MCTS player enters the public API and no model is promoted automatically.

## User-managed push

```sh
git push -u origin codex/policy-value-research
```

Open the branch against `main` and wait for Python, browser and container CI. Suggested title: **Unify the demo and add policy/value research**. The implementation is reviewable independently of long-running experimental evidence; retain accurate execution status in its description. Hosting continues to use the two existing Railway services and Dockerfiles, with no paid activation or new services.

## Local reproduction

The README contains the benchmark, pilot, full-run and report commands. The full runner refuses a launch before a passing pilot has written its forecast. Communicate that measured estimate before starting the main run. Avoid concurrent study runners. Do not edit frozen scientific code/configuration to force resume through a mismatch.

For the test-browser cache used in this workspace:

```sh
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/chessengine-browsers npm --prefix web run test:e2e
```

Use the normal Playwright install in a fresh environment. For real local integration tests, start the API and a fresh frontend preview, then supply both `PLAYWRIGHT_BASE_URL` and the actual `DEPLOYMENT_API_URL`, plus `RUN_LIVE_TESTS=1`. An incorrect API interception URL does not simulate an outage.
