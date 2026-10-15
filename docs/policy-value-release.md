# Completed feasibility release — 2026-10-15

The user explicitly requested an immediately pushable finished product after the measured main-study forecast of 559.6 hours (23.3 days, including a 50% margin). This amendment closes the release at the completed feasibility pilot. It does not revise the original preregistration, lower its quality gates, or relabel partial work as a completed comparison.

## Delivered

- One consolidated Demo with shared Play/Compare position history and the existing released models.
- One Research destination containing historical results, including negative and inconclusive studies, and the completed policy/value feasibility report.
- Implemented and tested policy/value model, PUCT, baseline/anchored objectives, deterministic resume, data separation, selection/final evaluation infrastructure, audits and reporting.
- All 60 pilot attempts across 128/256/512 simulations: 20/20, 20/20 and 19/20 rule-completed games respectively. All three budgets passed the frozen feasibility gates. The independent closure audit replayed every trajectory, checked legal visits/outcomes, verified all game hashes and the common initializer, and verified unchanged scientific sources and released-model checksums.

## Closed and deferred

The long continuation was stopped gracefully. Eight saved baseline seed-42 generation records and their ledgers remain in the private local run directory. No training update, development selection or final match completed. This cannot establish whether the anchor helps, harms or matches baseline playing strength. No checkpoint was promoted; the existing demo default and weights remain unchanged. Paid cost is C$0.

The original three-seed baseline-versus-anchor comparison is optional future work requiring new explicit authorization. Its frozen scientific protocol, config and source identities remain preserved. The administrative continuation refuses to resume the closed run while its closure marker is present. No ongoing computation, future report generation or full training is required to complete this release.

## Evidence and handoff

The report is `docs/experiments/policy-value/index.html`. Its evidence and protocol are published through the existing static asset allowlist. `docs/experiments/policy-value/release-scope.json` records the user scope decision and independent pilot audit; it is separate from the full-study `final-audit.json`, which does not exist and is not claimed.

Commit the verified website and report snapshot locally, then follow `push-handoff.md`. The user performs the remote push. Remote CI, PR merge and Railway deployment are subsequent delivery steps, not claims established by local validation.
