# Policy/value execution and handoff

## Baseline and release boundary

Implementation branch: `codex/policy-value-research`. Its starting tree at `e0b25a7` was verified identical to main after PRs #9/#10 merged (`1f6318b`); GitHub comparison showed two merge commits and zero changed files. CLI authentication could not fetch those merge objects. No user-managed push or account setting was changed.

The released default remains the older local-study RL checkpoint. The later negative v3 RL, coefficient-1 replay diagnosis, move ranker, and new coefficient-0.1 policy/value study have separate evidence. New public reports are published independently of model release; existing checksummed weights remain unchanged.

## Current release scope

The user closed this release at completed feasibility on 2026-10-15 to allow an immediate push. All three pilot budgets passed (59 rule-completed games in 60 attempts). The main-run forecast was 559.6 hours including margin. The continuation was gracefully stopped; eight generated games and its ledgers are retained locally, with no completed update, selection or final match. No background study remains active. The original full protocol below is historical/planned methodology, not a pending release requirement. See `policy-value-release.md` and the versioned `release-scope.json`. Any future full run requires new authorization.

## Historical compute protocol

The disposable 128-simulation, 32-ply benchmark took 2.65 seconds and implied an upper-envelope pilot estimate of 2.47 hours (maximum 512 plies, all three budgets, 50% margin). This estimate was communicated before launch. It is not the main-study forecast. The pilot begins by auditing fresh source games/opening positions, then attempts 20 games each at 128/256/512 simulations. No paid compute is used. No artificial local runtime cap is imposed.

Pilot output/log: `runs/policy-value/pilot.log`; frozen protocol: `runs/policy-value/protocol.json`; opening lineage: `runs/policy-value/openings-manifest.json`. Scientific code hashes were sealed before pilot measurements. Reporting is maintained in `pv_reporting.py`, separate from the frozen scientific runner.

Do not launch two copies of a study stage concurrently. Before a full launch, inspect `pilot-decision.json` and communicate `forecast.json`. If the pilot rejects every budget, publish that result and do not relax the gates. If a batch later fails its minimum completed-game count, preserve the partial study and failure record rather than inventing outcomes.

## Reproducibility and validation

The prospective protocol names exactly two treatments and run seeds 42/43/44. A disposable fixture study covers all six training paths, development selection, locking, fresh final comparisons, integrity auditing and report generation. This tiny integration fixture is not evidence for playing strength.

The automated suite covers search signs and budgets, legal priors, promotions/castling/en passant, history-based and halfmove draws, initial value and NumPy export parity, anchor detachment/gradients, truncation exclusion, uniform sampling, exact CPU minibatch resume, identity mismatches and paired comparisons. Production release-model checks remain active. Local browser checks exercise both mocked failure cases and the actual released-model API. Docker/remote CI and Railway deployment are separate and are not established by these local checks.

Public research assets are an explicit filename allowlist. Private run paths and non-allowlisted links are removed; opaque artifact IDs preserve distinct hashes. No raw training directory is copied into the website. The original PDF and released model-download URLs remain valid.

## Historical background continuation (stopped)

A single local continuation previously ran for this study. It waited for the existing pilot's recorded completion rather than starting a duplicate. A rejected pilot stops the continuation. A passing pilot publishes and logs the exact forecast before the main run, then executes the unchanged frozen runner through training, selection, final evaluation and reports.

Local process/status records: `runs/policy-value/continuation-process.json` and `continuation-status.json`; output: `continuation.log`; prelaunch estimate: `launch-estimate.json`. The main elapsed-time/RAM ledger is checkpointed every 30 seconds in `continuation-compute.json`. Its timing overlaps the completed-run compute ledger and must not be added to it. Continuation checks the recorded CPU package versions and project dependency identity before resume. A lock prevents duplicate continuation processes.

This was one local computation, without a scheduled automation or hosted service. It used no paid resource, online training, model promotion or remote push. The final interruption record and ledgers are preserved. The completed feasibility report explicitly separates pilot evidence from the deferred full study. Do not restart the closed study without new authorization.

The two main verified implementation milestones are 91 passing Python tests and 16 passing local browser workflows (one optional staged four-model preview workflow skipped). Frozen scientific source identities were rechecked after the implementation commit; administrative continuation and reporting do not alter the search/training protocol.

The interactive pilot was handed over after 34 completed games. `pilot-handoff.json` preserves every completed-game hash and `pilot-handoff-compute.json` preserves the original process ledger. All 34 records were verified unchanged after the continuation resumed, and later games advanced. Remaining pilot output now goes to `continuation.log`; `pilot.log` is the earlier process's historical log. Reported peak memory retains the maximum across the handoff ledger and the current run.

The separately staged real four-model preview compatibility workflow also passed for all four historical routes. It remains behind the explicit preview flag and does not change the consolidated demo or release decision.
