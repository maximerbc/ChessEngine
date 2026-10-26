# Local evolutionary execution

The initial measured forecast is **17.2 hours for the learning pilot** and
**287.2 hours for the main study**, including a 50% margin and two workers.
These are forecasts, not caps or promises. The main forecast is recomputed after
the pilot; a failed pilot ends this study. Paid compute is C$0.

`scripts/run_evolutionary_release.py` is the durable local coordinator. It runs
pilot, conditional training/evaluation/promotion, report publication, verification
and an evidence-only local commit. It never pushes, deploys, provisions resources
or resumes the closed policy/value study. Each scientific stage independently
checks the frozen source/config/input hashes. Source changes stop execution rather
than silently changing the experiment. Do not start a second runner.

Status: `.venv/bin/python scripts/run_evolutionary_release.py --status`.
Detailed progress is in `runs/evolutionary/execution.log`, `execution.json`,
`pilot/history.json` and per-generation match ledgers. Private data/optimizer files
stay under ignored `runs/`; only allowlisted evidence and qualified weights are
published. Public Research is a snapshot, not a training service or live monitor.

To stop, send SIGINT to the coordinator PID recorded in `execution.json`.
It interrupts its worker process group, retaining completed matches and pending
optimizer state. Resume using `.venv/bin/python scripts/run_evolutionary_release.py`;
the PID lock prevents duplicate runners. No automatic scientific retry or restart
occurs on error, and no failed quality gate is relaxed. Administrative verification
may be rerun without retraining or touching final selection.

The comparison feature committed concurrently is preserved independently. It may
invoke an explicitly named Stockfish reference for user-requested analysis; it is
not part of evolutionary training and supplies no teacher labels to this study.

Original release checksums remain frozen in `artifacts/research/original-release.json`.
Supervised remains the default; RL is removed from the live model catalog and its
public weight file is deleted. Until promotion, Evolutionary is an unavailable
third Demo card with an accessible Research link. Archived self-play reports and
recorded-game replay remain under Research.

Docker is unavailable in this local environment. Production containers, remote CI,
GitHub publication and Railway deployment remain unverified by local browser/tests.
