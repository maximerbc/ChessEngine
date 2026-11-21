# GPU study operations

## Current boundary

Implementation and local CPU verification are authorized. **No provider or new training budget has been chosen. No paid instance, NVIDIA benchmark or full learning experiment has been launched.** First choose a small benchmark budget/provider with the user; then generate the measured full-workload forecast and obtain the separate learning-budget decision. Merging and deployment are separate user decisions.

Read [study design](gpu-study.md), [training environment](../training/README.md) and the current amendment in `AGENTS.md`. Historical operations documents describe their original experiments. Never point these commands at their output directories.

All examples run from this branch's repository root. Explicit `PYTHONPATH=src` ensures that a shared development environment imports this checkout. Keep the private repository private. Transfer data/archives only through a user-approved private destination; never place raw examples, optimizers or full backups in `web/public`.

## Preserve and restore

The initial backup is in the **primary checkout**, `runs/preservation-20261121/legacy-results.tar.gz`, with inventory committed as `docs/preservation/legacy-20261121.json`. It contains 1,578 files, 1,033,857,195 logical bytes, and has passed a complete independent restoration. Archive SHA-256: `085913e019ba4ccdbb15ee922c10861cae132f9a849e8994fb9676f0f6e7d7ef`. This is a local backup, not off-machine disaster recovery.

```sh
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu verify-backup \
  --archive /path/to/primary/ChessEngine/runs/preservation-20261121/legacy-results.tar.gz \
  --inventory docs/preservation/legacy-20261121.json
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu restore-backup \
  --archive /path/to/primary/ChessEngine/runs/preservation-20261121/legacy-results.tar.gz \
  --inventory docs/preservation/legacy-20261121.json --destination /path/to/unused-restoration
```

For a future fresh snapshot, use `backup --root /path/to/primary/ChessEngine --extra-root pr15=/path/to/pr15/ChessEngine --output output/backups/UNIQUE-NAME`. It includes ignored research directories, configurations/reports and local Git changes, then verifies every member. Use a fresh destination, pause writers first, and keep cloud-backed inputs downloaded. In a worktree with symlinked historical directories, take the snapshot from the primary checkout so nested symlink directories are not missed. Existing archives may be included, so provision disk space accordingly. No prior backup is overwritten.

## Prepare the frozen sources and pools

Download public source archives and record checksums. Populate the historical inputs listed in `configs/gpu-study.json` from the verified backup, preserving relative paths. The preparation step pins every historical research JSON/JSONL input, not just the released corpus. It excludes old examples, reserved games/positions and complete recorded trajectories from new held-out pools. Estimate local preparation/cache time before running the million-position job; it is not a tiny smoke test.

```sh
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu download
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu prepare-data
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu openings
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu pin-teacher \
  --binary tools/stockfish/PINNED-BINARY \
  --eval-file tools/stockfish/PINNED-BIG.nnue \
  --eval-file-small tools/stockfish/PINNED-SMALL.nnue
```

Preparation fails if exact quotas cannot be reached from the frozen sources. Do not reuse old labels, loosen exclusions or silently add archives. The teacher command requires both explicit NNUE files, and labels pin their hashes alongside the executable identity.

## Authorize and measure a benchmark

`configs/gpu-execution-receipt.example.json` is an **inactive template**. After the user's explicit decision, save a private receipt outside Git with stage `benchmark`, chosen provider, approved positive C$ budget, `authorized_by_user: true` and the hash returned by `chessengine.gpu.protocol.identity(config)`. Receipt values record a human decision; editing a file is not permission to buy compute. These commands provision nothing.

Install the CUDA lock on the authorized instance, then run `doctor --require-nvidia` and `pytest -q tests/gpu` before any learning. Start a detached job on the remote machine using its service manager or `tmux`; checkpoint files and remote storage let the local computer disconnect. Spot interruption support does not guarantee capacity or protect an ephemeral disk after instance deletion. Arrange private durable checkpoint copies within the approved storage budget.

```sh
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu benchmark \
  --approval /private/benchmark-receipt.json --batch-size 16
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu benchmark-components \
  --approval /private/benchmark-receipt.json \
  --tournament runs/gpu-value-v1/benchmark/benchmark.json \
  --binary tools/stockfish/PINNED-BINARY --workers 8
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu forecast \
  --measurement runs/gpu-value-v1/components/measurement.json \
  --hourly-rate PROVIDER_QUOTE --storage-gb ESTIMATED_GB \
  --storage-monthly-rate PROVIDER_STORAGE_QUOTE
```

Benchmark complete tournaments at 32/64/128 simulations: four randomly initialized candidates, eight color-swapped pairs per opponent group, including host rule audits and checkpoint I/O. Before supervised fitting, the benchmark uses an explicitly named untrained supervised proxy; remeasure with `--supervised runs/gpu-value-v1/supervised/seed-42/best.npz` when available. It records compilation/first search, steady game batches, memory statistics, CPU serving, actual interruption recovery and coverage. A `--smoke` run uses two-ply capped fixtures, needs no paid authorization, and cannot resolve a workload or support a GPU forecast. Batch size may change in a new benchmark directory; game counts and coverage rules do not change. Completed measurements are reused as measurements, not retimed from cached games. Preserve interrupted timing trials and use a fresh directory for their timings.

The component benchmark labels only 1,024 **training** positions, measures the exact network/optimizer's batch work with synthetic arrays, and plays complete CPU matches against both old references. Its epoch and parallel-CPU throughput estimates are labeled estimates; include feature-cache preparation, first-epoch disk I/O, scheduling, transfers, taxes, provider interruption overhead and storage. The forecast charges all stages conservatively at the supplied hourly quote and adds 20% contingency. It includes both models' qualification/final matches, development selection, the separate 30,720-game pilot and all three main-game scenarios. Random-network game lengths remain provisional; update the forecast with the pilot before main evolution.

No numeric NVIDIA time or price is claimed before measurement. `docs/experiments/gpu-value/forecast.json` records this missing-measurement boundary.

## Freeze the workload, then execute approved learning

After reviewing the measured forecast, the user chooses 24/48/96 games per candidate (48 planned). Freeze the resolved configuration **before learning**. Its hash is different from the benchmark configuration; a separate private receipt with stage `learning` must identify this resolved config, approved provider/budget, `forecast_path` and reviewed `forecast_sha256`. The pilot remains 48 games per candidate and seed 20261121 only in every main scenario, using its separate training pool. No automatic downgrade or expansion occurs.

```sh
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu resolve-workload \
  --benchmark runs/gpu-value-v1/benchmark/benchmark.json --games-per-candidate 48
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu --config runs/gpu-value-v1/resolved.json label \
  --approval /private/learning-receipt.json --binary tools/stockfish/PINNED-BINARY --workers 8
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu --config runs/gpu-value-v1/resolved.json features
```

For each seed 42/43/44, run `train-supervised --seed SEED --approval /private/learning-receipt.json` with `--config runs/gpu-value-v1/resolved.json` **before** the subcommand. Each fit writes `runs/gpu-value-v1/supervised/seed-SEED`. Then use `select-supervised --directory runs/gpu-value-v1/supervised --output runs/gpu-value-v1/supervised/selections.json`. All selections must be sealed before `final-position-test --selections ... --output ...`; final test labels never select an epoch.

With that resolved config and learning receipt, run `pilot --seed 20261121`. Run `pilot-check --candidate runs/gpu-value-v1/pilot/initial.npz --output runs/gpu-value-v1/pilot/initial-matches.json --approval ... --workers 8` and the corresponding check of `mean-010.npz`. Both use the same 40 pilot pairs against handcrafted-value PUCT. Finish with `pilot-decision --initial-matches ... --last-matches ... --coverage runs/gpu-value-v1/benchmark/benchmark.json`. Failed decisions are preserved and close the study.

After a passing pilot and updated forecast/budget review, run `evolve --seed SEED --approval ... --batch-size 32` for seeds 42/43/44. The frozen augmented-supervised seed-42 checkpoint is the opponent; no supervised parameters initialize evolution. Each seed writes its own optimizer, pending population, candidate/mean exports and resumable game batches. Run the identical command to resume. Changed files, packages, sources or scientific settings are rejected. CPU labeling and evaluation workers are configurable without a two-worker cap; benchmark actual hardware concurrency. Each labeling worker uses one Stockfish thread and 32 MiB hash. GPU batch size and CPU worker count affect memory and throughput, not sample totals.

Continuation checkpoints use a trusted private pickle envelope and checksum. Never load checkpoints from an untrusted party. Accelerator results may differ across devices; CPU fixed-budget search and continuation are tested. Authorized invocations also write start/end/provider/budget records under `runs/gpu-value-v1/executions`; `hourly_rate_cad` in a receipt permits a clearly labeled compute-cost estimate. Actual invoice cost remains unknown until recorded from the provider. Keep an external cloud billing/elapsed-time ledger, including interrupted work and storage charges; do not infer total spending solely from active-game timers.

## Select, assess, report and stage

`select-evolution --directory runs/gpu-value-v1/evolution --output runs/gpu-value-v1/evolution/selections.json --approval ... --workers 8` evaluates the frozen generation set `[10,20,40]` on 40 development pairs. Earliest wins a tie. Seal all three choices together before any qualification/final evidence. For **each** new approach:

```sh
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu --config runs/gpu-value-v1/resolved.json evaluate \
  --selections PATH/TO/selections.json --stage qualification \
  --output runs/gpu-value-v1/APPROACH-evaluation --approval /private/learning-receipt.json --workers 8
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu --config runs/gpu-value-v1/resolved.json evaluate \
  --selections PATH/TO/selections.json --stage final \
  --output runs/gpu-value-v1/APPROACH-evaluation --approval /private/learning-receipt.json --workers 8
```

Final evaluation is blocked after failed qualification. Failed or inconclusive evidence is reported without promotion. `report --evidence PATH --output docs/experiments/gpu-value` updates the separate report, and `PYTHONPATH=src .venv/bin/python -m chessengine.research_catalog` publishes its allowlisted assets while retaining historical reports. Report publication is independent of release qualification; strip private paths and never copy raw data into public assets.

`stage-release --candidates PRIVATE-SPEC.json` recomputes qualification/final gates from replayable ledgers and copies only qualified seed-42 NumPy models into `artifacts/staging/gpu-value-v1`, alongside an explicit v4 manifest preserving the 80k default. The candidate spec is an array of objects with `id`, `name`, `selections`, `qualification`, `final`, `simulations` and `evidence_id`. Do not point staging at active model directories. No current release file is changed automatically; review the staged artifacts in a later PR before any release/deployment decision.

## Verification

Run the existing Python suite, CLI help, website build and browser suite, plus the separate Python 3.12 `tests/gpu` checks. The latter cover rule/history/action parity, NumPy inference, deterministic PUCT, optimizer continuation, interrupted games and Sep-CMA-ES state. Local browser mocks test qualified identities; they are not trained artifacts or playing-strength evidence. Container/live-service checks require their explicit environment; NVIDIA checks remain gated by the benchmark decision.
