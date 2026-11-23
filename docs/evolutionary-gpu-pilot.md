# GPU-first evolutionary pilot

The user authorized implementation and one Runpod pilot on 2026-11-22, within
C$5 including setup, compute, storage and conversion, with C$1 contingency.
This is a new `evolutionary-gpu-native-v1` study. Historical 324-parameter,
two-worker, alpha-beta and `gpu-value-v1` protocols are not resumed or rewritten.

## Architecture and scientific boundary

JAX/PGX/mctx run batched root-coverage uniform-prior PUCT, chess transitions, features and
853/128/64/1 value evaluation on the GPU. Separable CMA-ES evolves all weights
from an independent random initializer, without teacher labels or gradients.
The fixed seed-42 augmented checkpoint is an opponent, never an initializer.
Three groups contribute 50% handcrafted-value PUCT, 25% augmented-value PUCT and
25% own-initializer/earlier-mean PUCT fitness. Each candidate plays eight opening
pairs per group: 48 games, with common openings and both colors.

The eventual CPU evolutionary engine uses the same PUCT algorithm, legal-action
order, uniform priors, exploration constant, value perspective and automatic
draw convention. Hardware and fixed-work versus serving-time budgets are
reported separately. The currently released Classical, 80k Supervised and
experimental Augmented Supervised models and manifests remain unchanged.

This calibration pilot uses 8 candidates, 2 generations and 768 target games.
It does not replace the historical 30,720-game quality pilot or demonstrate
sufficient learning. Main workloads of 184,320/368,640/737,280 games are forecast
scenarios, not authorized runs. No qualification, merge or deployment is implied.

## GPU execution and recovery

`native_games` keeps a generation's network bank resident, advances fixed-shaped
chunks through `lax.scan`, masks finished slots and refills them at chunk
boundaries. There are no Python callbacks inside the rollout. Host replay and
durable state transfers occur between chunks, not every move. Checkpoint cost is
included in tournament measurements. Concurrency is tuned on actual hardware;
the historical two-worker restriction does not apply to this separate study.

The atomic continuation owns active GPU states, full original opening histories,
generated action histories, completed records, counters and pending task IDs.
The JSON ledger is a derived view repaired after an interrupted write. Every
completed game is independently replayed with python-chess, including checks
for automatic draws at intermediate positions. Illegal moves, inconsistent
terminal values, overflow, changed inputs or duplicate records reject fitness.
Capped games keep `*`, with neutral tournament adjudication stated separately.

The new `puct-rootcover-v1` first visits each legal root move in canonical order
once, charging every visit to the fixed simulation budget, then uses ordinary
PUCT. The final action maximizes visits, breaking visit ties by mean value and
then canonical action order so a shallow equal-visit root does not ignore the
network. The CPU counterpart implements exactly the same rule; historical callers
retain their default search unchanged. A preliminary CPU probe found ordinary
PUCT failed coverage at every 32/64/128/256 budget (augmented evaluator passed
only 4/12 roots at 256). This negative development result is preserved in
`docs/preservation/evolutionary-gpu-native-preliminary-failed-coverage.json`.
The search design was amended before any paid execution or evolutionary learning;
coverage thresholds, opponent composition and game counts were not relaxed.

The new search packs legal edges in canonical PGX order, using 256 edge slots
instead of 4,672 action labels at every tree node. This is storage compaction,
not move pruning: all legal moves must fit, otherwise an overflow anywhere in
the tree rejects the game. Dense-search visit/action/value parity is tested.

## Staged commands

Use Python 3.12 and the hash-locked CPU environment for local verification and
`training/requirements-cuda.lock` on NVIDIA. Training dependencies stay out of
production. Local new-run tests generally take under five minutes; preparation
scans historical position keys and may take a few minutes, with progress output.

```sh
PYTHONPATH=src /path/to/python312 -m pytest tests/gpu -q
PYTHONPATH=src /path/to/python312 scripts/prepare_gpu_evolution_pilot.py \
  --historical-root /path/to/preserved/ChessEngine \
  --output runs/evolutionary-gpu-native-v1/inputs/openings.json
```

Preparation generates seeded legal-random histories and excludes all supplied
supervised positions and reserved opening positions. It records source hashes,
separate coverage/benchmark/pilot pools and no target values. Synthetic openings
are explicitly throughput calibration, not final-strength evidence or an
assurance of future main-study game lengths. Final pools are read only as
exclusions and are never played, scored or used for model selection.

On the private Pod, after staging and checksum verification:

```sh
python3.12 -m venv /workspace/venv
/workspace/venv/bin/python -m pip install --require-hashes -r training/requirements-cuda.lock
PYTHONPATH=src /workspace/venv/bin/python -m chessengine.gpu.native_study \
  --openings runs/evolutionary-gpu-native-v1/inputs/openings.json \
  --output runs/evolutionary-gpu-native-v1/pilot \
  --deadline-unix "$PILOT_WORK_DEADLINE"
```

The native-study command requires actual NVIDIA execution. It verifies rules,
inference/search parity, 64-member optimizer overhead, coverage at 32/64/128/256
simulations, and interruption recovery. Select the smallest simulation budget
passing at least 80% legal-root coverage on at least 90% of frozen positions for
each evaluator class. Tune batch sizes 16/32/64 and chunks 1/8 with identical
roots, models and simulation counts. Complete-game benchmark coverage must also
pass before the 768-game pilot starts. Timing comparisons do not select weights.

## Cost controls and evidence

Use one Secure Cloud RTX 4090, initially quoted US$0.89/hour, plus 20 GB standard
network storage, quoted US$1.40/month prorated. Refresh price/stock before create.
Use the official CUDA 12.8 image, locked JAX environment and no public services.
Override template storage to avoid an additional host-local disk.

Record actual creation time and price. With a conservative all-in conversion
allowance of C$2/US$1, cap creation-to-stop at two hours: US$1.78 compute plus
small prorated storage/container charges. This fits the C$4 planned allowance;
the extra C$1 remains contingency. Verify this against the actual creation
receipt and adjust the deadline downward if any quoted charge changes.

Before installing packages, arm the independent Pod-local guard using its
injected stop credential. Verify the guard's armed file and process; otherwise
stop the Pod. The training deadline precedes hard shutdown by five minutes.
Detach execution so it survives a laptop disconnect. Stop immediately when
work completes or fails. The external operator verifies the stopped state.
No unimplemented provider timeout flag is used.

Download and verify results before deleting storage. Record billing separately
from conservative projections. Forecasts include observed compilation, running
and unfinished-game time and checkpoint overhead; missing complete games yield
no fabricated games-per-second figure. Evaluation/storage costs without direct
measurements are marked as additional estimates, not a complete spend promise.
Publish failure, partial progress and uncertainty. Preserve the failed historical
PUCT screen; GPU throughput does not establish playing strength.
