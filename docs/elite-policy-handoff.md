# Completed elite human-move baseline handoff

> Historical handoff: branch, lineup, and authorization details below apply to its original date. See the [current overview](../README.md) and [operations archive](archive/README.md).

Branch: `codex/elite-pytorch-baseline`, based on augmented alpha-beta integration
`b33405895c78e94c873687d5155dfdf59d2f3e5b`. The complete baseline and four-model
demo are staged locally. **No push, merge or deployment has occurred.**

## Completed study

The independently implemented reference CNN used fresh elite human played moves,
not Stockfish training labels. All three seeds completed 100 FP32 epochs with the
frozen Adam/batch/clipping recipe. No old weights were imported. Validation-only
selection was sealed before test evaluation; the fixed representative stays seed 42.

| Seed | Selected epoch | Test CE | Human top one | Human top three | Mean target rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 8 | 2.96556 | 25.695% | 47.607% | 7.11911 |
| 43 | 8 | 2.95742 | 26.018% | 47.456% | 7.11962 |
| 44 | 7 | 2.94613 | 25.899% | 47.533% | 7.13483 |

Seed 42 test top-one 95% whole-game interval: 25.318%–26.042%. The test uniform
legal baseline is 6.201%. Its validation paired advantage interval is
+19.570%–20.359 percentage points, passing the preregistered learning gate. The
sample standard deviation of test top-one agreement across seeds is 0.163 percentage
points; this is separate from the within-seed game-bootstrap uncertainty.
All three exports passed strict logits/probabilities/chosen-move parity, both on
the Runpod CPU and locally. No numerical training failure occurred.

All curves show substantial overfitting after the early minimum. The full 100-epoch
attempts are retained; the final tests use the validation-selected checkpoints.
Frozen-checkpoint training CE is 2.23975/2.26146/2.31167, separately recorded from
online epoch loss. Human-move agreement does not establish playing strength or Elo.
Full results and actual curves: `docs/experiments/elite-policy/index.html` and
`evidence.json`.

## Data and execution identity

- Exactly 800,000 train / 100,000 validation / 100,000 test examples. Whole-game
  splits, global position deduplication and previous-corpus/reserved-pool exclusions
  were frozen before training. All million raw/encoded labels, position keys and
  owners were audited; full-history replay/encoding/draw checks sampled 1004 positions.
- November 2025 input ZIP SHA-256:
  `1ffa577aa0a1352fa04c9a890a6706049a3a79edc52810b136758d1e2a1a2eab`.
- Dataset manifest SHA-256:
  `093938622f8d8903020ee4afc25c23d787bd5f47607bc479f3f2eeec4d558d49`.
- Frozen training source: `a7a142e0a5313a5538749a42a5c6d45aa4843991`.
  Later report/recovery/audit edits did not alter the model, data, configuration or
  training source. The original runnable package remains backed up byte-for-byte.
- Selected seed-42 SHA-256:
  `5e6deee52407776949178f846b97ed1b02d5af964328beb632b7361b6d4b7687`.
- Selections SHA-256:
  `0cf8201795baa918cf808aa7bbb218152af58db69efd9ccbe3191a5b3bdd3cbd`.
- Secure Cloud RTX PRO 4500 Blackwell, US$0.72/hour, EU-RO-1. Two RTX 4090
  placements were refused for capacity, and its queued request was verified cancelled.
- Pinned image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, digest
  `sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
  Actual Torch 2.8.0+cu128, NumPy 2.2.6, chess 1.11.2, Python 3.12.3, FP32.
- Discarded benchmark: 800,000 examples in 24.4773 seconds. Raw training forecast
  7,343 seconds; conservative launch forecast 11,915 seconds. Actual seed elapsed
  times were 3,124.9 / 3,142.2 / 3,149.2 seconds, excluding initial data loading.

## Four-model staging

Classical, Supervised, Augmented Supervised and Move Prediction appear in that order.
Supervised remains default. Both supervised checkpoint hashes remain unchanged.
Manifest v5 stages seed 42 as **experimental playable**, with direct legal-move
inference and nullable position-value/search fields. The backend policy checkpoint
and validation gate are private artifacts; no policy weights are copied into the
static public directory. Historical evolutionary, self-play and Stockfish-policy
experiments remain in Research.

## Verification and retained failures

- Final pytest: **147 passed, 2 skipped**. CPU continuation, tiny-set learning,
  alignment, exclusions, terminal/history-sensitive draws, promotion vocabulary,
  source freeze, all model identities and integrity checks passed.
- Browser suite: **48 passed, 9 skipped** (live/preview requirements). Separately,
  all four real models plus mobile outage/reconnect passed: **5 passed**, 18.3 seconds.
- TypeScript/Vite build, CLI/module help, Ruff and diff checks passed. Actual local
  seed-42 runtime checks used no installed Torch: seven boards / 21 repetitions,
  about 0.706 ms median and 6.84 ms maximum; this is a small correctness/latency check.
- Docker image execution is **not verified**: no Docker, Podman or other container
  runtime is installed. Existing container CI remains required before deployment.

Retained engineering failures: initial public/private manifest mismatch was fixed;
a source-freeze test rejected an edit made during its run and passed on stable
source; browser setup needed Chromium and execution outside the macOS sandbox;
old three-card expectations/selectors were updated. Some early commands exited 137,
with later clean verification passing. A foreground SSH launch request timed out,
but the detached process was confirmed alive and no duplicate fit was started.
The first post-staging full suite found a historical audit requiring a public policy
weight copy; it now validates the private checkpoint and gate. The combined real-model
browser test exhausted its total 30-second limit; independent per-model tests passed.
The final visual check also caught a stale Vite public-file cache: restarting the
preview served the valid generated SVG correctly. A new image-decode/mobile check
passed (1 test), and the actual report was visually inspected. Restart a local Vite
preview after generating new public files. These are engineering checks, separate
from the three finite training outcomes.

## Private backups, cleanup and cost

Persistent folder in the original checkout: `runs/elite-policy-20261124`.
The pretraining backup has 39 verified members (203,767,547 bytes), including the
original source archive, prepared data and runnable package. The completed Runpod
output archive is `runpod-results.tar.gz`, 166,316,542 bytes, 26 verified members,
SHA-256 `2726ae33c77a04a31bf1cb81fecdae2c09efa5fad5b6d48bbcbf63d8ccf02617`.
It preserves all three best/last models, model/optimizer/RNG/cursor checkpoints,
metrics, sealed selections, final results, gates, source stamp, approval and logs.
A local audit confirmed every minimum-validation selection, source hash and exact
last-export/resume-model match. Operator receipts and local parity checks accompany
this backup; the final receipt bundle is recorded in `delivery.json`.

Backups were verified in both local copies before the independent guard stopped the
GPU. This run's pod `nlapd6nrjsghs4` and 10 GB volume `sx2bx42702` were deleted;
fresh get calls returned 404. The old evolutionary pod/volume were untouched.

Rental window was at most 2.799 hours, including setup, benchmark, evaluation and
backup. Estimated compute is US$2.015; US$1.346 of this run's pod/volume charges
had posted at reconciliation, with posting still incomplete. The conservative
inclusive new-run allowance is **C$5.74**, using the approved conversion/fee/tax
bounds plus C$1.50 storage/rounding allowance, below the C$12 cap. With C$10
reserved for prior/pending spending, the cumulative operating envelope is C$15.74,
below C$40, with C$10 contingency separate. This is not a final CAD invoice.

## Reproduction and review

Commands and protocol: `docs/elite-policy.md`. With the shared local virtualenv,
set `PYTHONPATH=src`. Full training is not CI and requires a new bounded approval
receipt for any additional paid run. Rebuild reports/staging from the frozen outputs:

```sh
PYTHONPATH=src .venv/bin/python -m chessengine.elite_release \
  --root runs/elite-policy-20261124 --manifest artifacts/models/manifest.json
PYTHONPATH=src .venv/bin/python -m chessengine.elite_report \
  --root runs/elite-policy-20261124
.venv/bin/python -m pytest
npm --prefix web run build
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/elite-playwright npm --prefix web run test:e2e
```

Restore data/source from the pretraining backup and run outputs from the verified
archive before invoking those commands in another checkout. Review the focused
local commits on the isolated branch; the user handles any push. Deployment and
architectural improvement treatments remain separate work.
