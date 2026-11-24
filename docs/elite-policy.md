# Elite human-move baseline

This is a new, separately named baseline (`elite-pytorch-baseline-20261124`),
superseding the old Stockfish best-move experiment as the active move-prediction
direction. Historical studies, failures, data and weights remain untouched.

The user approved the protocol on 2026-11-24 before fitting. No architecture
search or second treatment is included. Synthetic correctness/overfit tests do
not constitute the study. Full GPU execution still requires the explicit launch
confirmation after this package and its budget forecast are reviewable.

## Reference and deliberate differences

Independent implementation of the method in
[Skripkon/chess-engine, commit 97238ead](https://github.com/Skripkon/chess-engine/tree/97238ead1892adca5e8626c568cd8129d5993ee9/engines/torch).
Credit belongs to the original author for the reference method. No reference
weights or serialized move dictionary are imported.

Input is exactly 13 absolute 8×8 planes: white pawn/knight/bishop/rook/queen/king,
black equivalents, then the union of legal move destination squares. a1 is
row zero/column zero. Two padded 3×3 convolutions (13→64→128), ReLUs, flatten,
8192→256 dense/ReLU and 256→4208 logits reproduce the reference design.
Train with unmasked cross-entropy, Adam 0.0001, batch 64, clip gradient norm 1.0,
FP32 and 100 epochs, without dropout, weight decay, scheduler, AMP or TF32.
Initialization matches the reference: Kaiming uniform convolutions and Xavier
uniform dense weights, with framework-default biases.

Approved differences: complete deterministic UCI vocabulary, fresh audited data,
validation-only checkpoint selection, three seeds, exact CPU continuation and
strict NumPy export. The 4208 classes are 4032 unequal source/destination pairs
plus 176 valid promotion geometries and pieces, sorted lexicographically. Classes
can be geometrically impossible; only python-chess legal moves can be played.

This encoder lacks explicit side-to-move, castling-right, en-passant, halfmove
and repetition planes. The legal-destination plane contains only partial state
information. This limitation is retained to establish the requested baseline,
not presented as an improved full-state model. Full original histories remain
in the data and all game/API rule handling. Draws are claimed automatically,
including a claim available through an intended next legal move.

## Frozen data and selection

Start with the November 2025 [Lichess Elite Database](https://database.nikonoel.fr/)
ZIP, then prior months descending if needed. Preserve ZIPs and SHA-256 receipts.
Each label is the next legal human move, paired with the board **before** it;
no Stockfish labels are generated. Whole games are parsed/validated before
examples are accepted. Ignore examples after an automatic terminal/draw state.

The exact quota is 800,000 training, 100,000 validation and 100,000 test examples.
Stable full-game hashes own splits (80/10/10) before sampling. Position keys
retain board/turn/castling/en-passant and omit clocks. Global deduplication keeps
one owner. Exclude previous corpora and their source games, reserved position
pools and puzzle positions. Quarantine source games touching reserved pools.
The immutable exclusion receipt lists every input hash. Disk-backed indexes and
bounded compact shards keep preparation within an 8 GB local environment.

Freeze `configs/elite-policy.json` and the complete dataset manifest before
fitting. Fresh seeds are 42/43/44. Seed 42 is always representative. Select each
checkpoint by minimum **global-vocabulary validation CE**, earliest epoch on
ties. Complete and seal every seed attempt before creating a test loader.
Test results cannot select an epoch, a replacement seed or another architecture.

Report global CE and legal top-one/top-three agreement and played-move rank,
frozen-checkpoint training CE separately from online loss, and all epoch curves.
Use 1000 whole-source-game bootstrap samples for 95% intervals; report seed
mean/sample standard deviation separately. Human-move agreement is not Elo,
playing strength, Stockfish imitation or a calibrated winning probability.

## Demo gate and interfaces

The user authorizes experimental play without a strength claim after seed 42
passes correctness/export checks and the lower 95% whole-game bootstrap bound
on validation (legal top-one accuracy minus uniform-legal accuracy) is positive.
Preserve failed seeds and failed gates. Never silently switch to seed 44.

Manifest v5 retains the exact original Supervised and Augmented Supervised
identities and Supervised default. Its optional `elite-policy` entry declares
`human-move-policy`, `direct-policy-v1`, schema/vocabulary hash and gate checksum.
Missing declared artifacts or mismatched identities fail startup. Until the real
gate passes, the fourth card is visible and unavailable; no random test weight is
shipped as a trained model. v3/v4 manifests remain supported.

`/models` declares position-value, search and move-probability capabilities.
`/analyze` returns nullable value/depth/nodes/search and sorted legal rankings
for the policy. It performs one forward prediction and no search; no Stockfish
fallback. `/evaluate-comparison` reports policy rankings alongside a separately
named Stockfish assessment. Probability is normalized over legal moves for
display, never described as winning chances. Runtime imports NumPy, not Torch.
The first three opponents retain shared alpha-beta at existing bounded settings.

## Reproducible preparation and Runpod handoff

```sh
PYTHONPATH=src .venv/bin/python -m chessengine.elite_data configs/elite-policy.json \
  --source-root /absolute/path/to/original/ChessEngine \
  --archives /absolute/path/to/elite-archives --output runs/elite-policy-20261124
PYTHONPATH=src .venv/bin/python scripts/package_elite_policy.py \
  --dataset runs/elite-policy-20261124 --output runs/elite-policy-training.tar.gz
```

The preparation command resumes only the same inputs/configuration and verifies
all completed shards. Histories/raw PGN, optimizer states and training bundles
stay private, outside public web assets.

Use one Secure Cloud GPU, pinned official PyTorch 2.8.0/CUDA 12.8 image and one
durable network volume, created only after stock/image/price reads. Pin host CUDA
floor 12.8. The GPU is not rented during local preparation. No Jupyter/public
HTTP endpoint is needed; transfer and execution use the supported SSH/CLI lane.

After explicit approval, verify the bundle manifest, install
`requirements-elite-policy.txt` into the template environment, and use
`scripts/run_elite_policy.py benchmark --approval <receipt>` for a bounded
30-second measured gradient benchmark. Benchmark updates are discarded.
The approval receipt binds study, exact dataset hash and packaged source hashes,
allowed actions, user confirmation reference, deadline and cumulative CAD
forecast. It is created only after actual user approval; no example approval is
automatically accepted. Before fitting, add 50% measured overhead plus 900 seconds
for setup/validation/export/backups; refuse a forecast beyond the approved
deadline. Do not reduce epochs, seeds, precision or data to make it fit.

An independent injected-credential guard must report `armed` before updates.
The guard stops this pod at the deadline even if training hangs, or after the
operator writes a verified-backup marker. Train/evaluate with the approved job
wrapper; preserve atomic model/optimizer/RNG/sampler position every 500 updates
and every epoch. CPU continuation is exact; CUDA bitwise determinism is not
promised. Capture runtime versions, source commit/file hashes and full failures.

Budget: C$50 cumulative project ceiling, C$40 cumulative operating allowance,
C$10 reserved contingency, including prior/pending charges, setup, benchmark,
storage, conversion fees and taxes. Refresh account billing and stock at launch.
Keep forecasts distinct from posted bills. Verify local byte backups and every
checkpoint hash before removing only this run's new pod/volume, with control-plane
read-backs. Existing evolutionary resources are not authorized for changes.

After completed evidence, stage the fixed representative explicitly:

```sh
PYTHONPATH=src .venv/bin/python -m chessengine.elite_release \
  --root runs/elite-policy-20261124 --manifest artifacts/models/manifest.json
PYTHONPATH=src .venv/bin/python -m chessengine.elite_report \
  --root runs/elite-policy-20261124
```

No remote push, merge, deployment, paid hosting or improvement treatment is
authorized by this implementation. The user receives local commits and a tested
release handoff; full training/release results must not be claimed in advance.
