# Direct Stockfish-move CNN study

This research CNN learns Stockfish's chosen legal move at a fixed 10,000-node
budget. At inference it scores all legal moves and chooses the largest score,
without Stockfish or search. Released demo models, search and manifests remain
unchanged.

The full million-position Runpod study completed on 2026-11-24. Representative
seed 42 achieved 31.008% test top-one agreement; seed 44 achieved 32.128%, while
seed 43 collapsed and failed strict CPU export probability parity. All artifacts
are backed up, and the temporary Pod/volume are deleted. No model was promoted.
The [complete report](experiments/stockfish-move-policy-1m-20261124/README.md)
contains every seed, intervals, hashes, failures and costs.
The frozen training source is commit
`7b43f0478d89c15562d28f48cdd235538430e752`.

## Why PyTorch, and what changes from the videos

Both videos by NikolAI Skripko point to
[Skripkon/chess-engine](https://github.com/Skripkon/chess-engine), inspected at
commit `97238ead1892adca5e8626c568cd8129d5993ee9`:

- [PyTorch video](https://www.youtube.com/watch?v=-WDRaRUjRMg&t=540s): 13 input
  planes (12 piece planes plus legal destinations), padded 64/128-channel
  convolutions, dense 256, move logits, integer cross-entropy, explicit CUDA
  selection. Its training notebook has no validation/checkpoint-selection loop.
- [TensorFlow video](https://www.youtube.com/watch?v=v3jMr0Ppd9Y&t=320s): 12 piece
  planes, unpadded 64/128-channel convolutions, dense 256, softmax, one-hot targets
  and categorical cross-entropy. Its validation split operates on positions,
  without an explicit source-game leakage audit.

Both imitate the moves in human PGN games. Neither generates Stockfish labels.
Both build a vocabulary from observed moves and filter illegal predictions only
at inference; neither explicitly represents full rule state/history. Both
materialize their datasets in RAM. Their differing data, architectures and
hardware do not establish a framework or playing-strength winner. PyTorch is our
engineering choice because the project already uses it and it supports our
explicit CUDA/resume/export workflow. We reimplement the ideas; no tutorial code
or weights are copied into the project.

## Exact million-position dataset

The approved split is exactly **800,000 training, 100,000 validation and 100,000
test positions**. `configs/stockfish-move-policy.json` owns all tunable settings.

The user selected the easier/faster route: reuse the existing million-position
value-study corpus as position inputs in a separate copy. Old scalar targets are
discarded; new Stockfish move labels are generated. Preparation verifies the
immutable source hash, checks source-game ownership and reserved pools, removes
canonical duplicates, and quarantines whole games involved in cross-split or
reserved conflicts. It retains 999,187 source records and refills 813 positions
from the audited fresh July pool (447 training, 168 validation, 198 test). The
original corpus remains unchanged. Exact quotas are required, rather than
rounding up by source game.

Canonical exclusion keys use the player-relative position's first four FEN
fields, conservatively ignoring clocks/history for overlap detection. Full
history is nevertheless retained for feature encoding, teacher analysis,
repetition and automatic draw claims. Required exclusions and source hashes are
recorded in `positions.manifest.json` and `reuse-audit.json`. Historical source
positions are reused; this is not a claim that the new test positions have never
appeared in another project study.

Two independent one-thread Stockfish workers label bounded chunks. Each uses the
same pinned Stockfish 19 executable and NNUE, 10,000 nodes, 32 MiB hash, MultiPV
one, maximum skill, strength limiting off and fresh hash per position. The first
PV move is the target. These are the teacher's choices at this budget, not
guaranteed optimal chess moves. Human moves provide positions, not targets.
Checksummed chunks support resume without relabeling completed work.

All 1,000,000 targets passed legality, terminal-state and canonical split checks.
Encoding produced 2,599 bounded NPZ shards, each with a recorded checksum.
Local stage wall times were 164.0 seconds for preparation, 8,773.4 seconds for
labeling and 853.6 seconds for encoding. Summed teacher-worker seconds are a
different quantity from elapsed wall time.

## Features, target and network

The target is an integer index into that position's complete legal-move list.
For legal moves A(s) and teacher move a*, cross-entropy is
`-log(exp(score(s,a*)) / sum(exp(score(s,a))) for a in A(s))`.
Padding is masked before the loss. Probabilities express move preference, not
winning probability. Terminal or automatically claimable positions return no
move.

Feature v3 contains 853 player-relative board features:

| Features | Count |
| --- | ---: |
| Twelve 8×8 piece planes | 768 |
| Castling rights | 4 |
| En-passant target or none | 65 |
| Normalized halfmove clock | 1 |
| Twofold/threefold repetition indicators | 2 |
| Normalized material counts | 10 |
| Phase and material balance | 2 |
| Canonical turn flag (always one) | 1 |
| Total | 853 |

The board is mirrored and colors swapped when Black is to move. Complete
histories determine repetition summaries; the network does not consume the
entire move sequence. Material summaries are handcrafted inputs, while every
network weight starts randomly.

Two padded 3×3 convolutions (64 then 128 channels, ReLU) process the twelve piece
planes. The flattened board representation joins the remaining 85 features in a
256-unit ReLU context layer. Every candidate has 133 one-hot-equivalent move
features: 64 origin squares, 64 destinations and five promotion categories
(none, knight, bishop, rook, queen). A shared 128-unit ReLU head emits one score.
The implementation computes the equivalent additive input projection
efficiently. No observed-move vocabulary can omit an underpromotion. The network
has 2,250,049 parameters and imports no existing value/ranker weights.

## Frozen training and evaluation

Train seeds 42/43/44 sequentially with fresh weights, AdamW, learning rate 0.0003,
weight decay 0.0001, batch size 256, at most 40 epochs, patience six and gradient
norm cap one. Minimum validation cross-entropy selects each checkpoint; seed 42
remains the representative regardless of seed outcomes.

All three selections and hashes must freeze in `test-protocol.json` before any
test evaluation. Report each seed's test cross-entropy, top-one/top-three teacher
agreement, mean teacher-move rank and whole-game bootstrap intervals (1,000
resamples). Report seed standard deviation separately from sampling uncertainty.
Also report frozen-checkpoint training error separately from online training
loss. NumPy latency uses the first 32 encoded test positions and excludes
history/board encoding and file loading. These metrics do not establish playing
strength or Elo.

Training shuffles shard order and rows within each shard. Each process caches one
decoded shard; zero DataLoader workers avoids multiplying caches and preserves
exact CPU resume. CUDA uses pinned transfers and mixed precision (BF16 when
supported, otherwise FP16 with a scaler). CPU uses float32. Resume stores model,
optimizer, scaler, RNG, CUDA RNG, history and elapsed time; changed data, config,
device, package versions or scientific source are rejected. Accelerator bitwise
determinism is not promised.

The separately versioned CNN NumPy export validates schema, array names, shapes,
dtypes and finiteness. PyTorch is training-only. Existing MLP/release loaders do
not accept this research format. Downloaded hashes and NumPy/PyTorch export
parity must be verified before cloud artifacts are removed.

## Reproduction and private Runpod execution

Run local preparation from the review worktree, pointing at the original
checkout containing the immutable corpus, refill pool, teacher and exclusions:

~~~bash
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/prepare_move_policy.py --input-root /path/to/original/ChessEngine
python scripts/package_move_policy.py --dataset runs/stockfish-move-policy \
  --output /tmp/chess-policy-private-1m.tar.gz
~~~

The data CLI lives in `chessengine.move_policy_data`, so CPU preparation need not
import PyTorch. Progress and stage receipts are written in the run directory.
Standard Stockfish [does not evaluate positions on a GPU](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html#can-stockfish-use-my-gpu).
GPU rental begins only after local preparation and encoded checksums pass.

The user authorized direct full training with **no separate pilot or compulsory
benchmark**, keeping every first-epoch update. The total cap is C$10 including
setup, storage, conversion and tax; plan within C$8 and retain C$2 contingency.
Pause with resumable checkpoints if the unchanged remaining protocol cannot fit
the budget deadline. Never silently shrink the dataset, seeds or settings.

The actual run uses one Secure Cloud RTX 4090 (24 GB), EU-RO-1, US$0.89/hour,
and a new 10 GB STANDARD network volume. These are recorded launch-time catalog
reads, not a promise of future price or capacity. The official catalog image is
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, with a host CUDA floor of 12.8.
Observed runtime: PyTorch 2.8.0+cu128, CUDA 12.8, NumPy 2.2.6, python-chess 1.11.2,
BF16 supported. No SSH/Jupyter or public ports are exposed.

The allowlisted private archive excludes credentials, Git history, released
weights, raw PGNs and unrelated runs. Verify its SHA-256 before unpacking and
verify every source/shard hash before fitting. Keep the template's CUDA PyTorch:

~~~bash
python -m pip install -r requirements-policy.txt
python -m pip install --no-deps -e .
CHESS_POLICY_RUNPOD_APPROVED=yes python scripts/run_move_policy_job.py
# Resume only after restoring the same verified inputs/runtime:
CHESS_POLICY_RUNPOD_APPROVED=yes python scripts/run_move_policy_job.py --resume
~~~

The explicit approval marker and exact million-position guard remain required.
The launcher trains all seeds, then evaluates only after complete selection.
A budget pause exits without opening test data. Outputs and resumable
checkpoints stay on the network volume. Encoded shards are cached on ephemeral
SSD for I/O; the verified complete source/input archive remains on durable
storage. Restore that cache from the same archive after a restart.

An independent container deadline guard uses the provider-injected stop
credential in memory only. Its 4.5-hour launch deadline and a ten-minute earlier
training deadline leave room for checkpoint backup. The conservative deadline
estimate is C$7.70, including C$0.10 allowance for prior setup and using
USD/CAD 1.4369 plus 5% conversion and 25% tax allowances. This is an estimate,
not an invoice. Actual first-epoch updates were retained; the measured
train-plus-validation duration was 18.4 seconds.

After successful downloads, verify selected checkpoint hashes, complete resume
artifacts and NumPy parity, retaining any explicit failures. Stop compute after
verified byte backups while local diagnostics finish; delete this run's Pod and
volume after the full artifact backup is confirmed. Failed exports remain
unqualified and do not justify leaving an idle GPU running.
Existing evolutionary resources remain untouched. Record posted billing
separately from estimates and pending charges.

## Verification and retained failures

The final focused suite passed cleanly: 20 tests including the independent guard.
All downloaded checkpoint hashes verify. Strict functional export checks pass
for seeds 42/44 and fail for seed 43; the actual-model verifier exits nonzero and
writes that failure. A relative-only logit comparison had hidden severe
probability distortion from seed 43's very large common logits. NumPy now reduces
each output row identically, with regression tests against artificial preferences
between identical large activations. Neither trained weights nor the Torch
training forward pass changed. All selections and original test measurements
remain frozen. Historical resume requires the original 7b43f04 source/runtime;
changed scientific-file hashes are correctly rejected.

All four workflows passed for training source commit 7b43f04: Python
correctness, GPU-study CPU checks, browser demo and production containers.
Tests cover CPU resume, NumPy parity, history/rule state, leakage rejection,
teacher chunk resume, reuse/refill, exact counts, two-worker teacher settings,
tiny-fixture overfitting, archive allowlisting and the independent guard.

An earlier combined local suite printed 150 passed / 2 skipped, then exited 137.
It is not counted as a clean process result. Subsequent separate baseline and
policy processes exited cleanly (136 passed / 2 skipped and 14 passed); later
integrity checks were added and CI passed. The cause of the local exit was not
established; the Mac showed memory pressure.

An initial teacher attempt rejected manual configuration of python-chess's
managed MultiPV option before labeling. Removing that redundant configure entry
preserved MultiPV one and the teacher budget; failure receipts remain archived.

The superseded 60,007-position preparation remains archived. No CNN fit used it.
The first temporary GPU rental was stopped/deleted after the user's million-
position correction, before uploading or fitting. Posted Pod billing for that
attempt is US$0.01443172; volume billing was still unposted at the subsequent
read, not proven free.

A foreground cloud unpack command timed out with incomplete extraction.
Detached setup completed from the same verified archive, cached encoded shards
on SSD, and reverified all 2,599 shard hashes before training. No gradient updates
were discarded or relabeled as a pilot.

No merge, deployment, model promotion or public research publication is implied.
