# Direct Stockfish-move CNN study

This research model chooses a move directly: score every legal move once and
return the largest logit. It does not call alpha-beta, MCTS or Stockfish during
inference. The released demo, its default and all model manifests stay unchanged.
The implementation is ready for validation; no dataset preparation, real teacher
labeling, full training, Runpod provisioning or strength study is claimed here.
Tiny local test fixtures are correctness evidence only.

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

## Frozen bounded protocol

`configs/stockfish-move-policy.json` owns tunable values. It requests 50,000/5,000/
5,000 train/validation/test positions, rounded upward to include the selected
source games' remaining deduplicated samples. Source-game identity defines the
split before phase sampling (up to two positions in each of three phases).

Preparation requires every exclusion file, fingerprints them, and reads only
position/provenance fields. The default source is July 2013, after the January–June
archives used by the newer million-position supervised study. This new archive
must be restored/downloaded before preparation; it is not downloaded implicitly.
Exclude historical corpora, the entire million-position corpus and previous reserved,
evolutionary and augmented-evaluation pools, plus reserved puzzle trajectories.
An overlap with an exclusion position
quarantines that source game. Canonical cross-split conflicts quarantine all
involved games. Complete histories are retained; JSON rows are staged in SQLite
and emitted only after the audit. Missing data, insufficient fresh positions or
changed frozen inputs fail rather than weakening exclusions. Preparation can be
rerun before labeling to validate source hashes; an interrupted preparation is
rebuilt, while completed preparation is immutable.

Teacher labels are the first PV move at 10,000 nodes, one thread, 32 MiB hash,
MultiPV one, maximum skill, strength limiting off, and fresh hash per position.
The executable SHA-256 is pinned to the existing macOS teacher. Record its actual
UCI identity in each resumable label chunk. A Linux teacher is not silently
substituted: freeze a separately identified configuration before labeling with a
different binary. These are the teacher's choices at this budget, not guaranteed
optimal chess moves. Scalar WDL labels remain a different task.

The CNN consumes the 12 player-relative piece planes from feature v3. Its other
85 features provide rights, en passant, clock, repetition and material summaries.
Two padded convolutions (64 then 128 channels, ReLU) feed a 256-unit context layer.
Concatenate that context with a candidate's origin/destination/promotion encoding
(64+64+5); a shared 128-unit ReLU head emits one score per legal move. The
implementation computes the equivalent additive input projection efficiently.
No learned vocabulary can omit a rare promotion. All weights start fresh.

For legal moves A(s) and Stockfish move a*, the loss is
`-log(exp(score(s,a*)) / sum(exp(score(s,a))) for a in A(s))`.
Padding is masked before cross-entropy. Probabilities are move preferences,
never calibrated winning probabilities. Draw claims are automatically taken
before choosing a move; terminal/claimable positions return no move.

Seeds 42/43/44 use AdamW (learning rate 0.0003, weight decay 0.0001), batch size
256, maximum 40 epochs, patience six and gradient norm cap one. Minimum validation
cross-entropy selects each checkpoint; seed 42 remains the representative. No
search expansion is triggered by results. Every seed must finish selection before
the test protocol freezes all checkpoint hashes and opens test shards. Test data
cannot select a replacement checkpoint. Report every seed's test cross-entropy,
top-one/top-three agreement and mean rank, frozen-checkpoint training error,
whole-game bootstrap intervals (1,000 resamples), seed standard deviation
separately, and NumPy encoded-position inference latency (first 32 test records).
The latency excludes board/history encoding and file loading.

Encoded NPZ shards have bounded size; each dataset process caches one decoded
shard. Training shuffles shard order and rows within shards, rather than causing
random decompression of the entire corpus. Zero DataLoader workers keeps CPU
resume exact and avoids multiplying caches. CUDA uses pinned transfers and AMP
(bfloat16 where supported, otherwise float16 with a scaler); CPU uses float32.
Resume stores optimizer/scaler/RNG state, history and elapsed time, and rejects
changed data, config, device, torch version or scientific code. GPU bitwise
reproducibility is not promised. NPZ inference validates schema, arrays, dtype,
shape and finiteness, and imports no PyTorch. The CNN has its own research-only
format; existing MLP and release loaders do not accept it.

## Local preparation and checks

Run from this branch's checkout. In an isolated worktree, point `--input-root` at
the original checkout containing ignored archives, prior corpora and reserved
pools. Outputs always go to the isolated run directory. These commands are
provided for the subsequent approved study; they were not executed on real data
as part of implementation. Report a measured local runtime forecast before
launching the full preparation/labeling stages; respect the two-worker local
limit shared with the evolutionary study.

```bash
export PYTHONPATH="$PWD/src"
python -m chessengine.move_policy_train --help
# Download the July archive separately from https://database.lichess.org/
# to /path/to/original/ChessEngine/data/lichess-2013-07.pgn.zst.
# Preparation records its SHA-256 with every required exclusion source.
python -m chessengine.move_policy_train prepare --input-root /path/to/original/ChessEngine
python -m chessengine.move_policy_train label --input-root /path/to/original/ChessEngine
python -m chessengine.move_policy_train encode
python -m pytest -q tests/test_move_policy.py
```

Preparation and labeling can run on CPUs before GPU rental. Standard Stockfish
[does not evaluate positions on a GPU](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html#can-stockfish-use-my-gpu).
No new experiment runs alongside the existing evolution controller implicitly.

## Runpod handoff and user approval

**Ask the user before starting any Runpod training, including the gradient-based
benchmark.** This PR provisions nothing and grants no paid-compute authorization.
Use a single Secure Cloud GPU Pod and an official PyTorch template, with a
compatible CUDA floor pinned at provisioning. Read current capacity and pricing
before choosing hardware; do not quote historical rates. Keep code and artifacts
private. No public container image or repository is needed.

Prepare the private allowlisted archive after encoding:

```bash
python scripts/package_move_policy.py --dataset runs/stockfish-move-policy \
  --output /tmp/chess-move-policy-private.tar.gz
```

The archive contains Python source, the frozen dataset config, encoded shards,
provenance stamp, requirements and these instructions. It excludes credentials,
Git history, released weights, raw PGNs and unrelated runs. Transfer it privately
to persistent Pod storage after approval. Extract into `/workspace/chess-policy`.
Use the template's CUDA-enabled torch rather than blindly reinstalling a CPU or
incompatible wheel; install runtime dependencies and the project with
`python -m pip install -r requirements-policy.txt` followed by
`python -m pip install --no-deps -e .`. The existing training requirements pin
`torch==2.7.0`; if a different template torch is chosen, record the exact version
in the run and do not resume across versions. Only the dependencies imported by
this pipeline are required: torch and the small pinned policy requirements
(numpy, python-chess, threadpoolctl, certifi, and zstandard for local PGN loading).

After the user approves the paid benchmark, run:

```bash
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m chessengine.move_policy_train benchmark --device cuda
```

`benchmark.json` reports measured FP32 throughput, peak CUDA allocation and a
training-only maximum-epoch forecast across all seeds, explicitly excluding
validation, checkpoint writes, data loading, labeling and setup. It is a
conservative compute reference, not an end-to-end timing guarantee. Quote those
additional costs separately, including storage, tax and currency conversion.
Present the concrete forecast and maximum spend to the user before full training.
This new study's paid budget is unset. Current main contains later GPU-budget
amendments, so the historical C$50 ceiling is not new authorization. Establish a
separate total including contingency and prior commitments with the user.

Once the user approves that training quote, the explicit launcher is:

```bash
CHESS_POLICY_RUNPOD_APPROVED=yes bash scripts/run_move_policy.sh
# After interruption; resumes existing seeds and starts any not yet begun:
CHESS_POLICY_RUNPOD_APPROVED=yes bash scripts/run_move_policy.sh --resume
```

It exits before Python unless the approval marker is set. Launch detached on the
Pod, persist logs/checkpoints on mounted storage, monitor exit status and GPU use,
and verify downloaded hashes before cleanup. The script never creates, stops,
or deletes resources: stop the authorized Pod promptly after outputs are safe;
retained storage can continue billing. Record actual elapsed time and invoiced
cost in the study handoff. CUDA availability failures must not fall back silently
to CPU. Inference accepts a history-preserving JSON position record:

```bash
python -m chessengine.move_policy_train predict \
  --checkpoint runs/stockfish-move-policy/seed-42/best.npz \
  --position /path/to/position-record.json
```

Imitation accuracy and tiny-fixture loss reductions do not establish chess
strength. A later strength comparison needs fresh frozen paired matches,
truncation sensitivity and a separate promotion decision; neither it nor any
public release is part of this implementation.

## Implementation verification

The 14 direct-policy tests pass in a standalone process, including exact CPU
resume, NumPy/PyTorch parity, rule/history handling, leakage rejection, fake
teacher chunk resume, tiny-fixture overfitting and the private archive allowlist.
Ruff and both CLI help commands pass. One combined run printed `150 passed,
2 skipped` and then exited 137; it is retained as an unsuccessful process exit,
not counted as a clean suite run. The Mac showed substantial memory pressure;
the cause of that exit was not established. CUDA training remains unverified and
requires the user's later Runpod approval.
