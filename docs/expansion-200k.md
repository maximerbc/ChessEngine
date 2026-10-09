# Training-data extension to 200k total positions

## Protocol frozen before extension labels or training

The user selected 200,000 total labelled positions on 2026-10-09. Preserve the existing 100,000 records byte-for-byte; add approximately 100,000 training positions from previously unused source games in the existing Lichess January 2013 archive. Preserve all accepted samples from the last game, so the total may exceed 200,000 by up to five. This increases training from 80,299 to about 180,299 while validation stays 9,789 and test stays 9,912. Test labels are not evaluated or used for selection.

Assign games with the existing deterministic split function; skip non-training assignments, any existing corpus game, and all reserved opening source games. Deduplicate against all existing positions and reserved opening positions. Sample up to two positions per existing move-count phase, with a deterministic per-game random seed. Reusing the same archive broadens coverage within that period, not across time controls, ratings or eras by design.

Reuse existing labels. New labels use the exact original Stockfish 19 executable hash, 10,000 nodes, one thread and 32 MB hash per engine, with cleared hash for each position. Two independent workers reduce elapsed time. Each completed 1,000-position chunk has a checksum and immutable identity for resume. Failed/incomplete chunks are recomputed; existing complete chunks must pass verification.

Train three new 128/64 baseline models with seeds 42, 43, 44, using the original optimizer, early stopping and epoch limit. No dropout or additional regularization search. Compare mean validation MSE and individual seeds against the 80,299-position baseline from the completed diagnostic study. This is an epoch-based comparison: more data also means more optimizer updates. An update-matched control is deferred, so no claim of isolating data quantity from compute is allowed. No candidate replaces the released model without a later playing-strength study.

The initial 100-position local labeling benchmark took 2.34 seconds, implying about 39 minutes for 100,000 positions with one worker. This is a rough estimate from existing training positions, not a guarantee for new games. Actual time is recorded in manifests. All work is local CPU, with C$0 paid compute.

## Commands

```bash
.venv/bin/python -m chessengine.expand_data configs/expansion-200k.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m chessengine.diagnostics configs/supervised-200k.json
```

The first command automatically resumes verified chunks. Append `--resume` to the second command to resume training. Source data, labels and experimental checkpoints remain ignored. The previous diagnostics branch is unchanged; this extension is a separate local PR batch based on it.

After both studies finish, generate the separate comparison report:

```bash
MPLCONFIGDIR=/private/tmp/chess-expansion-mpl .venv/bin/python -m chessengine.expansion_report runs/supervised-diagnostics runs/supervised-200k docs/experiments/supervised-200k
```

The report verifies identical validation records, checks all six checkpoint hashes and recomputes their predictions. It reports paired source-game bootstrap intervals for each seed's expanded-minus-original MSE, separately from training-seed standard deviation. Curves against optimizer steps expose unequal training budgets without claiming to control for them. Raw comparison predictions remain under `runs/supervised-200k`; compact evidence and figures are committed.
