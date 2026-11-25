# Historical supervised catalogue (v6)

The active release uses manifest v7 with Classical, Augmented Supervised (default), and Direct-Move CNN. This document preserves the earlier catalogue and its reproducible staging workflow. Its checkpoints and numerical checks remain available for research; the broad catalogue is not the current public lineup. Staging must use a separate output directory and refuses to overwrite the v7 release.

Manifest v6 lists nineteen supervised exports; Classical is built in. The
searchable grouped chooser uses `/models`, including descriptions, grouping,
strategies and capabilities. Saved opponent URLs and complete histories survive
refreshes and navigation. An unavailable requested model is never substituted.

Use the existing validation-selected seed 42 for every distinct completed
configuration. `configs/demo-models.json` pins each source export and source root.
The fixed-data reference duplicates the 180k model; it appears once. Intermediate
epochs, extra seeds, smoke fits, the prediction ensemble, learned root ordering,
RL and Evolutionary play are excluded.

Reproduce staging from the private original repository and completed elite worktree:

```sh
PYTHONPATH=src .venv/bin/python scripts/stage_demo_models.py \
  --source-root /path/to/original/ChessEngine \
  --elite-root /path/to/completed/elite-worktree \
  --output /path/to/separate-research-export
PYTHONPATH=src .venv/bin/python -m chessengine.research_catalog
```

Every source hash must match. Staging prepares and validates the entire collection
before copying artifacts, preserves original released exports, strips incidental
metadata from new value exports, checks exact trained arrays and representative
inference, then writes the manifest. Run staging from a clean review branch and
review all generated changes before committing. Production also checks schemas,
unique trained weights, complete declared identities and policy export gates.
Earlier supported manifest versions retain their existing loading paths.

Value models use alpha-beta with 10,000 nodes, 1,500 ms and qdepth four by default.
Direct policies use `direct-policy-v1`, `position_value=false`, `search=false`,
`move_probabilities=true`. `/analyze` and `/evaluate-comparison` keep their URLs
and request shapes; policy responses carry null value/search statistics and legal
move probabilities. Stockfish remains a separate reference, never a fallback.
The human policy retains its original thirteen-plane encoder and omissions;
full histories still enforce legality and automatic draw claims.

The 180k/combined release gates and averaging adoption gate remain failed. The
Stockfish-policy failed seed 43 remains reported; playable representative seed 42
passed recorded functional export checks. Experimental access establishes no
strength ranking, Elo or calibrated win probability. No new final tests or
checkpoint selection are part of this change.

Only inference weights and named gate files enter the API image. Newly exposed
weights are absent from public static assets; existing public Supervised and
Augmented exports remain unchanged. One API worker/replica, no training runtime,
no volume, no automatic deployment or paid resource activation. Roll back by
reverting this PR, restoring the preceding manifest and allowlisted image assets.


## Verification evidence

`docs/experiments/supervised-variants/validation.json` records the local checks and
retained integration failures. The full Python suite passed 219 checks with one
skip; the final catalogue suite passed 51 checks, including the two subsequently
added path/dedup cases. All nineteen actual exports matched training math on
synthetic legal boards without new labels or test-set selection. The compiled
website passed real gameplay/comparison for every opponent and mobile outage
recovery (21 checks); two additional live research/comparison checks passed.
Docker was unavailable locally; the existing production-container workflow is the
container validation gate for this PR. No hosted deployment is claimed.
