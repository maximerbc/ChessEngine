# Fixed-data optimization and representation study

## Frozen protocol

Approved 2026-10-11. Configuration: `configs/fixed-data-improvements.json`.
Use precisely the ordered development records in `runs/supervised-200k/subset-all.jsonl`: 180,302 training and 9,789 validation positions. The runner checks the dataset against the historical fits, split integrity and canonical (rank-reflected, color-exchanged) train/validation overlap. An overlap is an error, never a reason to silently delete examples. Test records are excluded; the source checksum must match the historical development-only file.

Fit all five predefined configurations for seeds 42, 43 and 44 from scratch: reference, smaller constant learning rate, adaptive learning rate, player-relative features, player-relative features plus material. Keep the approved 128/64 MLP, AdamW, batch size, epoch ceiling, patience and weight decay. The exact settings are versioned in the configuration. Stop if the reference does not reproduce historical weights and validation MSE within 1e-6. No treatment interpretation is allowed past a failed reference gate.

A configuration qualifies with at least 2% lower mean validation MSE and improvements in at least two seeds. Rank by mean validation MSE, then measured inference latency for exact ties. Combine the best qualifying optimization and representation configurations only if both groups qualify, adding three runs. Maximum 18 runs. Do not expand the search after observing results.

## Mathematics and compatibility

Minimize `mean((v_theta(x) - y)^2)` with unchanged side-to-move targets. A learning rate changes the scale of an optimizer update, not the target. The adaptive scheduler halves the rate after a validation plateau, using its own absolute 1e-4 improvement threshold; early stopping retains the existing strict-improvement rule. Persist both optimizer and scheduler state.

Feature v1 retains its original 841 inputs and inference behavior. Feature v2 changes only the representation: Black-to-move boards exchange own/opponent color roles and reverse ranks, retaining files. Castling rights and en passant follow the same transformation. The turn indicator is always one. Compute halfmove and repetition features using the original complete history, before transforming inputs. Do not negate targets.

Feature v3 appends 12 values: five own and five opponent piece counts divided by starting counts (8,2,2,2,1), total non-pawn phase units divided by 24, and own-minus-opponent material divided by 40. Piece values remain (1,3.2,3.3,5,9); phase units are (0,1,1,2,4). Promoted material is not clipped. The extra 12 inputs add 1,536 first-layer weights, so this is not an identical-parameter-count comparison.

Metadata dispatches the encoder and validates dimensions. Existing checkpoints need no conversion. Experimental v2/v3 models are rejected by RL entry points. Released checkpoints and public assets remain untouched.

## Evidence and reproducibility

Record frozen-checkpoint training/validation errors, all seeds, learning-rate/validation curves, updates, time, inference latency and existing subgroup errors. Bootstrap source games jointly for treatment and reference predictions, with 2,000 resamples per seed. These intervals describe prediction-error differences conditional on selected models, separately from seed variability; repeated validation selection remains a limitation. Lower MSE is not evidence of stronger chess without subsequent matches.

Run locally, with no paid compute:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m chessengine.fixed_data configs/fixed-data-improvements.json
# Add --resume after interruption; identical code, config and data are required.
MPLCONFIGDIR=/private/tmp/chess-fixed-mpl .venv/bin/python -m chessengine.fixed_report runs/fixed-data-improvements docs/experiments/fixed-data-improvements
```

Raw predictions/checkpoints remain under ignored `runs/`; compact evidence and report are committed. The runner freezes source/config/code hashes and commit provenance before fitting. CPU resume preserves weights and scheduler state exactly. Prior studies' strict code-hash checks require their original commits to resume; their saved models/results remain readable.

CI executes only the tiny synthetic study, including its conditional combination path. Full experiments run locally. Fresh final-test data and controlled playing-strength evaluation are deferred.

Reference APIs: [PyTorch scheduler](https://docs.pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) and [python-chess mirror convention](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.mirror).

## Executed results

Completed all 18 authorized CPU fits; no additional searches. All three historical references reproduced with exactly zero weight differences and identical reported MSE. The canonical-position audit found zero train/validation overlaps. Mean validation errors:

| Configuration | Mean MSE | Seed SD | Reduction versus reference |
|---|---:|---:|---:|
| Reference | 0.273497 | 0.002873 | — |
| Smaller learning rate | 0.263259 | 0.000467 | 3.74% |
| Adaptive learning rate | 0.272469 | 0.003086 | 0.38% |
| Player-relative | 0.266620 | 0.001805 | 2.51% |
| Player-relative + material | 0.244172 | 0.002870 | 10.72% |
| Combined: relative + material, LR 0.0003 | **0.240760** | **0.000839** | **11.97%** |

Smaller LR and relative-plus-material qualified independently, triggering the planned combination. All three combined seeds improved versus reference. Their paired source-game 95% MSE-difference intervals were [-0.04027,-0.02481], [-0.04350,-0.02868], and [-0.03745,-0.02071]. The combined mean is 1.40% below relative-plus-material alone, but that comparison was not the predefined success criterion. No claim of a separately established combination effect is made.

Adaptive LR improved only seed 42; seeds 43/44 selected the same weights as reference, before their later rate reductions. This is evidence against this specific schedule in this study, not against scheduling in general. Player-relative inputs alone met the mean/seed criterion but two seed-level bootstrap intervals span zero, illustrating why that rule is a development screen rather than proof.

The added material features are linear combinations of existing piece indicators. Writing them as m=Ax gives W_x x + W_m m = (W_x + W_m A)x in the first layer. Thus v3 adds no new position information and does not expand the unconstrained function class relative to v2. It changes parameterization, initialization and effective regularization. Those effects are not isolated by this experiment.

The combined model averages about 64.3 microseconds per single-board evaluation, versus 43.3 for reference in this local benchmark, including encoding. Stronger prediction accuracy therefore has a search-throughput tradeoff. Training/validation epoch work totaled 142.4 seconds; the full invocation took 363.1 seconds including data preparation and diagnostics. Paid compute C$0. Per-run fit/diagnostic wall times exclude bootstrap work, while full invocation time includes it. Report generation and verification are outside those timings.

Next recommendation: freeze the combined configuration for fresh confirmatory data and controlled chess matches, keeping all three seeds. Preserve the reference as the comparator. Evaluate both equal-node and equal-time play because of encoding overhead. This is a proposed separate follow-up, not an experiment performed here. Current released models remain unchanged.

The [comparison report](experiments/fixed-data-improvements/index.html) includes every seed, subgroup errors, learning-rate curves and compact evidence. Validation completed on 2026-10-12. Pending documentation/test edits during fitting explain dirty provenance flags; saved scientific code hashes match the pre-study implementation commit.
