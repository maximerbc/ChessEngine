# Fixed-data optimization and representation study

## Frozen protocol

Approved 2026-10-11. Configuration: `configs/fixed-data-improvements.json`.
Use precisely the ordered development records in `runs/supervised-200k/subset-all.jsonl`: 180,302 training and 9,789 validation positions. The runner checks the dataset against the historical fits, split integrity and canonical (rank-reflected, color-exchanged) train/validation overlap. An overlap is an error, never a reason to silently delete examples. Test records are rejected.

Fit all five predefined configurations for seeds 42, 43 and 44 from scratch: reference, smaller constant learning rate, adaptive learning rate, player-relative features, player-relative features plus material. Keep the approved 128/64 MLP, AdamW, batch size, epoch ceiling, patience and weight decay. The exact settings are versioned in the configuration. Stop if the reference does not reproduce historical weights and validation MSE within 1e-6. No treatment interpretation is allowed past a failed reference gate.

A configuration qualifies with at least 2% lower mean validation MSE and improvements in at least two seeds. Rank by mean validation MSE, then measured inference latency for exact ties. Combine the best qualifying optimization and representation configurations only if both groups qualify, adding three runs. Maximum 18 runs. Do not expand the search after observing results.

## Mathematics and compatibility

Minimize `mean((v_theta(x) - y)^2)` with unchanged side-to-move targets. A learning rate changes the scale of an optimizer update, not the target. The adaptive scheduler halves the rate after a validation plateau, using its own absolute 1e-4 improvement threshold; early stopping retains the existing strict-improvement rule. Persist both optimizer and scheduler state.

Feature v1 retains its original 841 inputs and inference behavior. Feature v2 changes only the representation: Black-to-move boards exchange own/opponent color roles and reverse ranks, retaining files. Castling rights and en passant follow the same transformation. The turn indicator is always one. Compute halfmove and repetition features using the original complete history, before transforming inputs. Do not negate targets.

Feature v3 appends 12 values: five own and five opponent piece counts divided by starting counts (8,2,2,2,1), total non-pawn phase units divided by 24, and own-minus-opponent material divided by 40. Piece values remain (1,3.2,3.3,5,9); phase units are (0,1,1,2,4). Promoted material is not clipped. The extra 12 inputs add 1,536 first-layer weights, so this is not an identical-parameter-count comparison.

Metadata dispatches the encoder and validates dimensions. Existing checkpoints need no conversion. Experimental v2/v3 models are rejected by RL entry points. Released checkpoints and public assets remain untouched.

## Evidence and reproducibility

Record frozen-checkpoint training/validation errors, all seeds, learning-rate/validation curves, updates, time, inference latency and existing subgroup errors. Bootstrap source games jointly for treatment and reference predictions, with 2,000 resamples per seed. These intervals describe matchwise prediction-error differences conditional on selected models, separately from seed variability; repeated validation selection remains a limitation. Lower MSE is not evidence of stronger chess without subsequent matches.

Run locally, with no paid compute:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m chessengine.fixed_data configs/fixed-data-improvements.json
# Add --resume after interruption; identical code, config and data are required.
MPLCONFIGDIR=/private/tmp/chess-fixed-mpl .venv/bin/python -m chessengine.fixed_report runs/fixed-data-improvements docs/experiments/fixed-data-improvements
```

Raw predictions/checkpoints remain under ignored `runs/`; compact evidence and report are committed. The runner freezes source/config/code hashes and commit provenance before fitting. CPU resume preserves weights and scheduler state exactly. Prior studies' strict code-hash checks require their original commits to resume; their saved models/results remain readable.

CI executes only the tiny synthetic study, including its conditional combination path. Full experiments run locally. Fresh final-test data and controlled playing-strength evaluation are deferred.
