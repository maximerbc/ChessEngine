# Reproducible study protocol

## Data

Source: https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst (CC0). This small historical corpus is convenient for reproducibility but is not a representative sample of current elite chess.

`make_openings.py` reserves 480 distinct positions after 12 plies: 240 for training starts, 40 for development, 200 for final testing. It deterministically shuffles source games with seed 2026. All reserved source game IDs are excluded from supervised preparation. Training, validation, and test game assignments are hashes of the initial position plus complete move sequence. Exact position duplicates cannot cross splits. Correlation through common opening motifs can still remain; this is not a claim of complete semantic independence.

Sampling takes at most two positions per phase, with phase labels defined by ply ranges rather than material. PGN histories are preserved so repetition features can be reconstructed. The final supervised test split is diagnostic only, never used for training or architecture selection.

Labels require Stockfish with `UCI_ShowWDL`. Teacher analyses clear hash between positions, use one thread and 32 MB hash, and stop at 10,000 nodes. The manifest pins reported identity and executable SHA256. No engine binary is distributed with the web application.

## Order of operations

1. Run the correctness suite and a 10,000-position pilot.
2. Benchmark CPU search and CPU/MPS training; report unavailable devices honestly.
3. Train both main supervised architectures. Select using validation MSE and inference cost.
4. Freeze the RL configuration/search budget after inspecting throughput, before main self-play or final test results.
5. Run identical RL settings for seeds 42, 43, and 44. Each seed gets a separate config and directory.
6. Evaluate candidate checkpoints on development openings. `select` rejects final-test results. Compare candidates with the same baseline/openings/budget.
7. Freeze final protocols with hashes and run the paired final matches. Protocol files are immutable; interrupted matches resume from saved games.
8. Run secondary handcrafted/reference and equal-time comparisons without using them to retroactively select the primary result.
9. Generate the report and public allowlisted assets.

Primary target: 1,000 self-play games per seed and 400 games per main final comparison. These are targets, not a claim of completed work. A benchmark-based reduced study must state actual counts and its reduced statistical reach. Pilot configurations intentionally run only a few games.

## Resumability and provenance

`resume.pt` is trusted local PyTorch state, never accepted from an untrusted upload. It includes optimizer, RNG states, progress, and a signature of identity settings/data. Inference exports are safe array-only NPZ files checked for schema, shape and finite weights.

Metrics record Git commit and dirty state. Pilot runs performed before the first commit explicitly say `uncommitted`. Dataset, teacher, openings, actor, and final-model hashes prevent accidental substitution. Timings differ between resumed runs even when CPU weights are identical.

## Costs and limits

Use local CPU first. Small networks may not benefit from a GPU; game generation remains CPU-heavy. Do not start paid cloud instances without a measured need and an affordable runtime cap. Keep total spend below C$50 equivalent with a 20% reserve; log actual paid costs. Colab chess training requires a paid plan/positive compute balance under its current rules. The provided notebook checks that the user understands this before running chess jobs.

## Reporting

Publish both teacher MSE and playing strength. Include bootstrap intervals, truncation counts, uncapped-pair sensitivity, seed variation, exact search budgets and run length. Calibration is expected-score calibration `(v+1)/2`, not a separately modeled win probability. Positions from the same game are correlated. No Elo claim is warranted from the project's narrow opponent pool.

The optional continued-supervised control matches network-update time and reports the RL generation overhead separately. It is not an equal-total-compute treatment.
