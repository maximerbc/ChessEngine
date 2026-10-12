# Stronger self-play RL study

Approved 2026-10-12; exact settings are in configs/rl-improvement.json. This follows the completed averaging study, whose 1.65% weight-average improvement missed the predefined adoption threshold. Common initialization: existing combined supervised seed 42. Three RL RNG seeds isolate learning variability from initialization.

Run locally:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m chessengine.rl_improvement configs/rl-improvement.json
```
Use the same command after interruption. It verifies frozen inputs/code and resumes completed pilot games, pending self-play games, training batches and match games. Add --stage pilot to stop after the affordable-protocol decision. No main training occurs if all pilots fail their quality/cost gates.

## Budget and pilot

Persist a 43,200-second local wall-time allowance, covering pilot, fresh-opening preparation, generation, updates and evaluation. Graceful interruption records active elapsed time; after an unclean crash, the unclosed wall interval is charged conservatively, including downtime. No automatic extension.

Pilot: 20 games each at 128/256/512 nodes, quiescence depth 2, cap 512 plies. Require at least 90% searched moves to complete depth one and at most 20% games to truncate. Try 1,000, 500, then 250 games per seed; choose the highest qualifying node budget within the first game count that fits. Forecast total must fit 32,400 seconds, including a 50% throughput margin and 20-second update allowance per batch. Equal-time forecasts use observed 90th-percentile game length and 100 ms per move. These remain estimates, so the hard allowance independently stops execution safely.

## Main learning and evaluation

Actor frozen per batch; 32 positions per completed game with equal game weighting. One AdamW pass, LR 1e-4, no weight decay or teacher labels. Ten percent random legal exploration during the first 20 plies. Truncated games receive no fabricated outcomes. V3 inputs retain original-history repetition and unchanged STM target signs. Update loss is MSE to actual self-play outcomes, not Stockfish teacher targets.

Evaluate middle (ceiling of half the batch count) and final checkpoints on all 40 development pairs; ties select earlier. Every seed must be selected before fresh final matches start. Fresh final 200 pairs use source-game and canonical-position exclusions against the entire corpus and all old reserved groups; teacher labels are never used in this audit. Each selected RL model plays 400 equal-node final games against the common frozen SL initialization, plus 80 equal-time development games at 100 ms per move with a non-binding node cap.

Primary aggregate: mean score of fixed RL seeds 42/43/44, bootstrapping the same opening pairs jointly across seeds 10,000 times. Report individual seed intervals and seed SD separately. Lower interval >50% supports improvement under this protocol; an interval crossing 50% is inconclusive. Report caps as draws plus uncapped-pair and pessimistic/optimistic sensitivity. Teacher-validation MSE is computed only after all selection and never controls it.

No new final labels, paid resources, public checkpoints or deployment changes. Do not conflate this study with the old 60-game/48-node study or isolate a causal effect of one changed setting.
