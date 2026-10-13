# Stronger self-play RL study

Approved 2026-10-12; exact settings are in configs/rl-improvement.json. This follows the completed averaging study, whose 1.65% weight-average improvement missed the predefined adoption threshold. Common initialization: existing combined supervised seed 42. Three RL RNG seeds isolate learning variability from initialization.

Run locally:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m chessengine.rl_improvement configs/rl-improvement.json --unlimited-local
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

## Executed pilot and main-run launch (2026-10-12)

All 60 pilot games finished without truncation. Depth-one completion was 99.49%, 100% and 100% at 128, 256 and 512 nodes, respectively. Mean game times were 3.46, 8.75 and 15.36 seconds. The frozen decision selected **1,000 games per seed, ten batches of 100, at 128 nodes**, with an 8.71-hour total forecast including the safety margin. Higher budgets did not fit while preserving the target game count.

The fresh final set contains 200 unique source games and canonical positions. An independent metadata-only audit confirmed zero source-game/canonical overlaps with the full supervised corpus or old reserved groups and verified the saved checksum. These checks do not inspect final match outcomes.

The main local run has started. Its results and playing-strength conclusion remain pending; the pilot establishes feasibility only. Raw resumable progress is in `runs/rl-improvement`, excluded from Git. Do not launch a second runner while one is active. After completion, generate compact evidence with:

```bash
MPLCONFIGDIR=/private/tmp/chess-rl-mpl .venv/bin/python -m chessengine.rl_improvement_report runs/rl-improvement docs/experiments/rl-improvement
```

### Authorized local-time amendment (2026-10-13)

The user removed the local CPU time cap. Use `--unlimited-local` for subsequent resumes; report a runtime estimate before launching. The historical pilot decision and all learning/evaluation settings remain frozen. Elapsed-time accounting continues, including conservatively charged downtime. The original protocol and explicit code-hash amendment are preserved with the run evidence.
