# Move-ranking study — evaluation pending

Three local CPU fits are complete. Seed 42 selected epoch 9, seed 43 epoch 11, and seed 44 epoch 10 using validation cross-entropy. Validation contains 5,003 positions; total labeled positions are 55,006. See `training.json` for every epoch and the split-exclusion audit.

| Seed | Validation cross-entropy | Top-one teacher agreement | Top-three agreement |
|---|---:|---:|---:|
| 42 | 3.01596 | 14.45% | 31.98% |
| 43 | 3.00225 | 14.79% | 33.32% |
| 44 | 3.02138 | 13.87% | 31.56% |

These classification results are modest and do not establish stronger chess. The seed-42 representative was fixed before fitting. Root-ordering match evaluation is pending; no deployment gate has passed and released models remain unchanged.

The original teacher-value MSE is a different task. The ranker's cross-entropy penalizes low probability assigned to Stockfish's preferred legal move. It can improve search ordering without changing the frozen value network, but inference overhead and imperfect rankings can also hurt. Equal-time matches test the combined effect.

Reproduce/resume from the repository root:

```sh
.venv/bin/python -m chessengine.ranking_data configs/move-ranking.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m chessengine.ranking_train configs/move-ranking.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m chessengine.ranking_evaluate configs/move-ranking.json --stage benchmark
# After reporting the measured full-run estimate:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m chessengine.ranking_evaluate configs/move-ranking.json
.venv/bin/python -m chessengine.ranking_report configs/move-ranking.json docs/experiments/move-ranking
```

Resume verifies data, checkpoint and scientific code identities. Do not edit frozen scientific code to force a resume through a mismatch. Raw labels, checkpoints and per-game records stay in ignored run directories.
