# Bounded averaging study

Approved 2026-10-12. Configuration: configs/supervised-averaging.json. Keep the fixed-data study's exact development file and three combined-model references. Predictions are averaged with equal weights. Compare against each individual and mean member loss; the latter is not loss of the mean prediction.

Train three runs only, seeds 42/43/44, with the previous combined configuration. Interpret “after epoch 8” as taking the first weight snapshot immediately after completing epoch 8, then every completed epoch. Maintain an equally weighted running mean. Select the lowest validation-MSE averaged checkpoint, but stop using the ordinary model's original patience rule. Verify every ordinary training/validation loss and final/best weight against the historical trajectory within 1e-6. A failed reproduction blocks interpretation.

Adopt averaging only if mean MSE improves at least 2% and at least two seeds improve. Otherwise retain the existing combined configuration. The preselected seed-42 checkpoint of the qualifying configuration initializes every subsequent RL seed. The prediction ensemble is diagnostic only.

Run:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m chessengine.averaging configs/supervised-averaging.json
# Add --resume after interruption; config/data/code hashes must match.
```

MSE remains teacher-target agreement, not chess strength. Bootstrap 2,000 source games jointly for paired comparisons. The ensemble represents one predictor, not three independent replicates. Validation-selection bias remains; final-test data and released models are untouched. Averaging source histories/resume tensors are local trusted artifacts.

Next: proceed directly to the approved RL pilot, regardless of the averaging result. No additional architecture or hyperparameter search.
