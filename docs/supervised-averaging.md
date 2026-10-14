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

## Completed evidence

All three ordinary trajectories reproduced exactly (maximum weight/loss delta zero). Mean ordinary MSE 0.240760; mean weight-average MSE 0.236791, a 1.648% reduction with all three seeds improving. This does **not** meet the predefined 2% threshold: do not adopt averaged weights for RL. The fixed common RL initialization is `runs/fixed-data-improvements/combined-42/best.npz`.

The prediction ensemble achieved MSE 0.226863, 5.772% below mean member MSE. Median per-position inference including encoding was approximately 193 microseconds versus 65 for one member. The ensemble is a diagnostic, not a selected RL model. No seed-level uncertainty is inferred from this single ensemble.

Full local invocation: 230.8 seconds; paid compute C$0. All seven saved prediction arrays were independently checked against reported MSE and their artifact checksum. Both figures were visually inspected. The local suite passed 55 tests before the separate RL changes. A tiny-study check initially used an unnecessarily strict 1e-10 cross-runtime MSE tolerance; NumPy/PyTorch differed by 3.6e-10. It was corrected to the approved 1e-6 tolerance before protocol commit and full training. Ordinary trajectories still reproduced exactly.

[Read the comparison report](experiments/supervised-averaging/index.html). The next work is the RL pilot, not another supervised search.
