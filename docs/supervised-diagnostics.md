# Supervised diagnostics: reproduce and interpret

This development study uses the existing teacher-labelled training and validation records. It does not evaluate test targets, change released weights, or deploy anything. The frozen protocol is `configs/supervised-diagnostics.json`; it was committed before the full study began.

## Completed findings

All 21 planned runs completed locally. [The report](experiments/supervised-diagnostics/index.html) contains the figures, source-game confidence intervals, subgroup tables, all seed results and twenty failure examples.

| Training positions | Mean validation MSE | Seed standard deviation |
|---:|---:|---:|
| 10,005 | 0.60172 | 0.00097 |
| 40,000 | 0.34322 | 0.00412 |
| 80,299 | 0.30132 | 0.00237 |

All three full-data seeds triggered the regularization screen. Stronger and weaker decay produced mean MSE 0.30194 and 0.30203; dropout 0.1 and 0.2 produced 0.31301 and 0.31252. None met the predefined improvement criterion. A training/validation gap can motivate regularization without guaranteeing that it helps: constraining the model can also make useful relationships harder to fit.

The released checkpoint's validation MSE is 0.30661 (95% source-game interval 0.29427–0.31946), versus 0.66609 for the training-mean baseline and 0.37874 for the handcrafted evaluator. The signed residual is −0.03233. Its largest phase MSE is in the move-count middlegame group (0.36568); positions with more than one and at most three pawns of absolute material imbalance have MSE 0.49822. These exploratory groups overlap and have different target distributions, so they do not establish causes. Near-zero-target positions favour a constant-zero predictor by construction; that subgroup result is not evidence that constant-zero is a better overall engine.

The ranked next steps are to test broader source-game coverage with an update-matched control, audit teacher-label stability on a fixed development sample before mass relabelling, and only then reconsider representation/capacity. No additional experiments or labels are included in this batch. Reserve fresh final-test data before any later confirmatory study.

## Commands

From the repository root, with the training environment installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m chessengine.diagnostics configs/supervised-diagnostics.json
```

To continue an interrupted study or reuse completed runs, append `--resume`. A change to input data, frozen checkpoint, configuration or core implementation causes a refusal rather than silently combining incompatible experiments. Intentionally changed studies need a new output directory and configuration. Training and per-run results resume at saved epoch boundaries; interrupted epoch work may be repeated.

Generate the report independently from the saved study:

```bash
MPLCONFIGDIR=/private/tmp/chess-diagnostics-mpl .venv/bin/python -m chessengine.diagnostics_report runs/supervised-diagnostics docs/experiments/supervised-diagnostics
```

`index.html` is a standalone local report with relative PNG figures and JSON evidence. Serve its directory or open it directly in a browser. It is intentionally outside the website's deployment output. Raw source data, predictions, subset files, resumable optimizer state and intermediate checkpoints remain under ignored `data/` and `runs/` directories.

## Interpretation

The frozen audit uses the released supervised evaluator. New baseline fits start from scratch with the same 128/64 architecture and optimizer settings, but use a deterministic whole-game ordering. Thus their seed-42 result need not equal the historical released checkpoint.

Subset thresholds are approximate position counts: complete games are added until the target is reached. Seeds 42/43/44 share identical nested subsets and validation data. More positions imply more updates per epoch; this does not isolate data quantity from compute. The report records actual positions, epochs, optimizer steps and times.

Compare the training/validation errors recomputed at the selected checkpoint, with dropout off. Online training curves combine predictions from changing weights and, for dropout runs, stochastic masks; they are not directly comparable with frozen validation error.

Bootstrap intervals resample source games with replacement, retaining every sampled position within each selected game. Metrics remain position-weighted; game sizes are not equalized. A paired difference below zero favours the supervised model. These intervals describe validation-sample uncertainty; seed standard deviation separately describes retraining variability. Neither controls selection bias from repeated validation use.

The diagnostic gap rule schedules the four predefined treatments only when at least two full-data seeds exceed a 10% validation/training MSE gap. It is not a statistical overfitting test. A 2% mean improvement and two improving seeds designate a promising development candidate, not a confirmatory result.

Material groups count minors as one phase unit, rooks as two and queens as four. The high group includes 17+ units rather than stopping at 24 because promotions can exceed the initial material total. Existing phase labels remain move-count proxies. Sparse groups are flagged, and the twenty largest errors are deliberately selected examples.

## Verification

`pytest` includes a tiny end-to-end diagnostic study, all conditional treatments, report generation, resume/configuration safeguards, metric checks, whole-game subset checks, grouped bootstrap checks, and exact dropout resume/export parity. Full nine-plus-twelve runs never execute in CI. The existing Python workflow runs the tiny smoke test automatically.

Before adopting any candidate, reserve new final-test data and run controlled paired matches. No playing-strength claim follows from this report alone.
