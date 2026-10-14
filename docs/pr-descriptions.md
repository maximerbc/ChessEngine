# Prepared pull request descriptions

These are local handoff drafts, not evidence of remote PR creation. Earlier batches may already have been pushed and merged by the user.

## PR 1: Build the shared chess engine and supervised learning foundation

Base: `main` | Head: `feat/foundation`

Adds a shared Python chess search engine and a reproducible supervised evaluator pipeline. Legal moves come from python-chess; negamax, alpha-beta pruning, bounded quiescence, draw-history handling, training and NumPy inference are implemented in the project. A FastAPI endpoint and React board make the same engine playable locally.

Validation: chess-rule integration, forced mates, node/time limits, repetition-history behavior, and PyTorch/NumPy export parity. The foundation branch contains the initial Python CI and browser project. Trained release assets arrive with the showcase PR; the initial app explicitly offers the handcrafted baseline when no manifest is installed.

## PR 2: Add self-play reinforcement learning and controlled evaluation

Base: `feat/foundation` (retarget to main after PR 1 merges) | Head: `feat/rl-study`

Adds Monte Carlo outcome learning with frozen self-play actors, equal game weighting, checkpoint/RNG resume, and exclusion of truncated games. Match protocols pin models and openings before evaluation, swap colors, bootstrap opening pairs, and keep checkpoint selection on development matches. Includes multi-seed runners, the preregistered local compute adjustment, report generation, and mathematical walkthroughs.

Validation: exact CPU training/resume equivalence, outcome-sign tests, actor freezing, truncation exclusion, leakage rejection, immutable protocols, paired statistics, and rejection of test-set checkpoint selection. Evidence levels distinguish smoke tests, pilots and the bounded local study.

## PR 3: Publish the research showcase and deployment artifacts

Base: `feat/rl-study` (retarget to main after PR 2 merges) | Head: `feat/showcase`

Packages measured research results and portable model checkpoints into a polished Play/Compare/Research experience. Adds all-seed reporting, recorded-game replay, public weight downloads, HTML/PDF reports, a private-repository Railway deployment setup, paid-Colab notebook guidance, and end-to-end verification.

Validation: the final validation summary is in `docs/validation.md`. Deployment requires the user's GitHub push and Railway repository authorization; no public deployment is claimed from local checks alone. No paid compute has been used.

## 4. Deploy the chess demo and engine on Railway

Replaces the Render blueprint with separate non-root production containers: an always-running Nginx website and a serverless FastAPI engine. The report and saved games remain accessible during engine wake-up. Production startup requires both checksum-validated neural models. Existing API contracts, model weights and research results are unchanged.

Adds container CI covering published assets, legal engine responses, checkpoint hashes, failure on missing/corrupt models, and real browser gameplay across separate origins. Documents private-repository setup, spending boundaries, health checks and rollback. Screenshots are test artifacts rather than changes to the showcase image.

Validation: see the Railway section in `docs/validation.md`. Container CI must pass before merge; local checks alone do not verify Docker images or a public deployment. Target main after the showcase PR. No remote push, account authorization, paid activation or deployment is included.

## 5. Diagnose supervised generalization with controlled experiments

Base: `main` | Head: `feat/supervised-diagnostics`

### Change

Adds a frozen-model audit, deterministic whole-game learning curves, and a preregistered conditional regularization comparison. Includes resumable sequential training, inference-compatible dropout, game-bootstrap intervals, subgroup errors, failure examples, and a separate reproducible HTML report with compact evidence. Raw predictions, data and experimental checkpoints remain ignored.

### Evidence

Completed all 21 local CPU runs at C$0. Mean validation MSE was 0.60172, 0.34322 and 0.30132 at 10,005, 40,000 and 80,299 training positions. The screening rule triggered for all three full-data seeds. Neither weight-decay treatment nor either dropout treatment met the predefined improvement criterion. Larger datasets also receive more optimizer updates; these results do not isolate the effect of data quantity or establish stronger chess play.

### Validation and boundaries

42 Python tests passed, including the tiny complete study, whole-game nesting, metric/bootstrap calculations, dropout/export parity, exact CPU resume and mismatch rejection. Ruff and whitespace checks passed. The generated report and figures were visually inspected and compact results were checked against saved raw predictions. GitHub CI remains pending the user push.

Released checkpoints, training labels, deployment settings and public website assets are unchanged. No paid resources or final-test selection were used. Full reproduction instructions are in `docs/supervised-diagnostics.md`.

## 6. Expand supervised training to 200k positions

Base: `main` after diagnostics merges | Head: `feat/supervised-200k`

### Change

Extends the corpus with new training-only source games while preserving every original record. Adds pinned Stockfish labeling in resumable, checksum-verified chunks, a fixed three-seed MLP experiment, and a comparison report with paired game-bootstrap intervals and compute accounting. Validation and test records remain unchanged; no released model or deployment settings change.

### Validation

The local Python suite passes 46 tests, including leakage exclusions, original-record preservation, teacher identity checks, corrupted chunk rejection, label reuse on resume and report metric verification. Ruff and whitespace checks pass. Full experiment findings are recorded in `docs/expansion-200k.md`; report figures and browser layout were visually inspected. GitHub CI requires the user push.

### Results

Completed 100,003 new labels from 22,192 new games, bringing training to 180,302 positions and the total corpus to 200,003. All three expanded fits improved: mean validation MSE fell from 0.30132 to 0.27350 (9.23%). Each paired source-game bootstrap interval favours the expanded model, conditional on these selected checkpoints. Original records and all chunk checksums were independently verified. Labeling used 13.7 minutes elapsed; all work used local CPU with C$0 paid compute.

### Interpretation

More data also produces more updates per epoch. This comparison cannot isolate data quantity from compute, and lower teacher-target error does not establish stronger chess play. The source remains the same historical Lichess archive. Paid compute is C$0; source data and trial checkpoints are ignored.

## 7. Improve supervised fitting with fixed training data

Base: `main` after the 200k PR merges | Head: `feat/fixed-data-improvements`

### Change

Adds a frozen, resumable optimization/representation study using the same 180,302 training and 9,789 validation examples. Introduces player-relative and material feature versions with checkpoint dispatch, preserves legacy inference, saves adaptive scheduler state, and rejects experimental features in RL. Includes a baseline-reproduction gate, canonical leakage audit, source-game bootstrap comparisons, subgroup analysis and a separate report.

### Results

Completed the 15 predefined fits plus three conditionally authorized combination fits. The reference reproduced historical weights exactly. Relative/material inputs with constant LR 0.0003 reduced mean validation MSE from 0.27350 to 0.24076 (11.97%); all three seeds improved. The adaptive schedule failed the development criterion. Single-position inference rose from approximately 43 to 64 microseconds including encoding. Local paid compute: C$0.

### Validation

52 local Python tests passed, covering exact CPU/scheduler resume, schema compatibility and export parity, orientation/history/castling/en-passant/material handling, target preservation, leakage rejection and tiny complete-study execution. Ruff and whitespace checks pass. Saved prediction metrics, hashes, subgroup totals and rankings were independently checked; report figures and browser content were inspected. Remote CI runs the tiny study only and remains pending push.

### Boundaries

No new labels, augmentation, final-test evaluation, RL training or deployment change. Released checkpoints and public website assets are untouched. Repeated validation selection and unequal compute remain limitations; lower teacher-target MSE does not establish stronger chess. Fresh confirmatory data and controlled matches are proposed separately. Full evidence and reproduction instructions: `docs/fixed-data-improvements.md`.

## 8. Compare supervised averaging methods

Base: `main` after fixed-data improvements | Head: `feat/supervised-averaging`

### Change and evidence

Adds equal-weight prediction ensembling and a bounded three-seed weight-averaging study on unchanged data. Saves averaging/resume state without changing ordinary updates or early stopping. Ordinary trajectories reproduced exactly. Weight averaging reduced mean MSE by 1.65%, below the predefined 2% adoption rule; retain the existing seed-42 model for RL. Prediction averaging reached MSE 0.22686 but costs about three times one-model inference and is diagnostic only.

### Validation

55 local Python tests passed, including arithmetic, unchanged trajectories, exact resume, export parity and the tiny complete study/report. Full prediction metrics and checksums were independently verified; figures inspected. Paid compute C$0. No final-test selection, new data, public model or deployment changes. Remote checks require user push.
