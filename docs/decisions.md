# Decision log

## 001 - Shared engine and lightweight inference
Use one Python search implementation for experiments and the API. Train an MLP in PyTorch and export arrays for NumPy CPU inference. This avoids a second JavaScript engine with subtly different semantics, and keeps the server free of the PyTorch runtime. Exact inference parity is tested.

## 002 - Outcome learning before more complex RL
Use Monte Carlo self-play value regression. No teacher labels enter RL. A frozen acting model generates each batch, then equal samples per completed game train the next actor. This is approximate policy iteration with search, not policy-gradient RL or AlphaZero. Truncated games are excluded, which can bias training toward short games; record that limitation.

## 003 - Draw and cache semantics
Automatically take claimable draws, including a claim possible on the next legal move. Search/API/self-play/matches share this convention. Preserve history, and key transpositions by full FEN and move history. This sacrifices transposition reuse for correctness; optimize only with proof/tests of equivalent draw semantics.

## 004 - Private source, public allowlisted demo
The GitHub repository stays private. Railway receives explicitly authorized repository access. Only the static build output is public; allowlisted NumPy model exports and research artifacts are copied by a dedicated command. Secrets and raw training data must never enter `web/public`.

## 005 - Evidence levels and compute gates
Smoke tests prove plumbing, pilots estimate feasibility, and a frozen main study tests hypotheses. Never relabel a small pilot as the planned 400-game, multi-seed experiment. Hardware observations and benchmark-driven scope reductions must be recorded before final testing, not after inspecting results.

## 006 - Budgets are study inputs
Store experimental values in `configs`, not in this log or AGENTS. Use single-threaded NumPy BLAS because threaded tiny matrix multiplies hurt search throughput. Paid compute stays at zero until there is evidence it is useful. The budget ceiling remains C$50 including contingency.

## 007 - First completed study uses a declared local budget
After measuring pilot throughput and before the main RL/matches, fix a bounded local study: three seeds, 60 self-play games each, 48 search nodes/move, and 200 final opening pairs per seed. Retain the original target configs separately. See the preregistration for exact settings and rationale. Do not describe this as completion of the larger target training schedule.

## 008 - Supervised architecture selection
Both specified architectures were fitted on the complete 100,000-position corpus. The small model's validation MSE was 0.306609, the larger model's 0.310330. Select the smaller network before RL, consistent with the predeclared 2% error tolerance and inference-cost preference. Validation selection is distinct from final-test teacher agreement and match evaluation.

## Railway deployment (2026-10-03)

User selected Railway and two services: always-running Nginx for the public website, serverless FastAPI for the engine. This supersedes the Render blueprint. Keep Docker builds separate and non-root, require both neural models in production, and preserve API interfaces and research artifacts. Dockerfiles plus documented dashboard settings define deployment; no legacy Railway configuration file is added. Available free credits are the initial funding source; paid activation needs explicit user approval within the existing C$50 ceiling.

## Supervised generalization diagnosis (2026-10-09)

The next batch diagnoses the released supervised model on the existing development split. Nine fixed-architecture data-size/seed runs precede a predefined optional twelve-run weight-decay/dropout comparison. `configs/supervised-diagnostics.json` freezes all thresholds. Whole-game nested subsets and bootstrap resampling respect within-game dependence. Dropout is training-only; exported inference remains the same feature-v1 dense/ReLU/tanh computation. No released models, final-test data, or hosted artifacts are changed. Fresh final data will be reserved for a later confirmatory study.

## Training coverage extension (2026-10-09)

After the negative regularization study, the user approved 200k total labelled positions. Keep the old corpus and held-out splits unchanged, adding training samples from new source games with the pinned teacher. Freeze this separate experiment in `docs/expansion-200k.md`; do not alter the completed diagnostic protocol or released model. This changes experimental scope, not project invariants.

## Fixed-data optimization and representation (2026-10-11)

The approved next study holds all expanded training/validation examples and labels fixed. Add explicit feature schemas: v1 remains unchanged, v2 rank-reflects Black-to-move inputs and swaps color roles, and v3 appends material counts, phase and balance. Preserve history features and target signs. Legacy defaults remain v1; experimental schemas are rejected by RL entry points. Compare two learning-rate treatments and two representation treatments with a reproduced reference before one conditional combination; no additional search or released-model replacement.

## 2026-10-12 — Bounded averaging before RL

Run one three-seed supervised weight-averaging experiment and an equal-weight prediction ensemble on unchanged data. First weight snapshot is taken after completing epoch 8; ordinary training controls stopping. Retain the approved 2%/two-seed adoption rule and preselect seed 42 for common RL initialization. The ensemble will not initialize RL. No further supervised search precedes the RL pilot.

## 2026-10-12 — v3 RL and bounded stronger study

Enable feature v3 in self-play while retaining v1; keep v2 rejected. Preserve full-history encoding and STM outcomes. Every RL seed starts from the preselected combined supervised seed-42 checkpoint because averaging missed the 2% criterion. Actor state remains frozen per batch, and per-game pending state supports interruption without losing completed generation.

Benchmark 20 games at each of 128/256/512 nodes. Require at least 90% depth-one completion on searched moves and at most 20% truncation. Forecast includes generation, both development checkpoints, fresh final matches, equal-time matches and conservative update overhead, with a 50% throughput margin. Prefer 1,000 games per seed, then 500, then 250; within each count choose the highest qualifying budget. Freeze before main training and enforce 12 hours including the pilot.

Fresh 12-ply final starts come from unused games in the existing archive, excluding canonical corpus/old reserved positions and source games. Do not use old final scores to select anything. Select all three RL candidates on development matches before any fresh final evaluation. Bootstrap shared opening pairs jointly across fixed seeds and report seed SD separately. C$0; no release change.

## 2026-10-13 — User removes local CPU time cap

The user explicitly revoked artificial local CPU allowances and requested runtime estimates before launches. Resume the existing study with `--unlimited-local`, retaining compute accounting, all scientific settings, fixed pilot decision, and completed games. This changes stopping administration only, not model selection or evaluation. Archive the original study protocol and record an explicit code-hash amendment for the runner/budget implementation before resuming. The original 12-hour forecast policy remains historical evidence. The ledger includes conservatively charged interruption downtime and is not an exact measure of active CPU work. Paid-resource authorization remains unchanged.
