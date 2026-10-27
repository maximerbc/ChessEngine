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

## 2026-10-13 — Diagnose RL regression with matched fixed trajectories

The user requested investigation and remediation of the completed negative RL result. Preserve it and all released assets. First isolate update sensitivity with a bounded nine-fit ablation on saved training-only trajectories: outcome-only reference, tenfold smaller step, and prediction anchoring to the frozen initializer. This is explicitly off-policy diagnostic fitting, not a replacement on-policy study. Select follow-up candidates using development matches only under the frozen screen in `docs/rl-update-diagnosis.md`. No automatic promotion or reuse of final outcomes for candidate selection. Estimate local runtime before execution; no artificial local cap.

## 2026-10-14 — Supervised move ranking and four model pages

The user approved four dedicated model pages and supervised root move ordering, extending the original no-policy-network boundary. A separate copied 128/64 board encoder plus a move scorer learns Stockfish first-PV targets; the existing value network remains frozen. Root-only ordering limits inference overhead and never filters legal moves. Compare classification, fixed-depth work and equal-time strength separately. Use source-game-separated 50k/5k target subsets, three fixed seeds, preselected seed-42 demo and fresh final openings. The RL page uses a real live-self-play checkpoint with its negative evidence; the anchored replay result remains diagnostic research. The latest supervised default requires a development release check against the old default. No additional live RL run is authorized by this batch.

## 2026-10-15 — Research consolidation and policy/value exception

The user authorized the bounded policy/value plan after PRs #9/#10 merged. The inspected local tree is identical to merged main (GitHub comparison: zero changed files); the implementation branch starts from that tree because remote CLI authentication is unavailable. Preserve the merge baseline identity in provenance; user-managed pushes remain mandatory.

Consolidate the merged website into Demo and Research, retaining its board, comparison mode, historical report and exact release identities. Publish an explicit report-asset allowlist independently of the failed checkpoint release gate. The existing released RL default belongs to the original local study; do not attach later v3 regression evidence to it.

Permit a named local research PUCT strategy with one shared policy/value trunk and legal-move scorer. It is separate from released alpha-beta search, uses NumPy inference, history-preserving rule draws and normalized terminal outcomes, and has no transposition merging or tree reuse. Freeze actors per batch. Compare outcome/visit learning with an otherwise identical 0.1 detached prediction anchor. Use common initial parameters, three paired seeds and prospective quality gates; no ranker weights or teacher labels. No release or paid compute is authorized by this experiment.

## 2026-10-15 — User closes release at completed feasibility scope

After the 559.6-hour full-study forecast, the user explicitly requested an immediately pushable finished product and authorized reducing scope. Close this release after the tested website/research implementation and the completed 60-attempt feasibility pilot. All three budgets qualified; 59 games reached rule-based outcomes. This establishes feasibility only. The anchoring hypothesis remains untested.

Gracefully stop the local continuation, preserve its eight partial-main-run game records, and retain the original preregistration/config/scientific source identities unchanged. No update, selection or final match completed. Record the closure and independent pilot audit in a separate versioned release-scope artifact. Future full training needs fresh user authorization; it is no longer a release prerequisite. Keep the released models/default and hosting settings unchanged, verify the final static assets, commit locally, and leave the remote push to the user.

## 2026-10-25 — Three released opponents and explicit explanations

Present Classical, Supervised and Self-play within the single Demo, retaining Research as the other main destination. All use existing released evaluators and shared alpha-beta search. Preserve the manifest-selected default, artifacts, API identities and four-model local-preview compatibility. The public approach definitions are separate from experimental preview descriptions.

Explain evaluation, search and evidence together, with expandable mathematics/checkpoint details. Supervision learns Stockfish assessments, not moves; gameplay does not train models. Match released neural checkpoints against original-study identities before showing its evidence. The original self-play result is inconclusive; later RL regression and policy/value feasibility have separate identities and scope. Show unavailable opponents explicitly without substituting a fallback. Compare available released opponents in the same order with identical position history and budgets.


## 2026-10-26 — Evolutionary evaluation and retiring live self-play

The user authorized an independently initialized, teacher-free CMA-ES evaluator and full quality-gated local study. All evaluation weights, including material effects, are learned; 63 human-designed summaries are explicitly disclosed. The prospective playable representative is seed 42. The frozen protocol specifies pilot, source separation, selection, qualification, fresh confirmation, cap sensitivity and release conditions. A failed pilot stops this study without automatic retries.

Retire RL from public/preview play and analysis, keeping private reproducibility artifacts and a prominent research archive. The original local study was inconclusive; the separate larger study regressed. Historical checkpoint identities are immutable and separate from active release identities. Public cards become Classical, Supervised and Evolutionary; the latter stays disabled until qualified and Supervised remains default. Manifest v3 explicitly declares required models. The old requirement to deploy both Supervised and RL is superseded.

Candidate tournament caps retain unfinished rule results, with neutral tournament fitness reported separately; there is no fabricated position outcome target. Confirmatory final results may block promotion under a preregistered rule, never select a different model. No paid compute, remote pushes, deployment or closed policy/value continuation is authorized.

## 2026-10-27 — Simplify the public demo

The user approved 22 UI annotations. Keep a text-only CHESSENGINE brand, Demo/Research navigation, the demo description and Play control, opponent selection and explanations, Stockfish comparison, gameplay insights, move history and footer links. Remove the demo headline/eyebrow, duplicate opponent links and introduction, public manual-analysis control, and the multi-model Compare workflow. Demo retains only the unnumbered “Test the difference” method note; Research has no method strip or introductory paragraph. The shared research question now asks whether tournament evolution can produce a competitive evaluator, linking to the evolutionary study without changing historical evidence.

Normalize legacy `mode=compare` links to Play while retaining the starting FEN, complete move history and selected model, even offline or through browser Back/Forward. Delete the obsolete multi-model request loop and UI state. Keep `/analyze` for gameplay and optional local-preview inspection, and `/evaluate-comparison` for the retained Stockfish panel. No backend contract, search budget, checkpoint, release gate, experiment, hosting or deployment change is implied. Remote pushes remain user-managed.

### Follow-up annotations — compact research summary and footer

The user removed the remaining “Test the difference” note and requested a shorter opponent explanation and current footer. Replace the public explanation headings, mathematics disclosure and multiple research links with a compact Research summary covering evaluation, shared search and checkpoint-matched evidence, ending in one link to the relevant study (or the Research index when evidence is unavailable). Keep full technical and historical details in the Research reports and optional local previews. The footer now contains only CHESSENGINE and a Research link, preserving the current game when followed. Remove public weight-download links from the footer without changing model artifacts or their release status. Continue on `codex/evolutionary-engine`; the user will push the committed HEAD to a new remote branch and open the PR.
