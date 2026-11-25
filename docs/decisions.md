# Decision log

## 2026-11-22 — Augmented-supervised fitting completed and preserved

All one million positions were labeled locally with the pinned Mac teacher in 7,000.68 seconds. The A40 pipeline then completed full feature preparation, three independent fits, validation selection sealing and NumPy export parity in 1,192.49 seconds. Summed fit active timers were 426.45 seconds. Seeds 42/43/44 stopped after 15/14/16 epochs under the frozen patience-six rule; best epochs were 9/8/10. Their validation MSEs were 0.2145383/0.2155261/0.2142254 (mean 0.2147633, sample seed SD 0.0006789). Different historical validation pools preclude direct old/new MSE comparisons, and no playing-strength claim follows. Seed42 remains representative.

The complete training state, source/runtime/data identities, source archives and logs passed an independent 204-file local restoration. Preservation identity is in `docs/preservation/gpu-supervised-20261122.json`; large private files remain ignored locally. All optimizer states reload as complete and all three inference schemas/checksums validate. Commit only the small private Research inference bundles and compact evidence. Final position tests and playing matches remain sealed/unexecuted. Keep the released80k default and public assets unchanged.

Following the setup's approved cleanup condition, remove this conversation's four temporary Pods and 20GB volume after verified backup. Provider read-back confirms no Pods or network volumes remain. Available prepaid credits were US$9.16658 from initial US$10; observed credit decrease is not a final tax/FX-inclusive invoice and posted billing is incomplete. The supervised C$25 approval does not authorize evolutionary computation or model promotion.

## 2026-11-21 — Approved supervised execution and local labeling

The user chose Runpod with a C$25 total supervised setup/training ceiling including storage, taxes and conversion, C$5 contingency, and up to C$3 initial hardware checks. This is not evolutionary-compute authorization. The approved instance is one A40 at US$0.59/hour with a 20 GB HIGH_PERFORMANCE volume at US$2.80/month prorated. Keep prepaid credits separate from consumption and actual invoice figures. Review-branch merging, deployment and model promotion remain separate decisions.

Actual A40 verification passed the locked NVIDIA environment doctor, 17 GPU tests and NumPy export parity (maximum benchmark prediction difference 3.58e-7). These checks establish infrastructure correctness, not playing strength. A 1,024-position training-only labeling probe projected 4.58 hours for one million positions on the rented host; the full three-seed synthetic update/checkpoint projection was 22.3 minutes, excluding first-epoch I/O and other pipeline stages. Neither estimate is a completed study result.

After comparing historical local labeling and cloud CPU prices, the user explicitly accepted local labeling followed by GPU training. A new same-sample Mac benchmark measured 156.21 positions/second with four workers (1.78-hour linear projection); two/four-worker targets matched for all 1,024 probe positions. Worker count is an operational setting recorded in the private receipt, not a scientific sample change. Allow for sustained throughput and startup overhead. Do not require the GPU to remain rented while a CPU labeling stage runs.

The prepared corpus passed exact 800,000/100,000/100,000 counts and canonical-position/source-game separation. It retains 180,302 eligible old training positions without reusing targets. Positions SHA256 is 124cce87b231a32acdd0f7eab8916791f728f9b7ec8b83f7c1fbe6e2c06ea0a9. The verified private download preserves the preparation protocol and six source archive checksums. All new-study positions are labeled with the pinned macOS Stockfish 19 binary and explicitly exported NNUE; preserve Linux probe provenance separately and never mix those benchmark labels into the corpus. Local labeling has started; no completed new model or final assessment is claimed.

## 2026-11-21 — Teacher network identity during Runpod setup

The official Stockfish 19 Linux release declares EvalFile only. Teacher pinning must inspect the pinned executable’s UCI options and require exactly its declared NNUE files, including EvalFileSmall only when that option exists. Every declared file retains an explicit hash and path. This corrects the previous two-file setup assumption without changing label nodes, threads, hash, perspective, data splits or learning protocol. No study labels or training runs preceded the correction.

## 2026-11-21 — GPU value studies and preserved legacy release

The user approved a new study architecture and review branch from verified main `ba08f1822285676c33f3dd7c98bf664ab1105896`. This supersedes earlier prohibitions on agent-managed private review pushes and GPU/PUCT production candidates. Agents may push review branches and open PRs; merging, direct-main pushes, visibility changes, paid provisioning and deployment remain separate decisions. Historical protocols, including the two-worker 324-weight experiment, are unchanged.

Before applying implementation changes, archive 1,578 files (1,033,857,195 logical bytes) including all private data/runs, configurations, model exports, reports, PR #15 material and the primary checkout's binary Git diff. Every compressed member was reread and SHA-256 checked. The inventory is `docs/preservation/legacy-20261121.json`; the large local archive stays ignored under the primary checkout's `runs/preservation-20261121`. Initial reads of iCloud placeholders failed and were not accepted as preservation evidence; files were materialized before the successful archive. A separate actual restoration check accompanies delivery validation.

Preserve the released 80,299-training-position Supervised checksum `ff3436d07d18eb4953e24e6c83b55d25bbb46b44094ed347f99320148ae04c4b`, Classical behavior, all 180,302-position candidates, negative historical research and the interrupted evolutionary optimizer. The larger candidate's lower teacher MSE did not pass its development playing gate; no model promotion is implied.

Create separate `gpu-value-v1` studies. Both new value networks use feature v3, 853/128/64/1 layers, ReLU hidden activations and tanh output. JAX/Optax train from scratch; evosax separable CMA-ES evolves independent random parameters. PGX/mctx provide batched legal simulation/tree machinery with a history-preserving automatic-claim adapter. Uniform legal priors, c_puct 1.5, deterministic canonical action ties and no noise define the new PUCT family; NumPy/python-chess provide its CPU serving counterpart. Existing released alpha-beta behavior is unchanged.

The dataset contains exactly one million positions overall: 800k train, 100k validation and 100k final test. Source games split before sampling; canonical positions and games cannot cross splits. Eligible old training positions are retained, but all new-study labels are recomputed with pinned executable and explicit NNUE hashes, 10,000 nodes, one thread and 32 MiB hash per worker. January–June 2013 Lichess sources and complete source manifests are preserved. Fresh validation/test positions exclude old research data/reservations.

The main default is population 64 × 40 generations × 48 games × 3 seeds = 368,640 training games. Opponent groups receive 16 games each, color paired, with 50% handcrafted-PUCT, 25% frozen augmented-supervised-PUCT and 25% archived evolutionary-PUCT fitness. The separate ten-generation **single-seed (20261121)** pilot contains 30,720 games; pilot and all evaluation games are excluded from main totals. The frozen pilot seed 20261121 and its 600 training openings are separate from main seeds 42/43/44 and the 1,000 main training openings. This avoids repeating pilot trajectories in the main prefix. Forecast 184,320/368,640/737,280 main-game scenarios before fixing execution. None is a sufficiency claim. Pilot failure closes the named study without relaxed criteria. Prior depth gates become explicit PUCT coverage checks; draw/cap, decisiveness and improvement gates remain.

The new training budget is unset pending a measured forecast and user provider/budget decision; the old C$50 compute ceiling does not authorize or constrain this new study. No paid resource is provisioned by these commands. Benchmark compilation, complete games, tournaments, memory, CPU serving and interruption/resume. Python 3.12 dependencies are separately locked; CPU checks do not verify NVIDIA execution. New concurrency is hardware-scalable, not capped at two workers. Record all costs and include 20% contingency.

Freeze validation/development choices for all seeds before sealed final assessments. Keep seed 42 representative; use 40 development pairs and separate 200-pair qualification/final pools. Require ≤5% caps and paired conservative 95% lower bounds >50% against Classical and ≥45% against current Supervised for representative and aggregate results. Count capped games as candidate losses in gate sensitivity. Publish seed outcomes and negative/inconclusive findings. Equal CPU time allowances compare PUCT and alpha-beta; simulation counts are never labeled alpha-beta nodes.

The four intended cards are Classical, current Supervised, Augmented Supervised and GPU-trained Evolutionary. Current Supervised remains default. New cards remain unavailable/Research-only until qualification and fresh confirmation. Manifest v4 names architecture, features, search, checksum and evidence; v3 remains readable. Existing API routes and default 1,500-ms serving allowance remain. Release staging is an artifact operation, not deployment.

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

## 2026-11-22: playable augmented preview and smaller strength screen

The user requested a playable augmented model alongside Classical and the unchanged 80k Supervised default, then reduced the initial 2,400-game assessment. Manifest v4 now explicitly authorizes only the frozen seed-42 augmented checkpoint as an experimental playable preview, with qualification false and strength not established. This is an access amendment, not a claim that the historical scientific gates passed. The three public choices describe the one-million-position dataset accurately (800k training/100k validation/100k test). Evolution stays in Research.

The new local development screen uses 20 fresh color-swapped opening pairs against each reference: 80 games in total, seed 42 only, 128 PUCT simulations maximum and 1,500 ms per move for both search families. This provisional simulation ceiling is frozen independently of the still-unresolved GPU evolutionary workload. Record actual completed simulations and root coverage. Report whole-opening-pair uncertainty, conservative capped-game losses and neutral-cap sensitivity; close results may be inconclusive. This screen does not choose a seed/checkpoint, open final position labels, satisfy qualification, or trigger the larger qualification/final runs. The original 2,400-game qualification and separate final protocols remain unchanged and deferred.

Prepare fresh pools from the preserved sources, excluding the complete new corpus, all archived research examples, historical reserved games/positions and puzzle trajectories. These pools are prepared after training, with model selections already sealed; do not claim they were prepared before training. Only development openings are consumed by this screen. Run `PYTHONPATH=src .venv/bin/python -m chessengine.strength_screen --openings PRIVATE/openings.json --output PRIVATE/screen --workers 4`. Preserve the private preparation protocol, source hashes, status, per-game ledgers and resume instructions. Initial estimate: roughly 1–4 hours with four local workers; the mechanical maximum-time allowance is about 4.3 hours plus per-move overhead for 80 games capped at 512 plies. Measure completed-game throughput before refining this estimate. No new cloud resources are provisioned.

Website/release changes are prepared for review. Merging and deployment remain separate user decisions. The existing `stage-release` qualification path remains strict and continues to require replayable qualification and final evidence before marking a model qualified.

## 2026-11-22: alpha-beta search ablation after the failed PUCT screen

The completed 80-game PUCT screen lost every game (40 versus each reference), all by replay-verified checkmate. Preserve its negative result in `docs/experiments/gpu-value/puct-screen-result.json`. Recorded search time was about one ninth of the references' total search time. A separate off-policy diagnostic found 34 last decisions with a legal defense against mate in one: PUCT-128 repeated all 34 blunders, full-time PUCT repeated 26, and the same weights under alpha-beta repeated none. This diagnoses a tactical weakness; it does not establish a counterfactual match win rate. A fixed 2,048-validation-position diagnostic measured MSE 0.22152 for the new evaluator and 0.32449 for the old evaluator on the same records; final tests were not opened. Details are in `search-diagnosis.json`.

The user authorized `configs/supervised-alpha-beta-screen.json`, a separate 80-game local development ablation using the same fixed seed-42 weights and original first 20 development opening pairs with colors swapped. Both sides use the original alpha-beta, qdepth four, a 1,500-ms move allowance and the same nonbinding node ceiling. This tests the evaluator change with search held constant; it does not replay serving's additional 10,000-node cap. Previously seen development openings support a controlled diagnostic, not independent confirmation. Final position labels and final/qualification pools remain sealed. No new fits, cloud resources, model selection, promotion, large-study continuation or website search change follows automatically.

Run `PYTHONPATH=src .venv/bin/python -m chessengine.alpha_beta_screen --openings PRIVATE/openings.json --prior-screen PRIVATE/summary.json --diagnosis PRIVATE/diagnosis.json --output PRIVATE/alpha-beta-screen --workers 4`. The config, inputs, source/runtime and model hashes are frozen; use the private snapshot to resume. Completed games persist, while an unfinished game restarts. Initial planning allowance is roughly 30 minutes to three hours; 80 × 512 plies × 1.5 seconds / four workers = 4.27 hours plus overhead in the time-limit arithmetic. Refine from complete-game throughput and record elapsed time. Keep the Mac awake with its lid open. C$0 new paid compute. Website serving stays unchanged while this test runs.

## Augmented alpha-beta serving amendment (2026-11-22)
- The user authorized a new review branch switching the existing seed-42 Augmented Supervised preview from PUCT-128 to the shared CPU alpha-beta. This supersedes the PUCT serving restriction for `supervised-1m` only. Keep its weights, 800k-training/100k-validation/100k-test dataset identity, experimental access, qualification=false and the unchanged 80k default. Manifest v4 declares alpha-beta and `user-20261122-augmented-alpha-beta`; allow this exact checkpoint/search exception only. GPU training provenance in the immutable checkpoint is not a serving-strategy declaration.
- Preserve the completed same-search development screen: 16 wins/3 draws/21 losses against Classical (43.75%, paired 95% interval 30–57.5%); 26/3/11 against 80k Supervised (68.75%, 55–81.25%). All 80 games replay-verified, no caps, 2,492.89 seconds locally and C$0 new paid compute. One fixed seed and 20 reused color-swapped opening pairs per opponent do not establish qualification, general superiority or an Elo.
- Serving retains the existing 10,000-node/1,500-ms defaults and qdepth four. The comparison used a nonbinding 2^31-1 node ceiling; its scores are not evidence for the exact bounded demo configuration. API limits, Classical and 80k behavior stay unchanged. Display the declared search consistently in play, analysis, Stockfish comparison and Research.
- Preserve the failed PUCT study and its frozen protocol, all backups and historical evolutionary requirements. This change does not retrain models, resolve GPU evolutionary simulations, relax qualification/final gates or open final tests. Merge and deployment remain user decisions.


## 2026-11-24: fresh elite human-move CNN baseline

The user replaced the prior Stockfish best-move direction with the reference
PyTorch CNN trained on actual moves in fresh elite games. The old studies remain
immutable historical evidence. Freeze the one-million-example protocol in
`configs/elite-policy.json` and `docs/elite-policy.md`; no additional architecture
or optimization treatment precedes review of this baseline.

The approved reference exception retains thirteen absolute planes and a global
classifier, corrected to a complete 4208-class vocabulary. History still controls
legality and automatic draws. NumPy direct-policy inference has no position-value
head or search and cannot masquerade as alpha-beta or a win probability.

The requested lineup is Classical, Supervised, Augmented Supervised and Move
Prediction. Manifest v5 extends v3/v4 identity validation. Existing supervised
weights/default are unchanged. Fixed seed 42 can be experimental-playable after
validation learning and functional export gates, without a playing-strength
claim; failed gates remain unavailable.

All local preparation, tests, backups and the concrete Runpod package precede
launch confirmation. No billable benchmark or training is authorized before that
confirmation. The new cumulative budget is C$40 operating plus C$10 contingency,
inside C$50 total; prior and pending charges count. No remote push, merge or
deployment is included.

## 2026-11-24: completed elite baseline staging

All three fresh seeds completed the frozen 100-epoch recipe. Validation CE selected
epochs 8/8/7; the fixed seed-42 checkpoint passed the learning and CPU-export gates
and is staged locally as experimental playable. Human-move agreement is not
playing-strength evidence. Full metrics, uncertainty and actual curves are in
`docs/experiments/elite-policy`; no improvement treatment was added.

The backend manifest declares the private NumPy policy and its validation gate;
static assets contain aggregate evidence and the manifest. Historical integrity
audits validate that private checkpoint/gate without requiring a public policy
weight copy. The two existing supervised public artifacts remain unchanged.

Verified private backups preceded independent guard shutdown and removal of only
this run's pod and volume. Billing remains partly unposted; conservative allowances
are recorded separately from posted charges. No push, merge or deployment occurred.
## 2026-11-22 — Direct Stockfish-move CNN research

The user selected a direct legal-move policy and a PyTorch CNN, inspired by the
Skripkon PyTorch/TensorFlow videos. The tutorials imitate human PGN moves; our
separate experiment labels positions with the pinned Stockfish first-PV move.
Use feature-v3 spatial inputs plus complete rule/history features and a shared
candidate scorer, without an observed-move vocabulary or search at inference.
All weights start fresh. This permits a research-only NumPy CNN export alongside
the existing MLP schemas, preserving the training-only PyTorch boundary.

The bounded protocol and Runpod handoff are in `docs/stockfish-move-policy.md`;
values are in `configs/stockfish-move-policy.json`. Source-game splits, canonical
exclusions, all-seed checkpoint selection and a frozen held-out-test protocol
protect study integrity. Training metrics measure teacher imitation only.
Implementation and tiny-fixture verification do not count as a completed study.
The user explicitly requires confirmation before Runpod training; no paid
benchmark, full run, provisioning, publication or promotion is authorized here.
Released models, production search and ongoing evolutionary work are unchanged.

## 2026-11-23 — Direct-policy full run without a pilot

The user approved local full-data preparation and labeling followed by direct
Runpod training of seeds 42/43/44, with no separate pilot/benchmark. The first
real epoch remains part of the fit and supplies measured time/cost forecasts.
The new authorization is C$10 total inclusive of setup/storage/taxes/conversion,
with C$2 contingency and C$8 operational allowance. Retain the frozen scientific
settings and stop with resumable checkpoints rather than silently reducing scope.
Pin the teacher's actual EvalFile NNUE hash, avoid replaying full histories when
indexing position-only exclusions, and retain full histories for training records.
Record local progress, cloud receipts, all seed results and any failures. Back up
and verify outputs before deleting only the new training resources; preserve the
existing evolutionary Pod/volume. No deployment or model promotion is included.

## 2026-11-23 — Correct direct-policy size to one million

The user corrected the intended supervised move-policy dataset to exactly
800,000 training / 100,000 validation / 100,000 test positions. The earlier
60,007-position preparation is preserved as superseded, with no student CNN fit
started. Temporary Pod `k6g519tjra6wdk` was stopped and deleted, followed by its
new volume `x371etfrtl`; existing evolutionary resources were untouched. The
first billing reads had no posted records and must not be interpreted as zero
actual charge. The previous local labels are not a completed million-position
study or a pilot. Enforce exact counts before labeling, encoding and cloud fitting;
source games remain in one split even if unused samples at a quota boundary are
discarded. The C$10 budget is unchanged. Corpus reuse versus a fresh source pool
is being resolved with the user before dependent work proceeds.

## 2026-11-23 — Reuse position inputs and relabel best moves

The user delegated the easier/faster corpus choice and authorized proceeding to
Runpod. Reuse the existing million-position pool's inputs and split assignments
in a new copy. Its checksum is
`124cce87b231a32acdd0f7eab8916791f728f9b7ec8b83f7c1fbe6e2c06ea0a9`.
The stricter canonical audit found 188 duplicate rows including 69 cross-split
overlaps. Quarantine whole conflicting/reserved source games, deduplicate and
refill from the already audited fresh July pool, enforcing exact 800k/100k/100k
counts without changing original files. Scalar teacher values never become move
targets. Generate new pinned first-PV labels with at most two independent teacher
processes, one thread/32 MiB each, using bounded chunks. Historical measured
single-worker throughput forecasts roughly four hours for one million labels;
two workers target about two to three hours. No new GPU is rented during local
labeling; no separate pilot or automatic model promotion is introduced.

## 2026-11-24 — Complete the million-position direct-policy study

The full no-pilot Runpod job completed on exactly 800k/100k/100k positions using
frozen source 7b43f04. Early stopping completed 10/7/13 epochs for seeds 42/43/44;
selected epochs remain 4/1/7 and seed 42 remains representative. Held-out top-one
teacher agreement is 31.008%/6.858%/32.128%. Seed 43 collapsed to uniform scores;
publish it as a failed seed rather than replacing it or launching extra fits.
The complete private report and machine-readable evidence live in
`docs/experiments/stockfish-move-policy-1m-20261124`.

The initial relative-logit-only export check masked large probability errors.
Require explicit probability and chosen-move parity and record all failures.
Correct NumPy's final reduction so identical large activations cannot gain
artificial move preferences through row-dependent rounding. Preserve original
weights, Torch training forward, selections and test results; historical resume
still requires the original scientific source/runtime. Seeds 42/44 pass the
strict sampled check; seed 43 still fails probability parity and is unqualified.
Do not relax tolerance or claim that every export passed.

All artifact bytes, selected hashes, raw teacher chunks and source/data archives
are backed up locally. The independent guard stopped the GPU after the verified
backup marker; its Pod and new volume were deleted with 404 read-backs. Existing
evolutionary resources and production models remain unchanged. Conservative
total spend is approximately C$0.70 including the aborted setup and allowances;
main-run/volume billing was still unposted, not proven free. No new training,
merge, deployment or promotion is authorized by closing this study.


## 2026-11-24: all supervised variants available experimentally

The user approved a single demo catalogue with Classical plus nineteen distinct
supervised configurations, represented by their existing validation-selected
seed-42 exports. Start from elite baseline commit bdc0f82, integrate current main
8c64143 and Stockfish-policy commit 797582f, and preserve unrelated worktrees.
The 180k model and fixed-data reference have identical trained arrays and appear
once. Historical pilot fits, seed alternatives, epoch snapshots, the diagnostic
prediction ensemble, learned ordering, RL and Evolutionary play are excluded.

Manifest v6 supports the complete explicit catalogue with fail-closed artifact,
kind, encoder, trained-array and gate checks. Supervised remains default; existing
Supervised, Augmented Supervised and human-policy checkpoints remain byte-for-byte
unchanged. Additional value variants use shared alpha-beta. The Stockfish and
human move policies use distinct exact NumPy encoders and direct legal-move
selection, without invented position scores or search statistics. The human
encoder's history omissions remain documented; complete histories govern rules.

This is an experimental-play authorization, not qualification or a re-evaluation
of historical failed gates. Source and export hashes live in the frozen config
and manifest; reproducible staging cannot change trained arrays. Only aggregate
checkpoint-matched evidence enters the public research catalogue. New weights
remain in the private API image, with no new public downloads. The user authorized
this feature branch push and PR creation; merging and deployment remain separate.


## 2026-11-24 — Three playable approaches, manifest v6

The approved lineup is Classical, Augmented Supervised and Direct-Move CNN. The augmented seed-42 value evaluator becomes the default; it retains the shared alpha-beta search. The later human-game CNN uses its existing direct-policy NumPy implementation and fixed seed-42 validation selection (epoch 8). Its completed export gate is required. This explicitly supersedes the older requirement to retain the 80k supervised default and the production alpha-beta-only restriction.

Manifest v6 requires both neural artifacts and exact approved checkpoint identities. Earlier declarations are retained in `artifacts/research/release-v5.json` and `public-release-v5.json`; historical checkpoints are preserved. Playable availability is distinct from qualification or demonstrated playing strength. The augmented three-seed mean validation MSE (0.21476) is agreement on its own validation dataset, not a comparison of playing strength across studies.

Research presents the three active methods first, then four collapsed experiment families. Original charts/replay live within their original-study archive entry. Study links expand the containing family. Earlier CNN collapse/export failures, negative self-play results, failed ordering gates and interrupted evolutionary runs remain discoverable. A snapshot of the augmented-baseline evolutionary interruption is published independently of the original study; no run is restarted.

## 2026-11-25 — Integrate main without reusing its manifest version

Main introduced a broader v6 supervised catalogue while the three-approach release was being prepared. The three-approach release now uses v7, retaining its exact checkpoint identities, augmented default and direct-human-CNN gate. Main’s v6 declaration is archived in `artifacts/research/catalogue-release-v6.json`; its loader, provenance, private artifacts, training safeguards and numerical checks remain supported by explicit historical fixtures. The approved three-opponent product supersedes the catalogue’s public selection.

Earlier supervised-variant reports remain in the position-evaluation archive. The old Stockfish study permalink aliases its comprehensive dated entry, preserving both report URLs without duplicate cards. No studies are rerun or promoted by this merge.
