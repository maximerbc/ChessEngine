# Bounded policy/value self-play study and website consolidation

Status: implementation authorized by the user on 2026-10-15. The website and research pipeline are implemented and locally validated. The prospective scientific protocol is `docs/policy-value-preregistration.md`, with frozen settings in `configs/policy-value.json`; this planning document remains the design rationale. The actual feasibility pilot is in progress. No complete main study, model promotion or deployment is claimed. See `docs/policy-value-operations.md` and `docs/experiments/policy-value/` for execution evidence.

## Post-merge baseline check

Verified directly on GitHub after the user merged both PRs on 2026-10-15:

| Merged work | Merge commit | Verified state |
|---|---|---|
| [PR #9 — Codex/move ranking](https://github.com/maximerbc/ChessEngine/pull/9), including the prior RL update diagnosis | `ccc976ea7aecb46a509a1ba81fb1d471cd567ec3` | Merged into main; six checks passed |
| [PR #10 — Codex/showcase four models](https://github.com/maximerbc/ChessEngine/pull/10) | `1f6318bb0d1320427898dbe2e63ec185d127c334` | Merged into main; six checks passed |

GitHub's comparison from the inspected local head `e0b25a7c21f30d747352d57d1a660b2fe250d22a` to the second merge commit shows **two merge commits and zero changed files**. The original proposal therefore already incorporates the merged implementation and evidence. These merges change integration status, not experimental results, release eligibility, default model, or the preview-only gallery boundary.

Treat ranking, replay diagnosis, showcase components and their checks as completed, merged foundations. Do not schedule their implementation or historical experiment runs again. Start subsequent implementation from an up-to-date main containing both merges, carry this proposal into that branch, and inspect any later changes before editing. The local checkout used for this planning audit remains on `codex/showcase-four-models`; its cached `origin/main` is stale and was not used to infer remote merge status. The GitHub connector and command-line remote lookup could not authenticate; the signed-in GitHub browser provided the verification above. No branch switch or remote synchronization was performed.

Merging source does not verify a Railway deployment or activate the experimental gallery. Runtime deployment status remains a separate check if a later release requires it. The two-page consolidation and policy/value study remain future work.

## Fit with the completed project

The move-ranking evaluation and showcase release decision are complete. Its preselected seed-42 model scored 48.75% on 200 fresh final opening pairs, with paired 95% interval 45.75–51.625%. The newer supervised initializer scored 46.25% against the released default and failed the explicit release gate. The four-model gallery is therefore a local preview; production models were preserved. Sources: `docs/experiments/move-ranking/README.md`, `final-audit.json`, and `docs/four-model-showcase.md`.

The merged application already has Play, Compare, and Research views, four preview model routes, checkpoint identity checks, and a historical HTML/PDF report. Several subsequent study reports exist under `docs/experiments/`, but the public Research view is centered on the original local study rather than indexing the complete research sequence. Consolidation should refactor these components and assemble the existing reports rather than build a second website or duplicate their content.

The current release manifest defaults to the **older local-study RL checkpoint**, selected on development matches. It is not the later feature-v3 RL checkpoint that regressed, nor the anchored replay diagnostic. Preserve those distinctions in every page and report. Nothing in this proposal establishes that the newer supervised initializer should replace the current release.

The earlier fixed-trajectory diagnosis provides motivation: anchoring improved development scores relative to its matched replay reference. It was off-policy, used anchor coefficient **1.0**, and did not establish live self-play improvement. The proposed coefficient **0.1** is a new, tenfold weaker intervention fixed in advance, not the earlier diagnostic treatment or an empirically selected optimum.

## Research question and claim boundary

Test: **Does prediction anchoring improve a compact, supervised-initialized policy/value self-play system under the same training and search budgets?**

Use the description “AlphaZero-inspired policy/value self-play.” The original paper combines a policy/value network, MCTS, search-policy targets and terminal outcomes: https://arxiv.org/abs/1712.01815. Our supervised initialization, compact MLP, engineered features, batch-frozen actors and CPU budget are explicit adaptations. This is not tabula-rasa training or a reproduction of the original compute scale.

There are exactly two training treatments. Initializer and released-engine comparisons are evaluation controls, not additional training arms. Because each treatment generates its own subsequent games, this estimates the anchor's effect on the complete learning process. It does not isolate its gradient effect on identical trajectories; the completed replay diagnosis addresses that narrower question.

## Architecture and reuse

Reuse `python-chess`, feature v3, history serialization, player-relative move coordinates including underpromotions, provenance helpers, opening audits, paired statistics, and reporting conventions.

Create a separately versioned policy/value checkpoint. Copy the exact seed-42 combined supervised checkpoint used by the prior v3 studies: `runs/fixed-data-improvements/combined-42/best.npz`, verified by hash. This is a research initializer, not a newly approved deployment. Use its existing 853 → 128 → 64 trunk and scalar tanh value head. Add a legal-move policy scorer using the existing board-context and move-coordinate design, with fresh policy parameters. Do not import trained ranker weights or teacher move labels.

Before training, exported value predictions must match the initializer. Both treatments and all three run seeds start from the same complete parameter initialization, including the same random policy head. Seeds 42/43/44 vary self-play and training randomness; the report must state that policy initialization itself was held fixed. Paired treatments share corresponding RNG stream seeds and opening schedules, without pretending their trajectories remain identical after divergence.

Keep inference in NumPy and training in PyTorch. Add explicit policy/value and move-schema validation and PyTorch/NumPy parity checks. Never guess a checkpoint type from its filename or load it as the existing scalar model.

Add a research search strategy rather than changing `engine.Search`. Use the same PUCT implementation for generation, development and final evaluation. Generalize match orchestration through an explicit search-player adapter: the current evaluator loader alone cannot represent a different search algorithm. Existing alpha-beta callers retain their defaults and budget accounting.

Before implementation, amend `AGENTS.md` and `docs/decisions.md` together: permit MCTS only in this local research path; qualify the shared-engine rule by named search strategy; retain alpha-beta for the released API; distinguish this policy/value objective from existing outcome-only RL. Website consolidation also supersedes the four-dedicated-page presentation rule. No custom move generator, online training, production PyTorch, hosting change or model replacement is included.

## PUCT and game semantics

Use legal-move-normalized priors and a documented PUCT score such as `Q(s,a) + c_puct P(s,a) sqrt(sum_b N(s,b)) / (1 + N(s,a))`. Store edge Q from the selecting player's perspective, with exactly one sign reversal per traversed ply. Define first expansion, unvisited-edge values and deterministic tie breaking explicitly.

MCTS terminal values are in {-1, 0, +1}; do not reuse alpha-beta's out-of-range mate sentinel. Preserve automatic draw claims, halfmove state and complete history. Begin without transposition merging, cross-request caches or tree reuse between moves. Each request starts a fresh tree; this simplifies history and model-identity correctness and must be included in runtime estimates.

Record the root visit distribution before action sampling. Keep target temperature separate from the move-sampling temperature. Add root noise and stochastic action selection only during training; evaluation disables noise and uses deterministic highest-visit choice. Freeze the PUCT constant, first-play rule, noise settings, temperature schedule, target transformation and tie rules in configuration.

A simulation limit counts completed simulations under a precise definition. Also record expanded leaves, network calls, terminal hits, traversed edges and elapsed time. A simulation is not an alpha-beta node, and these counters must never be presented as interchangeable compute units.

## Two treatments and data weighting

For self-play state s, visit target pi, completed outcome z, predicted policy p and value v:

- Baseline: `L0 = CE(pi, p) + (v - z)^2`.
- Anchored: `L1 = L0 + 0.1 * (v - stop_gradient(v_initial(s)))^2`.

Apply any optimizer weight decay identically in both treatments. Average the entire per-position objective with **uniform game weighting**, using a fixed number of sampled positions per completed game. The initializer is a detached, permanently frozen copy, with an independently checked identity. The shared trunk can change under policy gradients; the anchor constrains value predictions rather than freezing representation parameters.

Generate each batch with a frozen actor, then update and use the new actor for the next batch. Use current-batch data only for this first study; do not add replay-window tuning. Retain all completed training PGNs and visits needed to audit targets. Exclude supplied opening-prefix moves from training examples. A truncated game contributes neither value nor policy targets to either arm. Record its visits and termination cause for audit.

Match attempted games, opening schedules, update counts, sampling rules, optimizer, learning rate and search settings between arms. A proposed run is ten batches of 100 **attempted** games per arm and seed, for 6,000 attempted self-play games total. Completed-game counts may differ; report this rather than silently generating extra games in one arm. Freeze the insufficient-completed-data rule before running and never invent targets or silently skip a failed batch.

## Pilot, configuration freeze and resources

Run correctness tests and a tiny disposable integration study first. Then benchmark an independent pilot seed and disjoint pilot openings using the common initial model. A proposed simulation ladder is 128/256/512, with 20 attempted games at each budget. This pilot can select a common feasible search budget using a predeclared rule; it cannot compare anchor coefficients, rank treatments or select a favorable playing result.

Freeze the pilot decision rule before observing its measurements. Include completed-game rate, root-visit concentration/coverage, search depth distribution, memory, latency and throughput. A proposed minimum is 80% completed games, with root coverage and degenerate-policy failure criteria specified in the pilot protocol. Completion alone does not establish good self-play. If no budget qualifies, record a blocked/failed pilot and write a separate prospective amendment; do not lower thresholds after looking at treatment results.

After the pilot, freeze the scientific configuration and opening manifests before the main runs. Include every schedule, sampling count, loss scale, update count, development checkpoint, selection tie rule and final match budget. No outcome-driven hyperparameter expansion.

Report a measured full-run forecast **before launching**: all six arms, updates, development comparisons, final comparisons, audit/export overhead, disk use and a conservative throughput margin. Earlier alpha-beta timings cannot estimate PUCT throughput reliably. Run sequentially on local CPU with bounded threads and memory on the 8 GB machine. There is no artificial local wall-clock cap; do track elapsed time and remaining forecast. Paid compute remains outside this proposal.

## Evaluation and selection

Reserve fresh source-game-separated training starts, development starts and final starts before main training. Final openings exclude the corpus and every previously reserved study's source games and audited canonical positions, including move-ranking finals. Pilot data is separate. Reconstruct opening history rather than starting from FEN alone. Record naturally recurring positions encountered during play; the opening audit is not a claim that chess trajectories can never transpose.

Use the following proposed bounded schedule, subject to a prospective configuration freeze after the feasibility pilot:

| Stage | Protocol | Purpose |
|---|---|---|
| Development | Batches 5 and 10; 40 shared color-swapped pairs per checkpoint against the frozen initial policy/value MCTS player | Select each treatment/seed checkpoint by match score; earlier checkpoint wins ties |
| Primary final | Selected anchored versus selected baseline, 200 fresh pairs for each of seeds 42/43/44, equal simulation budgets | Measure the anchoring treatment effect |
| Secondary final control | Both selected treatments versus the frozen initial policy/value MCTS player, 40 pairs each per seed | Distinguish improvement over initialization from merely regressing less |
| Secondary final deployment reference | Both selected treatments versus the unchanged released default, 40 pairs each per seed, equal time per move | Put practical strength/latency in the project's existing context |

This is 960 development games and 2,160 final games, in addition to self-play and pilot work. Give the secondary controls separate final opening blocks. Freeze both simulation and time budgets before evaluation. Charge all feature encoding, policy/value inference and search overhead to equal-time comparisons, including quiescence on the alpha-beta side. Deterministic resume applies to simulation-budget runs; wall-clock-limited match trajectories cannot promise exact replay.

Complete and hash all six selection records before starting any final matches. Do not choose a favorable seed. Seed 42 remains the predeclared illustration for reports; aggregate and individual results include all three seeds. The final checkpoint from each run is also retained for learning diagnostics, regardless of selection.

Use win=1, draw=0.5, loss=0 for playing score. Bootstrap complete color-swapped opening pairs jointly across the fixed seeds, and report seed SD separately. The aggregate interval is conditional on these three trained runs. Predeclare the primary comparison; label all others secondary. Report capped-game counts, uncapped-pair analysis and conservative cap-score bounds, without calling a cap a rule-based draw.

A proposed positive-result criterion is primary aggregate lower 95% bound above 50% and point scores above 50% in at least two seeds. Otherwise report uncertainty or regression as appropriate. This is a study conclusion rule, not a release gate. Smaller value drift alone does not demonstrate stronger play. No Elo, calibrated win probabilities, or attribution of an MCTS-versus-alpha-beta difference solely to anchoring.

## Verification and completion

Test PUCT mechanics, masked legal probabilities, visits, deterministic ties, signs, terminal cases, claimable draws with history, castling, en passant, and every promotion. Test initial value parity, policy/value export parity, detached anchor arithmetic/gradients, zero baseline anchor, unchanged frozen initializer, and actor freezing.

Test exact CPU resume at generation and update boundaries, including independent RNG streams, partial-game state or deterministic replay, optimizer state, sampled visits and file identities. Reject changed scientific configs, opening manifests, schemas and model hashes. Audit game weighting, truncation exclusion and split provenance.

Run a tiny end-to-end pipeline through both treatments, all seed paths, development selection, final-pair aggregation and report generation using disposable fixtures. Re-run existing engine/API/release checks; browser checks cover the website changes separately. Do not rerun historical training or select anything using historical final results.

The study is complete only when all six runs, selection records, planned final comparisons, integrity audits, compute accounting and report are present. An incomplete pilot or partial run is published as such. Negative and inconclusive outcomes remain valid completed results when the protocol finishes.

## Website: two main destinations

Recommend **Demo** and **Research** as the only top-level navigation items. Demo contains Play and Compare as modes sharing one board, complete history, model selection and inline explanations. Research is a browsable study index, with stable deep links to individual reports. Detailed report pages do not need their own top-level navigation items.

Reuse the current board, inspector, mathematical explanations, comparison cards, wake-up/error states and checkpoint-matching guards from the merged `web/src/App.tsx` and `web/src/ModelGuide.tsx`. Consolidate the existing `web/tests/showcase.spec.ts` coverage alongside the routes; retain the identity checks and feature-specific explanations. Avoid a separate playable page for every seed, objective variant or unsuccessful experiment. Do not introduce MCTS into the public API merely to illustrate the new study; saved games, visits and measured figures can explain it in the report.

Use two independent labels: **availability** (released / local preview / research only) and **evidence** (improvement / regression / inconclusive / pilot / diagnostic). “Released” does not mean “proven stronger.” An inconclusive result does not automatically make an existing release unusable.

| Approach | Demo treatment | Research treatment |
|---|---|---|
| Handcrafted baseline | Available as an interpretable comparison | Brief method explanation |
| Released supervised evaluator | Playable, using its actual released schema and evidence | Original study plus clearly separated later diagnostics |
| Older released self-play checkpoint | Preserve manifest-selected default initially; label it “Earlier released self-play” | Its own original-study evidence, distinct from later v3 regression |
| Newer supervised and move-guided candidates | Local preview only; no promotion after the failed gate | Completed move-ranking study and release decision |
| Later live v3 RL | Research entry and saved games, outside the main selector | Completed negative live-self-play study |
| Anchored fixed-trajectory replay | Research only | Explicit off-policy development diagnostic |
| New policy/value baseline and anchored variant | Research only | One study containing both treatments and all seeds |

Put the earlier released RL option in an expandable historical-model group if a shorter selector helps, while retaining the current manifest default during initial consolidation. Replacing it or changing the demo default is a separate explicit release/presentation decision. Never apply the later v3 result to this older checkpoint or silently substitute the newer 853-feature model for the released one.

Keep all research outcomes discoverable. Use concise study cards with question, method, actual evidence level, result and practical takeaway. Group the sequence as original study, supervised diagnostics/representation/averaging, live RL regression, replay diagnosis, move ranking, and the proposed policy/value follow-up. Existing reports become detail pages; maintain the historical report links and downloads.

Each detailed report includes the frozen protocol, exact model identities, data/opening lineage, seed-level results, paired uncertainty, compute cost, limitations, and genuine figures. No global “best model” chart across incompatible opponents, checkpoints or budgets. Show the new study as “planned” until it has evidence; do not populate empty plots with illustrative results.

Generate the study index and model presentation metadata from a small explicit catalog, with report paths, component identities, evidence level, result status and availability. Architecture descriptions must follow the exact checkpoint: the current v3 preview explanation cannot describe a different released feature schema. Keep trained seeds inside their study rather than creating separate public model names.

Separate **report publication** from **checkpoint publication**. The failed showcase model gate must not prevent allowlisted negative/inconclusive reports from being published. Keep the existing model release gate and checksums intact; sanitize public reports and copy only allowlisted assets, never raw run directories or private paths.

Provide stable `/research` and study deep links. Redirect previous `/lab/*` links to the corresponding explanation or study while preserving valid position/history state. For an unreleased checkpoint, show its study/preview status rather than silently redirecting it to a different model. Verify direct navigation, refresh, back/forward, mobile layouts, promotions, comparison, offline reports, service wake-up, model mismatch and old report downloads.

## Delivery order

1. Use current main containing merged PRs #9 and #10 as the next implementation baseline, preserving all completed studies and the current release. The move-ranking evaluation, showcase decision and PR merges are complete prerequisites; do not wait for or repeat them. Recheck subsequent main changes when implementation starts.
2. Consolidate Demo/Research using the merged UI and publish the existing research archive independently of model promotion. Update the presentation contract and retain preview components for reuse. Run the relevant existing checks and add coverage for the new navigation and research index; the six passing PR checks establish the previous baseline, not validation of this future refactor.
3. Before implementing MCTS, record the narrow architectural exception and preregister correctness/pilot rules. Add the research model, search adapter, training path and tests in a separate batch.
4. Run the disposable integration study and feasibility pilot; report measured runtime and freeze the main protocol.
5. Complete the six runs, development selections, fresh final comparisons and audited report. Add the study to Research regardless of direction.
6. Consider deployment eligibility only in a later explicit release decision. Private visibility, user-managed pushes and paid-resource restrictions remain in force.
