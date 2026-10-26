# ChessEngine project contract

## Purpose and workflow
- Demonstrate supervised learning, evolutionary learning, archived self-play research, careful experiments, and a polished chess demo for master's applications. Playing strength and positive RL results are not promised.
- The repository stays private. The user handles remote pushes; never push or change visibility. Prepare three substantial review batches: foundation/supervised, RL/evidence, showcase/release. Use focused local commits where Git permissions allow. Do not block independent local work on a push.
- Implement autonomously, explain mathematics and evidence at milestones, and minimize questions. Ask only for consequential scope changes, identity/account access, or costs outside authorization.
- Total paid spend ceiling is C$50 equivalent; reserve 20% contingency. Prefer free Railway credits; paid hosting requires explicit approval. Local Apple Silicon/8 GB development, optional paid Colab. Never assume free Colab permits chess training. No paid resources are provisioned implicitly.

## Architecture
- `src/chessengine`: Python engine, features, NumPy inference, data, PyTorch training, RL, evaluation, reports, FastAPI, CLI. `web`: React/TypeScript UI. `configs`: experimental settings. `tests`: correctness and research-integrity checks. `docs`: decisions, mathematics, operations, PR handoffs.
- python-chess owns rules and PGN; our negamax/alpha-beta engine chooses released demo moves. The approved local policy/value study uses a separately named PUCT strategy. Stockfish only labels data or acts as an explicitly named reference opponent. Never silently substitute Stockfish for our demo engine.
- Share each named search strategy across its self-play and matches; the released API stays alpha-beta. Runtime inference uses exported NumPy MLP weights; PyTorch is training-only. Feature/checkpoint schema versions must be validated, not guessed.
- Side-to-move value in [-1,1]; terminal mate search scores are outside that range. Negate child scores exactly once per ply. Feature v1 uses absolute square/color coordinates; experimental v2 uses player-relative coordinates and v3 adds normalized material summaries. All preserve castling, en passant, halfmove clock and original-history repetition indicators. Labels retain their side-to-move sign. Checkpoints select the encoder explicitly; RL supports v1 and v3 explicitly; v2 remains unsupported.
- Preserve complete move history. Draw claims are automatically taken consistently across API/self-play/evaluation; document this convention. TT keys must distinguish reversible history/halfmove state. Never reuse TT values across models or games.
- MCTS is permitted only for the approved local policy/value experiment, never implicitly in production. No custom move generator, UCI engine protocol, user accounts, database, or online training. The approved showcase extension permits a supervised legal-move ranker for root ordering only; it does not replace the value evaluator or prune legal moves.

## Research invariants
- Split source games before position sampling, deduplicate positions across splits, exclude reserved evaluation positions, and preserve data/source manifests. Do not train on validation/test labels or select checkpoints using final-test matches.
- Teacher targets are (wins-losses)/total from Stockfish WDL estimates, from side-to-move perspective. Pin teacher executable hash/version and node/hash/thread settings.
- Existing RL is Monte Carlo outcome value learning. The separately named policy/value study adds MCTS visit-policy targets and compares zero anchoring with coefficient 0.1. Actor frozen per batch; no teacher labels in either treatment. Outcome targets are +1/0/-1 for the player to move. Uniform game weighting. Truncated self-play games do not get fabricated outcomes or enter training.
- Store seeds, configs, commit IDs, checkpoint/data hashes, optimizer/RNG state, generation/update times, and costs. CPU resume reproducibility is required; accelerator determinism is not assumed.
- Select on development matches, freeze final protocol, swap colors per opening, use equal node budgets including quiescence, clear game caches, bootstrap opening pairs, and distinguish seed variance from match uncertainty. Report truncation/adjudication sensitivity.
- Smoke/pilot evidence is never presented as the final study. Publish negative/inconclusive results. No unsupported Elo, calibrated win-probability claims, or invented graphs.
- Stable decisions live here; tunable values live in JSON configs. Record changes to invariants in `docs/decisions.md` and update this file together.

## Delivery and verification
- Host a static public demo/report and a bounded FastAPI engine on two Railway services (always-running Nginx website, serverless FastAPI engine), with explicit wake-up/error states. Private source stays private. Only allowlisted assets enter static deployment.
- Test chess integration/search signs/budgets, history-sensitive draws, feature/target perspective, leakage, overfit/resume/export parity, RL actor freezing/outcomes/truncation, paired statistics, and browser flows.
- Standard commands: `.venv/bin/python -m pytest`, `.venv/bin/python -m chessengine --help`, `npm --prefix web run build`, `npm --prefix web run test:e2e`. Full training is not CI.
- Do not claim deployment, full experiments, or a verified PDF unless actually completed. Preserve all meaningful failures in the report and provide reproducible follow-up commands for work requiring external account access.

## Executed study boundary
- The original target configs remain in `configs/study.json`. The first executed local study uses `configs/local-study.json`, with its rationale frozen in `docs/local-study-preregistration.md` before RL/final matches. Always name the actual evidence level; do not conflate these studies.
- Select the representative RL checkpoint and demo default using development results only. Publish all seed-level final scores, and keep seed variability separate from opening-pair uncertainty. The default stays supervised unless the selected RL model's development score exceeds 50%.

## Railway deployment contract
- A fourth deployment PR follows showcase. User-managed pushes and private visibility remain mandatory.
- Production requires a version-3 manifest declaring Supervised and any qualified Evolutionary model, with valid checksums; missing declared artifacts fail startup. One API replica, one Uvicorn worker, no persistent volume or training dependencies.
- Dockerfiles and dashboard instructions define deployment. No paid activation or automatic PR environments. Track hosting costs within the C$50 ceiling.

## Supervised diagnostics
- Diagnostics use training/validation records only; old test results cannot select candidates. Keep released checkpoints and public website assets unchanged.
- Freeze the diagnostic configuration before running experiments. Conditional regularization follows its predefined gap rule; do not expand the search after viewing outcomes.
- Report frozen-checkpoint training error separately from online training loss. Preserve per-seed evidence and whole-game bootstrap uncertainty; development improvements do not establish stronger play.

## Averaging and RL follow-up
- Averaging adoption requires at least 2% lower mean MSE and two improving seeds. The stronger RL study uses seed 42 of the qualifying supervised configuration as the common initialization; never choose the best seed or initialize RL from the prediction ensemble. Record the actual selection and results in the study evidence.
- RL improvement retains its frozen quality-gated pilot and experimental settings. On 2026-10-13 the user removed local CPU time caps: provide a runtime estimate before launching local compute and continue tracking elapsed time. Paid compute still requires authorization; do not relax scientific quality thresholds.
- Fresh final openings exclude corpus and old reserved source games/canonical positions. Complete all development selections before any new final matches. Teacher-validation error is secondary and cannot select RL checkpoints.
- No new supervised search precedes this RL study. Released models remain unchanged.

## RL regression diagnosis
- Preserve the completed negative RL study. Fixed-trajectory replay is explicitly an off-policy development diagnostic, not new on-policy self-play or fresh final evidence.
- Prediction anchoring is a separately named treatment toward a detached frozen initializer; never describe it as the original pure outcome-only treatment. Do not silently add teacher labels.
- Keep treatments, selection screens and replay sampling in versioned configs/protocols. Candidates require subsequent live self-play and fresh confirmatory evaluation before a release decision.

## Four-model showcase and move ranking
- Present Demo and Research as the two main destinations. Reuse the four approach explanations in context; unreleased candidates and inconclusive/negative studies remain discoverable in Research, not separate main playable pages. Do not expose each seed as a separate model or label offline replay as live RL.
- Ranker inputs use feature v3 and player-relative move coordinates including promotions. Teacher move labels are separate from scalar value labels. Freeze whole-game splits and fresh final openings before training.
- Learned ordering applies only at the search root, computed once per request; previous-iteration principal variation stays first. Time accounting includes ranking inference. Existing behavior remains identical when no ranker is supplied.
- Production component identities must match the evidence shown on each page. Publish new checkpoints only through an explicit release manifest and validation gate; preserve the old release if the supervised candidate fails its development gate.

## Bounded policy/value follow-up
- The user authorized implementation on 2026-10-15. Use the seed-42 combined v3 value initializer, one fixed fresh policy initialization, paired run seeds 42/43/44, and exactly two treatments. Never import trained ranker weights or teacher labels into this study.
- Freeze pilot criteria before pilot measurements and the main protocol before training. Report a measured forecast before full local compute; no local CPU time cap, no paid compute authorization.
- Preserve history and automatic draw claims. MCTS terminal values are -1/0/+1; rebuild each tree per move and do not merge transpositions. Count simulations separately from alpha-beta nodes.
- Select all six checkpoints on development matches before fresh final comparisons. Publish complete seed-level evidence, truncation sensitivity and negative/inconclusive outcomes. Research report publication is independent of model promotion; released checkpoints and default remain unchanged.

## Immediate feasibility release (2026-10-15)
- The user explicitly reduced scope to an immediately pushable product after the 559.6-hour main-study forecast. The release includes the consolidated website, existing released models, tested policy/value research infrastructure and the completed 60-attempt feasibility pilot. The full baseline-versus-anchor comparison is deferred, not completed or a release blocker.
- Stop the long run gracefully and preserve its partial generation and frozen original protocol. Record the scope amendment and independent pilot audit in `docs/experiments/policy-value/release-scope.json`; do not rewrite the preregistration or call pilot results strength evidence. No model promotion or new deployment is authorized.
- No study continuation runs for this release. A future full study needs new explicit user authorization; preserve its original quality and evidence requirements. Commit the verified final release assets locally and hand the branch to the user for pushing.

## Three released playable approaches (2026-10-25)
- The public Demo presents Classical (`handcrafted`), Supervised (`supervised`), and Self-play (`rl`) in that order, using existing released evaluators and shared alpha-beta search. Keep the manifest-selected default and model artifacts unchanged.
- Explain Stockfish position assessments separately from move imitation. Gameplay does not train models. Tie neural evidence to the original local-study release identities; unmatched identities show unavailable evidence.
- The released self-play evidence is inconclusive. Later negative RL, move ranking, and policy/value feasibility remain separate Research entries; preserve optional four-model local previews. No new training or model promotion is implied.

## Evolutionary engine and self-play retirement (2026-10-26)
- This amendment supersedes playable self-play and its old default requirements. Public Demo cards are Classical, Supervised, Evolutionary; Supervised stays default. Evolutionary remains disabled until the preregistered gates pass. No RL play, live analysis, preview selection or public weight downloads; historical reports, replay and private checkpoints remain.
- Freeze original-study identities independently of the active manifest. Do not relabel original inconclusive evidence as the later regression.
- Train every one of 324 evaluation weights randomly with CMA-ES tournaments and compact versioned features. No fixed material score, imported weights, teacher labels, gradients, learned ordering or production MCTS. This is a hybrid with human-designed input features, not invented research or tabula-rasa rule discovery.
- Use the fixed protocol in docs/evolutionary-preregistration.md and configs/evolutionary.json. Freeze scientific sources before measurement. Pilot failure closes the study without changing architecture, thresholds or seeds. Full local computation is authorized after passing pilot gates, with measured forecasts, elapsed-time accounting, two workers maximum and no CPU wall-time cap. No paid compute is authorized.
- Evolutionary tournaments use complete legal histories and automatic draws. Capped tournament matches retain '*' rule outcomes and a separately named neutral fitness adjudication; no position-level outcome labels are generated. This does not change the exclusion of truncated games from historical Monte Carlo RL training.
- Select all checkpoints on development before qualification/finals; representative seed 42 is fixed. Fresh final confirmation can only block promotion, never choose another checkpoint. Gates use conservative cap sensitivity and require beating Classical and near-parity with Supervised. All seed outcomes are published.
- Runtime uses the separately versioned NumPy evolutionary evaluator and unchanged shared alpha-beta. Active-release manifest v3 declares required model IDs; checksums/kinds/schema are validated. Play and Compare use 10,000 nodes/1,500 ms/qdepth four for every released approach.
- Implement in local review batches. No remote push, deployment, paid resources or restart of the closed policy/value experiment.
