# ChessEngine project contract

## Purpose and workflow
- Demonstrate supervised learning, self-play reinforcement learning, careful experiments, and a polished chess demo for master's applications. Playing strength and positive RL results are not promised.
- The repository stays private. The user handles remote pushes; never push or change visibility. Prepare three substantial review batches: foundation/supervised, RL/evidence, showcase/release. Use focused local commits where Git permissions allow. Do not block independent local work on a push.
- Implement autonomously, explain mathematics and evidence at milestones, and minimize questions. Ask only for consequential scope changes, identity/account access, or costs outside authorization.
- Total paid spend ceiling is C$50 equivalent; reserve 20% contingency. Prefer free Railway credits; paid hosting requires explicit approval. Local Apple Silicon/8 GB development, optional paid Colab. Never assume free Colab permits chess training. No paid resources are provisioned implicitly.

## Architecture
- `src/chessengine`: Python engine, features, NumPy inference, data, PyTorch training, RL, evaluation, reports, FastAPI, CLI. `web`: React/TypeScript UI. `configs`: experimental settings. `tests`: correctness and research-integrity checks. `docs`: decisions, mathematics, operations, PR handoffs.
- python-chess owns rules and PGN; our negamax/alpha-beta engine chooses moves. Stockfish only labels data or acts as an explicitly named reference opponent. Never silently substitute Stockfish for our demo engine.
- One shared engine for self-play, matches, and API. Runtime inference uses exported NumPy MLP weights; PyTorch is training-only. Feature/checkpoint schema versions must be validated, not guessed.
- Side-to-move value in [-1,1]; terminal mate search scores are outside that range. Negate child scores exactly once per ply. Feature v1 uses absolute square/color coordinates; experimental v2 uses player-relative coordinates and v3 adds normalized material summaries. All preserve castling, en passant, halfmove clock and original-history repetition indicators. Labels retain their side-to-move sign. Checkpoints select the encoder explicitly; RL supports v1 and v3 explicitly; v2 remains unsupported.
- Preserve complete move history. Draw claims are automatically taken consistently across API/self-play/evaluation; document this convention. TT keys must distinguish reversible history/halfmove state. Never reuse TT values across models or games.
- No MCTS, policy network, custom move generator, UCI engine protocol, user accounts, database, or online training in v1.

## Research invariants
- Split source games before position sampling, deduplicate positions across splits, exclude reserved evaluation positions, and preserve data/source manifests. Do not train on validation/test labels or select checkpoints using final-test matches.
- Teacher targets are (wins-losses)/total from Stockfish WDL estimates, from side-to-move perspective. Pin teacher executable hash/version and node/hash/thread settings.
- RL is Monte Carlo outcome value learning. Actor frozen per batch; no teacher labels in the primary RL treatment. Outcome targets are +1/0/-1 for the player to move. Uniform game weighting. Truncated self-play games do not get fabricated outcomes or enter training.
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
- Production requires both released neural models and valid checksums; missing artifacts fail startup. One API replica, one Uvicorn worker, no persistent volume or training dependencies.
- Dockerfiles and dashboard instructions define deployment. No paid activation or automatic PR environments. Track hosting costs within the C$50 ceiling.

## Supervised diagnostics
- Diagnostics use training/validation records only; old test results cannot select candidates. Keep released checkpoints and public website assets unchanged.
- Freeze the diagnostic configuration before running experiments. Conditional regularization follows its predefined gap rule; do not expand the search after viewing outcomes.
- Report frozen-checkpoint training error separately from online training loss. Preserve per-seed evidence and whole-game bootstrap uncertainty; development improvements do not establish stronger play.

## Averaging and RL follow-up
- Averaging adoption requires at least 2% lower mean MSE and two improving seeds. The stronger RL study uses seed 42 of the qualifying supervised configuration as the common initialization; never choose the best seed or initialize RL from the prediction ensemble. Record the actual selection and results in the study evidence.
- RL improvement uses a quality-gated 128/256/512-node pilot and a persisted 12-hour local allowance. Do not relax quality thresholds or extend compute silently.
- Fresh final openings exclude corpus and old reserved source games/canonical positions. Complete all development selections before any new final matches. Teacher-validation error is secondary and cannot select RL checkpoints.
- No new supervised search precedes this RL study. Released models remain unchanged.
