# ChessEngine project contract

## Purpose and workflow
- Demonstrate supervised learning, self-play reinforcement learning, careful experiments, and a polished chess demo for master's applications. Playing strength and positive RL results are not promised.
- The repository stays private. The user handles remote pushes; never push or change visibility. Prepare three substantial review batches: foundation/supervised, RL/evidence, showcase/release. Use focused local commits where Git permissions allow. Do not block independent local work on a push.
- Implement autonomously, explain mathematics and evidence at milestones, and minimize questions. Ask only for consequential scope changes, identity/account access, or costs outside authorization.
- Total paid spend ceiling is C$50 equivalent; reserve 20% contingency. Free hosting, local Apple Silicon/8 GB development, optional paid Colab. Never assume free Colab permits chess training. No paid resources are provisioned implicitly.

## Architecture
- `src/chessengine`: Python engine, features, NumPy inference, data, PyTorch training, RL, evaluation, reports, FastAPI, CLI. `web`: React/TypeScript UI. `configs`: experimental settings. `tests`: correctness and research-integrity checks. `docs`: decisions, mathematics, operations, PR handoffs.
- python-chess owns rules and PGN; our negamax/alpha-beta engine chooses moves. Stockfish only labels data or acts as an explicitly named reference opponent. Never silently substitute Stockfish for our demo engine.
- One shared engine for self-play, matches, and API. Runtime inference uses exported NumPy MLP weights; PyTorch is training-only. Feature/checkpoint schema versions must be validated, not guessed.
- Side-to-move value in [-1,1]; terminal mate search scores are outside that range. Negate child scores exactly once per ply. Network inputs use absolute square/color coordinates, plus turn, castling, en passant, halfmove clock and repetition indicators.
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
- Host a static public demo/report and a bounded FastAPI engine on Render, with explicit wake-up/error states. Private source stays private. Only allowlisted assets enter static deployment.
- Test chess integration/search signs/budgets, history-sensitive draws, feature/target perspective, leakage, overfit/resume/export parity, RL actor freezing/outcomes/truncation, paired statistics, and browser flows.
- Standard commands: `.venv/bin/python -m pytest`, `.venv/bin/python -m chessengine --help`, `npm --prefix web run build`, `npm --prefix web run test:e2e`. Full training is not CI.
- Do not claim deployment, full experiments, or a verified PDF unless actually completed. Preserve all meaningful failures in the report and provide reproducible follow-up commands for work requiring external account access.
