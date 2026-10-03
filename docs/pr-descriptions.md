# Prepared pull request descriptions

These are local handoff drafts. No remote PR has been created and nothing has been pushed.

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
