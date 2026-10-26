# Evolutionary study and current demoThe active Demo is Classical, Supervised and Evolutionary, with Supervised as the
default. Evolutionary stays disabled until its frozen quality and strength gates
pass. Self-play is retired from live play; its reports and recorded games remain
in **Research → Self-play: results and lessons**. Original inconclusive evidence
and the later regression remain separate historical studies.

The [evolutionary protocol](docs/evolutionary-preregistration.md) specifies a
teacher-free, randomly initialized 324-parameter evaluator optimized by CMA-ES.
Run `.venv/bin/python -m chessengine evolutionary --stage all`; the runner reports
measured forecasts and stops without full training if pilot gates fail. Paid
compute, automatic pushes and deployment are not authorized. Older release and
four-model workflow descriptions below are historical; the v3 active manifest is
the authority for playable models.

# ChessEngine Lab

**Can a chess evaluator improve by learning from its own games?**

A reproducible research project combining supervised value learning, Monte Carlo self-play reinforcement learning, controlled matches, and a playable web demo. The Python engine is shared by training, evaluation, and deployment. Stockfish provides teacher labels; it does not play moves on behalf of our neural models.

> Evidence status is recorded in the generated report. Pilot results validate the pipeline and must not be described as the final multi-seed study. Source remains private; remote pushes are user-managed.

## Run the demo locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-train.txt
pip install -e '.[dev]' --no-build-isolation
python -m chessengine serve
```

In a second terminal:

```bash
cd web
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. The backend listens on port 8000. Supplied release exports are discovered in `artifacts/models/manifest.json`; without them, the app explicitly offers only the handcrafted baseline. It never invents a trained model or silently substitutes another engine.

## What is implemented

- Iterative-deepening negamax, alpha-beta pruning, bounded quiescence, history-sensitive draw handling, and node/time budgets.
- Streamed PGN sampling, game-level splits, duplicate filtering, reserved opening exclusion, and pinned Stockfish WDL labeling.
- PyTorch MLP regression, early stopping, exact CPU resume, schema-checked NumPy inference exports.
- Frozen-actor self-play, completed-outcome returns, equal game weighting, resumable RL, and explicit truncation records.
- Frozen match protocols, color-swapped openings, paired-bootstrap intervals, development-only checkpoint selection, and runtime/phase diagnostics.
- React Play/Compare/Research views, PGN export/replay, accessible click/drag board, promotion, undo, wake-up and failure states.
- HTML/PDF reporting, production Railway containers, and Python/browser CI.

## Reproduce an experiment

Install a local Stockfish executable from the official project. Do not replace its version inside an existing study. The data manifest records executable SHA256, reported version, nodes, threads, and hash settings.

```bash
python scripts/make_openings.py data/lichess-2013-01.pgn.zst configs/openings.json
python -m chessengine prepare data/lichess-2013-01.pgn.zst data/positions.jsonl --limit 100000
python -m chessengine label data/positions.jsonl data/labeled.jsonl --stockfish /absolute/path/to/stockfish --nodes 10000
python -m chessengine train data/labeled.jsonl runs/supervised --config configs/supervised.json
python -m chessengine train data/labeled.jsonl runs/large --config configs/supervised-large.json
```

Choose the supervised architecture using validation loss and measured inference cost, **before self-play**. Then:

```bash
python -m chessengine selfplay runs/supervised/best.npz runs/rl-42 --config configs/rl.json
python -m chessengine freeze runs/rl-42/last.npz runs/supervised/best.npz runs/development-protocol.json --split development --config configs/pilot-evaluation.json
python -m chessengine evaluate runs/development-protocol.json runs/development
python -m chessengine select runs/selection.json runs/development
python -m chessengine freeze runs/rl-42/last.npz runs/supervised/best.npz runs/final-protocol.json --config configs/evaluation.json
python -m chessengine evaluate runs/final-protocol.json runs/final
```

Use different directories for different experiments. `--resume` resumes training without changing the dataset or identity settings; epochs/batches may increase. Final protocols cannot be overwritten. Full commands and execution order are in [the study guide](docs/study.md).

## Verify

```bash
python -m pytest -q
ruff check src tests scripts
npm --prefix web run build
cd web && npx playwright install chromium && npm run test:e2e
```

Tests include tiny CPU training runs; they are not scientific experiment results.

## Read next

- [Project contract](AGENTS.md): durable decisions and contributor instructions.
- [Mathematical walkthrough](docs/mathematics.md): value functions, search, returns, training and uncertainty.
- [Study protocol](docs/study.md): reproducibility, data, checkpoints, costs and limitations.
- [Supervised diagnostics](docs/experiments/supervised-diagnostics/index.html): frozen-model audit and 21 completed development runs; [reproduction guide](docs/supervised-diagnostics.md).
- [200k data extension](docs/experiments/supervised-200k/index.html): three expanded-data fits reduced mean validation MSE by 9.2%; [protocol and reproduction](docs/expansion-200k.md).
- [Deployment](docs/deployment.md): private GitHub access and public Railway services.
- [GitHub handoff](docs/github.md): three substantial review batches and push instructions.
- [Demo walkthrough](docs/demo.md): a short project presentation.

The project is provided under GPL-3.0-or-later, consistent with the chosen chess library. Third-party sources and licensing are listed in [NOTICE](NOTICE). Publishing source or changing repository visibility remains the user's decision.

## First completed local study

The release includes a **bounded local study**: 100,000 teacher-labeled positions, three RL seeds with 60 self-play games each, and 400 final games per seed. Final RL scores were 50.00%, 49.25%, and 50.25%; all paired confidence intervals include 50%. This is **inconclusive evidence of improvement**. Shallow 48-node search limits the finding; it is not the larger target training schedule.

The Research tab contains all seeds, downloadable game records, and an HTML/PDF report. See [validation and limitations](docs/validation.md), [the preregistered local protocol](docs/local-study-preregistration.md), and [the prepared PR descriptions](docs/pr-descriptions.md). Actual paid compute cost: **C$0**.

The [fixed-data study protocol](docs/fixed-data-improvements.md) compares optimization and player-relative representations on the same 180,302 training positions. Its [separate comparison report](docs/experiments/fixed-data-improvements/index.html) records development evidence; these experimental models do not replace the demo checkpoints.

[Averaging study](docs/experiments/supervised-averaging/index.html): prediction averaging improves MSE at higher inference cost; weight averaging misses the predefined adoption threshold. [Protocol and RL handoff](docs/supervised-averaging.md).

[Stronger RL study](docs/experiments/rl-improvement/index.html): 3,000 self-play games and 1,200 final comparison games. RL scored 44.13% against its supervised initializer (95% paired interval 41.54–46.58%), supporting regression under this protocol. Released models remain unchanged. [Methods and limitations](docs/rl-improvement.md).

## Demo and research journal

The website has two main destinations: **Demo** and **Research**. Demo offers three released opponents in order: **Classical** (handcrafted evaluation), **Supervised** (learned Stockfish position assessments), and **Self-play** (the released outcome-learning experiment). All use the same alpha-beta search; playing does not train them. Play and Compare share the current position/history. Explanations cover evaluation, move choice and checkpoint-matched original-study evidence; the released self-play study is inconclusive. Research preserves every study, including later RL regression and the completed policy/value feasibility pilot with its full comparison deferred. The historical interactive charts and game viewer remain in an expandable original-study archive. `/lab/*` links resolve to the matching research entry or classical demo; they do not substitute an unreleased checkpoint for a released one. `VITE_SHOWCASE=1` still enables explicitly staged, identity-checked local preview routes with matching preview assets.

Publish only the explicit report allowlist, independently of any model-promotion gate:

```sh
.venv/bin/python -m chessengine.research_catalog
npm --prefix web run build
```

This does not copy checkpoints or change the released manifest/default. The newer move-guided candidate remains a local preview after its failed release gate. The older released self-play checkpoint is a distinct model from the later negative v3 RL study.

## Local policy/value research

**Release status: completed feasibility pilot; full comparison deferred.** The user requested an immediate release after the 559.6-hour forecast. All three pilot budgets passed (59 rule-completed games in 60 attempts). The long run is stopped. The anchoring comparison is untested; released models are unchanged. See `docs/policy-value-release.md` for the scope decision and audited report.

For the original future-study methodology, see `docs/policy-value-preregistration.md` and `configs/policy-value.json`. This adapted AlphaZero-inspired path uses the existing v3 supervised initializer, a fresh legal-move policy head, and history-preserving PUCT. Exactly two treatments compare visit-policy/outcome learning with a detached 0.1 prediction anchor. It never enters the production API automatically.

```sh
# Disposable timing benchmark; report the estimate before the pilot.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m chessengine.pv_study configs/policy-value.json --stage benchmark
# Freeze inputs/code and run the independent quality-gated pilot.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m chessengine.pv_study configs/policy-value.json --stage pilot
# Only after a passing pilot and communicating its measured full-study forecast:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m chessengine.pv_study configs/policy-value.json --stage all
# Regenerate the public research index/report from actual evidence.
.venv/bin/python -m chessengine.research_catalog
```

All stages resume from verified inputs and completed games. CPU updates resume at minibatch boundaries; interrupted simulation-budget games replay from deterministic seeds. Wall-clock match trajectories are not promised bitwise reproducible. Do not change scientific code/configuration to bypass a frozen identity mismatch. Historical studies and negative results remain intact; no paid resources or remote pushes are part of this workflow.

The following continuation is retained for a separately authorized future study; it is stopped and outside this release. A closure marker prevents resuming the closed run. Do not remove that marker or launch the full-run commands above without new user authorization:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m chessengine.pv_continue configs/policy-value.json --wait-pilot
```

Do not run this alongside another continuation or a raw main-study process. It checks the recorded runtime environment, preserves failures, handles interruption through existing resume state, and checkpoints elapsed time every 30 seconds. Reports state the actual evidence level; no background process promotes a model or deploys the website.
