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
