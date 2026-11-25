# Live search budget: 30,000 nodes / 3 seconds

Classical and Augmented Supervised use up to 30,000 nodes or 3,000 ms per move,
whichever comes first. Quiescence nodes count toward that allowance, with qdepth
four. The same defaults apply to position inspection and current-position
comparison. Stockfish uses a separate 30,000-node / 3,000-ms reference search;
Direct-Move CNN continues to choose a legal move without search.

For casual individual play, a roughly three-second thinking pause is reasonable.
Network overhead and a sleeping server's wake-up add to that pause. The model
search and Stockfish reference run sequentially, so a comparison can spend about
six seconds searching, plus process and request overhead. Time checks happen
between nodes; these are search allowances, not hard HTTP response deadlines.

Time-limited moves may consume approximately twice the search CPU time of the
old 1,500-ms default. Model weights are unchanged; the temporary search table may
grow. One API search runs at a time. Three-second searches imply at most about
20 completed searches per minute before overhead; concurrent callers receive
HTTP 429 instead of joining a queue. This is a capacity estimate, not a hosted
load test or billing estimate. The change suits a small portfolio demo, while
simultaneous visitors can still encounter the busy state.

## Local responsiveness check

Raw measurements and full position histories are in
[`experiments/live-search-budget-20261125.json`](experiments/live-search-budget-20261125.json).
The check uses the manifest-v7 augmented seed-42 evaluator, three illustrative
positions, two repetitions per budget, a fresh search per call and one BLAS
thread, matching API inference. Budget order reverses on the second repetition.
These positions are not a representative match sample or new strength evidence.

| Model | Position | Completed depth: old → new | Nodes at new budget |
|---|---|---:|---:|
| Classical | Starting position | 4 → 4 | 9,589–9,984 |
| Classical | Middlegame | 2 → 3 | 10,302–10,538 |
| Classical | Pawn ending | 6 → 6 | 18,390–20,670 |
| Augmented Supervised | Starting position | 3 → 3 | 5,480–6,790 |
| Augmented Supervised | Middlegame | 2 → 2–3 | 5,329–7,865 |
| Augmented Supervised | Pawn ending | 6 → 6 | 13,530–15,217 |

All twelve proposed-budget searches stopped near the time ceiling (3.000–3.006
seconds); none reached 30,000 nodes. The measured maximum node count was 20,670.
The augmented middlegame completed depth three once and depth two once, showing
that wall-clock limits and background load affect completed depth. These 24
searches took approximately 54 seconds in total.

Additional time may complete a deeper iteration or only search more of the next
one. The returned move comes from the last completed iteration. A higher ceiling
does not promise 30,000 visited nodes, a deeper completed search on every move,
better play or an Elo gain. Hosted performance has not been measured.

Reproduce the timing check from the repository root after installing the normal
runtime dependencies; allow about one minute locally:

```sh
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from pathlib import Path
import chess
from threadpoolctl import threadpool_limits
from chessengine.engine import Handcrafted, Search
from chessengine.model import NumpyEvaluator

spec = json.loads(Path('docs/experiments/live-search-budget-20261125.json').read_text())
with threadpool_limits(limits=spec['blas_threads']):
    for name, evaluator in [('handcrafted', Handcrafted()),
                            ('supervised-1m', NumpyEvaluator('artifacts/models/supervised-1m.npz'))]:
        Search(evaluator, nodes=64, time_ms=100).run(chess.Board())
        for position in spec['positions']:
            board = chess.Board(position['initial_fen'])
            for move in position['moves']:
                board.push_uci(move)
            for repeat in range(spec['repetitions']):
                for budget in spec['budgets'][::1 if repeat == 0 else -1]:
                    result = Search(evaluator, qdepth=spec['qdepth'], **budget).run(board)
                    print(name, position['name'], repeat, budget, result.to_dict())
PY
```

## Verification

- Python: the complete run recorded 240 passed / 1 skipped, with one transient
  installed-package metadata failure in `test_environment_change_blocks_cpu_resume`.
  Its focused rerun, including the complete continuation test file and the
  subprocess NumPy-only inference check, passed all five tests.
- Browser: 60 passed / 2 optional-preview tests skipped, including real API and
  Stockfish checks. The optional mocked manual-inspection case passed separately;
  both live tests passed again with exact 30,000-node / 3,000-ms assertions.
- Frontend production build, CLI help, Ruff, changed-frontend-file Prettier checks
  and `git diff --check` passed. Checkpoint SHA-256 values still match manifest v7.

Reproduce local verification (set `STOCKFISH_PATH` to a Stockfish 19 executable):

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
npm --prefix web run build
# With the local API and Vite server already running:
# PLAYWRIGHT_BASE_URL selects that server; container tests need their own API URL.
PLAYWRIGHT_BASE_URL=http://127.0.0.1:5173 RUN_LIVE_TESTS=1 npm --prefix web run test:e2e -- --grep-invert 'production containers'
```

The first browser attempt was blocked by macOS sandbox restrictions on Chromium;
the verified runs used an approved unsandboxed browser process. Optional preview
and production-container tests use different environments from the normal demo;
the preview inspection case was checked separately, and production containers
were not run locally. The initial Python run had a subprocess import failure
because the reused virtual environment referred to the older checkout. Explicit
`PYTHONPATH=src` ensures subprocesses also use this branch.

This is a serving change with no training, checkpoint selection, model promotion
or deployment. Frozen research budgets and historical results remain unchanged.
