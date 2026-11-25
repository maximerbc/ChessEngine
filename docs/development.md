# Development and reproduction

For play only, use the [README setup](../README.md#run-the-demo-locally). Runtime inference uses NumPy; training dependencies are optional.

## Local checks

In the activated Python environment, from the repository root:

```bash
python -m pip install -e '.[train,dev]'
python -m chessengine --help
python -m pytest -q
ruff check src tests scripts
npm --prefix web ci
npm --prefix web run build
cd web
npx playwright install chromium
npm run test:e2e
```

Python tests include small training fixtures, not full studies. Some browser tests require live API/Stockfish services, production containers, or optional preview builds and skip without them. The normal suite starts Vite; see [live checks](three-approach-handoff.md#reproduce-locally) for additional environments. Reused virtual environments should set `PYTHONPATH=src` to import this checkout.

## Reproduce research

Start with the [research index](research.md), then use the exact study's protocol, configuration, source identity, input hashes, and environment. A fresh clone includes inference exports and selected evidence, not raw corpora, complete optimizer states, or local backup archives. Some reproduction therefore requires preparing external data and substantial compute.

- [Original local study](study.md) and [frozen local protocol](local-study-preregistration.md).
- [Augmented value study](gpu-study.md) and [locked GPU environment](../training/README.md).
- [Elite human-move CNN](elite-policy.md) and [Stockfish move-policy results](experiments/stockfish-move-policy-1m-20261124/README.md).
- [Supervised diagnostics](supervised-diagnostics.md), [fixed-data study](fixed-data-improvements.md), and [averaging](supervised-averaging.md).

Never overwrite frozen evidence or use final-test outcomes for checkpoint selection. Use new output directories for new runs. The closed policy/value continuation remains deferred and must not be restarted by following historical commands. No training, paid compute, or deployment is part of documentation setup.

## Operations

[Deployment](deployment.md) describes the existing container setup. [Historical handoffs](archive/README.md) preserve earlier branch and run instructions; they are not a current release checklist. [Public-readiness](public-readiness.md) records the publication audit and its limits.
