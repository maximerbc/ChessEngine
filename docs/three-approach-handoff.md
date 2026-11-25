# Three playable approaches — local review handoff

> Historical handoff: branch, lineup, and authorization details below apply to its original date. See the [current overview](../README.md) and [operations archive](archive/README.md).

The approved Demo lineup is Classical, Augmented Supervised (default), and Direct-Move CNN. Research now starts with those three methods, followed by four collapsed experiment families. The original charts/replay sit inside the original-study entry; study links reveal the correct family.

## Release identities

| ID | Representative | SHA-256 |
| --- | --- | --- |
| `handcrafted` | Existing baseline | `handcrafted-v1` identity |
| `supervised-1m` | Augmented supervised seed 42 | `1d1b3fa0d436d19be062c4ae782af29f455455d3e7a4038279aa45225d2dbbc2` |
| `elite-policy` | Elite-human CNN seed 42, selected epoch 8 | `5e6deee52407776949178f846b97ed1b02d5af964328beb632b7361b6d4b7687` |

Manifest v7 requires both neural artifacts, exact identities, schemas and the retained CNN export/validation gate. Classical and the augmented evaluator share alpha-beta; the CNN selects a legal move directly through NumPy. Its targets are elite human moves, not Stockfish moves. Neither neural model is presented as having established superior playing strength.

Earlier manifests are preserved under `artifacts/research/`; private historical checkpoints remain intact. The old supervised export is excluded from static/container packaging. Retired API identities explicitly return unavailable; browser links select the augmented default with a notice and preserve the position/history, including offline navigation.

## Verification

- Integrated Python suite: **228 passed, 2 skipped**, 118.06 seconds.
- Integrated default browser suite: **58 passed, 8 skipped**. The skipped cases require production containers, a live backend, or the optional research-preview build.
- Real local API/Stockfish browser integration: **2 passed**. All three opponents were exercised with current-position reference analysis; the CNN exposes no position value or search.
- Final Research navigation checks: **6 passed**, including same-page study navigation, automatic archive expansion and offline retired links.
- Focused release/catalog/retirement checks after separating tests by subsystem: **13 passed**.
- TypeScript/Vite production build, CLI help and Ruff checks passed.
- Demo and Research screenshots were visually inspected at desktop/mobile sizes; browser assertions verify no horizontal overflow.
- Docker was unavailable locally. Production-container checks were not executed and no deployment is claimed.

The first browser attempts failed because the default browser cache was empty and the sandbox blocked Chromium process registration. The installed browser and an approved local browser process resolved these environment failures. The first live reference check timed out using the original executable path; it passed using a byte-identical local Stockfish 19 copy and an isolated API/proxy. Initial Python/browser assertions for the retired default and old page labels were updated, and the final checks above passed. No scientific thresholds, checkpoint selections or measurements were changed.

## Reproduce locally

Use the project's documented Python environment and `npm ci` in `web`. The existing shared verification environment needs `PYTHONPATH=src` so it imports this checkout rather than its older editable installation.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m chessengine --help
npm --prefix web run build
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/elite-playwright npm --prefix web run test:e2e
```

`API_PROXY_TARGET` optionally selects a separate development API; its default remains `http://127.0.0.1:8000`. For live browser checks, run the v7 API with `REQUIRE_MODELS=1` and the pinned Stockfish 19 binary, then use `RUN_LIVE_TESTS=1` with `tests/live.spec.ts`.

The branch is `codex/three-approach-release`, based on verified CNN release `bdc0f82` and now integrated with main `05541a8`. The user explicitly authorized the branch push, PR creation and conflict-resolution update. PR #22 remains for the user to merge; no deployment, new training or paid resources were used.

## Main integration (2026-11-25)

Manifest v7 distinguishes this three-opponent release from main’s earlier v6 catalogue. The complete v6 declaration, private checkpoints, loader and numerical tests remain preserved as historical research infrastructure. The active production image contains only its two declared neural weights and gate; earlier catalogue staging refuses to overwrite v7. Research retains main’s supervised-variant evidence and both Stockfish-report URLs; its old study permalink aliases the comprehensive dated entry without duplicating archive cards.

The first integrated Python run found a checksum error-text compatibility mismatch and a release test using the historical v6 loader. Both were corrected; the complete rerun passed. A live-test attempt ran before API startup and was retried after a successful health check; both live checks passed.
