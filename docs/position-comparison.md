# Current-position comparison

The Demo's **Compare with Stockfish** button assesses the displayed board without
applying a move. It supports every available selected evaluator; the current
release keeps Classical and Supervised playable and Evolutionary gated. Archived
Self-play remains unavailable. This feature never changes release selection.

The first row shows the selected evaluator's direct output. The second shows its
shared alpha-beta search assessment. Both rows repeat the **same searched
Stockfish reference**, not Stockfish's internal static evaluation. All values are
converted to White's perspective. Stockfish's normalized value is its native
`(wins - losses) / total` WDL estimate; its pawn score is shown separately. Neither
model values nor agreement gaps are calibrated win probabilities, strength
estimates, or measures of training efficiency. Classical has no training.

`POST /evaluate-comparison` accepts `initial_fen`, the complete UCI `moves` list,
and `model_id`, using the same validation and model availability as `/analyze`.
The response identifies the full request position, checkpoint, White perspective,
`position_value`, `search`, and `stockfish` assessments. Search limits live in
`configs/comparison.json` (override its path with `COMPARISON_CONFIG`). Both
engines currently use 10,000 nodes / 1,500 ms, following the updated gameplay
contract; node definitions differ between engines and this is not a strength
test. Stockfish uses one thread and 32 MB hash, in a fresh process per request.
Reported elapsed reference time includes process startup and analysis. Process
initialization, commands and shutdown have bounded protocol timeouts. The shared
API worker remains occupied until background work and cleanup finish, including
when the client disconnects. No persistent process, cache, volume, new service,
paid resource, model training or automatic comparison is introduced.

Full histories and automatic draw claims are preserved. Terminal positions use
their exact rules outcome without invoking Stockfish. Mate scores remain
explicit rather than being clamped into ordinary advantage values. A terminal
position's direct evaluator output is informational; the exact outcome takes
precedence. Missing native WDL, incompatible engine versions, missing binaries,
crashes and timeouts produce an unavailable reference with a retry message while
retaining the selected engine's assessments. They do not substitute another
reference or prevent normal gameplay. Results clear after position/model changes;
late responses are rejected using position and checkpoint identity.

## Local setup and production image

The default local executable is the existing
`tools/stockfish/stockfish-macos-universal`. Set `STOCKFISH_PATH` to another
Stockfish 19 executable when needed. For example, from the repository root:

```sh
STOCKFISH_PATH="$PWD/tools/stockfish/stockfish-macos-universal" .venv/bin/uvicorn chessengine.api:app --host 127.0.0.1 --port 8000
```

The production Dockerfile downloads the official `sf_19` universal Linux release
for x86-64 or ARM64 at build time. `scripts/install_stockfish.py` verifies the
architecture-specific archive SHA-256 published in the upstream release metadata
before extracting it. The image retains upstream source, `Copying.txt`, authors,
and installation provenance in `/opt/stockfish`; nothing is downloaded at request
time. The API returns the actual binary hash and engine version for analysis.
Existing private-source and static-asset allowlists remain in force.

Upstream: [Stockfish 19 release](https://github.com/official-stockfish/Stockfish/releases/tag/sf_19),
[exact source](https://github.com/official-stockfish/Stockfish/tree/sf_19),
[GPLv3 license](https://github.com/official-stockfish/Stockfish/blob/sf_19/Copying.txt).

Verification uses mocked reference failures/concurrency, real local Stockfish
analysis, and browser tests for adjacent scores, invalidation, mobile layout,
and unchanged moves. Container checks require Docker; no deployment is implied.

## Verification recorded 2026-10-26

- Full Python suite: **114 passed**. After adding a guard against invalid
  zero-total WDL, the focused comparison suite passed **13 tests**, including
  real local Stockfish 19 analysis using the released Supervised checkpoint.
- Browser suite with `RUN_LIVE_TESTS=1`: **38 passed, 3 skipped**. Real backend
  checks cover installed Classical and Supervised evaluators, White- and
  Black-to-move positions, and unchanged move history. Isolated mocked/temporary
  fixtures cover Evolutionary without publishing or promoting it. The skipped
  checks require production containers or an optional showcase preview.
- Frontend production build, focused Ruff checks and `git diff --check` passed.
  Desktop and 390-pixel mobile screenshots were visually inspected; comparisons
  appear beside the board on desktop and directly after it on mobile.
- Downloaded and checksum-verified the pinned Linux x86-64 release with the
  installer, confirming archive layout, executable, source and license. The
  ARM64 checksum is pinned from the same official release metadata; its archive
  and execution were not tested locally. macOS used the existing local binary.
- Docker is not installed in this environment. Image build/runtime and the
  production-container checks remain unverified; no deployment occurred.

Initial browser attempts stopped because the required Chromium build was absent
and then because the macOS sandbox denied its launch. Chromium was installed
into a temporary cache and the local browser run was approved outside the
sandbox. The first full run found three stale Evolutionary test expectations
(old Self-play hash/presentation wording); those fixtures were corrected, and
the full suite passed. An initial repetition fixture supplied a move after an
automatic draw claim; it was corrected to the existing claim convention.

Reproduce from the repository root:

```sh
.venv/bin/python -m pytest -q
npm --prefix web run build
# Start the local API with the command above in a separate terminal first.
RUN_LIVE_TESTS=1 npm --prefix web run test:e2e
```

For production containers, follow `docs/deployment.md`; both the HTTP smoke
script and production browser test now require successful real Stockfish
comparisons. Remote pushes and deployments remain user-managed.
