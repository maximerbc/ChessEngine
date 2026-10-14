# Verified local release

## Implementation checks

- Python test suite: 27 tests covering rule/search integration, forced mates and signs, bounded work, history reconstruction and repetition, portable inference parity, exact CPU training resume, frozen-actor RL resume, terminal rewards, truncation exclusion, reserved-game exclusion, deduplication, paired statistics, immutable protocols, development-only selection, API validation/concurrency, and release consistency.
- Browser suite: 7 passing tests, including a real-backend test with released models. Mocked tests cover move/undo/export/compare, mobile layout, unavailable/busy services, reconnect, drag moves, model choice, restart and offline replay. The live test validates actual model loading, a legal response, research artifacts and PDF delivery.
- TypeScript and Vite production build passed. Ruff checks passed.
- The generated four-page PDF was rendered and visually inspected. The final development-performance chart uses self-play counts (20 and 60), distinct from its 16-game development match samples.

## Completed experiments

- 100,000 Stockfish-19-labeled positions: 80,299 train, 9,789 validation, 9,912 test.
- Small model validation MSE 0.306609; large model 0.310330. Small selected before RL. Selected model test MSE approximately 0.300447.
- Three independent RL seeds, 60 self-play games each. All 180 games completed; no truncated outcomes entered main RL updates.
- 400 final games per seed, 1,200 total, with 200 color-swapped opening pairs per seed. No final games reached the move cap.
- Seed 42 score 50.00%, interval 48.375-51.625%; seed 43 score 49.25%, interval 47.50-51.00%; seed 44 score 50.25%, interval 48.375-52.25%. All inconclusive.
- The representative is seed 43, batch 1, selected on development score 56.25%, not final results. All seeds are published.
- Secondary 16-game comparisons: 53.125% vs handcrafted, 50% vs supervised at equal time, 0% vs the pinned Stockfish reference.
- Fifty exact-position-held-out diagnostic puzzles, rating ceiling 1600: 46 solved by supervised, 45 by RL at 1,000 nodes. This is a small diagnostic, not a strength rating.
- Search-depth diagnostic: 38 of 133 sampled positions did not complete depth one under the main 48-node cap. This materially limits the result's reach.
- Actual paid expenditure: C$0.

## Important boundaries

This completed bounded local study is smaller than the original 1,000-self-play-game-per-seed target. It uses shallow search and does not establish RL improvement or transfer to stronger opponents. The larger target configurations and resumable runner remain available.

Local builds and checks do not prove that GitHub Actions has run remotely or that Railway has deployed. Those require the user's push and account authorization. No remote PR, push, public URL, or paid resource has been created by the assistant.

## Railway deployment PR (2026-10-03)

- Python: 30 tests passed, including production manifest requirements, checksum rejection, released checkpoint identities and CORS preflight behavior. Ruff passed.
- TypeScript/Vite production build passed; deployment workflow YAML parsed successfully.
- Existing mocked browser suite: 6 passed. The separately gated live test and container tests are skipped in that default invocation.
- New deployment browser scenarios: 2 passed against real local FastAPI and Vite servers on separate origins. Covered both neural models, legal replies, comparison, PGN export, mobile research during simulated API outage and reconnect. This exercises app behavior, not the Docker/Nginx layer.
- Initial browser launch failed due to macOS sandbox restrictions; rerunning with browser access succeeded. The first mobile recovery assertion expected a desktop-only status badge to be visible; corrected it to check the ready state and disappearance of the outage notice.
- Docker is unavailable locally. Image builds, Nginx serving/headers, container filesystem boundaries, arbitrary PORT binding and container smoke checks remain pending in the new Production containers CI job. Require that job to pass before merge.
- No remote push, PR creation, Railway provisioning, paid activation or public deployment was performed. Model weights and research results are unchanged.

### CI startup race correction

PR #4 run `37101886836`, job `111142883904`, built both images and passed container boundary/model-failure checks, then failed immediately on the first API health probe with `curl: (52) Empty reply from server`. Docker's published port could accept and close a connection before Uvicorn became ready. The previous curl flags retried refused connections but not empty replies.

Both health probes now retry all curl errors, with a 5-second per-attempt timeout and a 60-second retry window. A regression test executes the actual workflow probe commands against a local server that closes the first connection without replying and then returns HTTP 200. The previous flags reproduce exit 52; the corrected API and website probes pass. Full remote container/browser execution still requires pushing this fix and a successful CI rerun.

## Supervised diagnostics (2026-10-09)

- Full local Python suite: 42 passed. Ruff and whitespace checks passed. CI uses the tiny study through the existing Python test workflow; remote execution awaits the user push.
- Completed nine data-size fits plus twelve conditional regularization fits, using existing training/validation records only. Protocol committed at `5bb9a05` before execution. All three full-data seeds exceeded the screening threshold; no treatment met the promising-candidate rule.
- Study used 51.7 seconds of training/validation epoch work and 165.5 seconds invocation wall time, excluding report generation. Actual paid compute: C$0. Per-run times and hashes are in the report evidence.
- Verified complete-game nesting, exclusion of test records, bootstrap reproducibility, boundary calculations, training/evaluation dropout behavior, exact CPU resume and NumPy inference parity. The smoke study also verifies report generation and evidence counts.
- Inspected the HTML report and all three figures. Cross-checked full-study raw validation predictions, subset counts, per-seed aggregates, 20 distinct-game failure examples and released-model hash. No model or public asset replacement.
- Evidence is development-only: repeated validation selection, three seeds, a single nested subset ordering and extra updates with larger data limit causal interpretation. No new final-test evaluation or playing-strength experiment was performed.

## 200k training-data extension (2026-10-11)

- Full local suite: 46 tests passed, including a tiny extension, resumable labeling, teacher mismatch/corruption rejection, original-record preservation, validation-set equality and comparison-report prediction checks. Ruff and whitespace checks pass; remote CI awaits the user push.
- Completed 100,003 new training labels from 22,192 unused source games; original corpus verified byte-for-byte as the combined file prefix. Every completed chunk checksum and combined checksum matched. An independent full-candidate audit found zero overlaps with existing/reserved source games or positions and zero duplicate candidate position keys.
- Completed all three expanded baseline fits. Mean validation MSE 0.273497, versus 0.301320 at the old training size; all three seeds improved. Report generation reloaded all six checkpoint exports, checked their hashes, verified identical validation records, recomputed metrics and bootstrapped paired source games 2,000 times per seed.
- Visually inspected both generated figures and the HTML report. The original diagnostics report, public website assets and released checkpoint files remain unchanged.
- Expanded corpus: 180,302 train, 9,789 validation, 9,912 unused test. Labeling wall time 822.2 seconds; expanded training/diagnostics wall time 216.6 seconds. C$0 paid compute.
- Limitations: same historical source, repeated development validation use, three training seeds and unequal optimizer-update budgets. No claim of stronger chess or deployment readiness for a new checkpoint; controlled matches and fresh confirmatory data remain future work.

## Fixed-data improvements (2026-10-11)

- Full local Python suite: 52 tests passed; Ruff and whitespace checks passed. The tiny study covers all five configurations and the conditional combination path. Additional assertions verify asymmetric castling/en-passant mapping and unchanged targets across all trial versions.
- CPU resume reproduces uninterrupted model weights exactly for all three feature versions with an active scheduler. Export parity, canonical leakage rejection, mismatch rejection and experimental-feature RL guards pass.
- A pre-study smoke test caught a NumPy-boolean JSON serialization failure in summary generation. It was fixed and retested before the protocol commit and any full experiment.
- Scientific implementation/configuration frozen at `5b9882e` before running the full study. Raw predictions and trial weights are ignored; released models, public assets and final-test data remain unchanged.
- Completed 18 CPU runs. Historical references reproduced exactly (zero weight differences); no canonical train/validation overlap. Combined relative/material features and LR 0.0003 achieved mean MSE 0.240760, an 11.97% reduction; all three paired source-game intervals versus reference are below zero. Adaptive LR did not meet the development criterion.
- Independently verified all 18 checkpoint/prediction checksums, recomputed validation MSE from saved predictions, checked identical target/game arrays, subgroup position totals, optimizer-step totals, final ranking and frozen dataset/code hashes. Visually inspected both figures and opened the HTML report in the browser.
- Final local suite on 2026-10-12: 52 passed. Remote GitHub CI remains pending the user push. Epoch work 142.4 seconds, full invocation 363.1 seconds, C$0. Repeated validation selection, unequal compute and slower experimental encoding limit interpretation; no claim of stronger play.

## Averaging and RL follow-up implementation (2026-10-12)

- Averaging study: 55 local tests passed before the RL changes. Three ordinary trajectories reproduced exactly. Weight-average MSE 0.236791 is a 1.65% reduction, below the 2% adoption rule; all three seeds improved. Ensemble MSE 0.226863 costs about three times one-model inference. Saved prediction MSE/checksums independently verified; both figures visually inspected. C$0.
- RL implementation: 60 local tests passed with Ruff and whitespace checks. Includes complete tiny three-seed training/development/final/equal-time/report flow; selection-before-final enforcement; v3 target/pending-resume tests; fresh-source-game/canonical exclusions; pilot selection and shared-pair statistics; persisted compute allowance. An intermediate smoke resume correctly rejected a code change made while that test was running; the final fixed-code suite passed.
- Scientific RL implementation frozen at `36ce76b` before the pilot. The actual full-study outcome is recorded separately in the RL report once available. Tiny test results are not the main experiment. Remote CI requires user pushes.

## RL continuation audit (2026-10-13)

- Full local suite after the authorized time-cap removal: 60 passed. Local CPU computation remains accounted for, but the explicitly requested unlimited-local mode no longer stops at 12 hours. Scientific settings, selection rules and original protocol evidence remain preserved.
- Independently replayed all 3,000 saved self-play PGNs: legal histories and recorded terminal outcomes agree. Seeds 42 and 43 each completed 1,000 games; seed 44 completed 999 and correctly marked one truncated game. Per-batch completed/truncated counts and generation-time totals agree with raw records.
- Reconstructed all 30 frozen actor exports using the preceding model weights and original actor metadata; hashes match every batch and every game. A preliminary audit compared whole-file hashes against training exports and correctly failed because actor metadata differs; reconstruction with the actual actor metadata resolves that audit assumption without changing models or evidence.
- Final evaluation remains in progress. No playing-strength conclusion, remote CI result, push or deployment is claimed.

## Completed stronger RL study (2026-10-13)

- All 3,000 self-play games, 480 development matches, 1,200 final matches and 240 equal-time matches completed. The final runner status is complete and its ledger is closed.
- Independently replayed all 1,920 evaluation PGNs and verified legal outcomes, candidate-perspective scoring, complete swapped-color pairs, candidate/baseline hashes, selection locks and the prescribed development tie rule. Recomputed the shared-pair 10,000-resample aggregate interval independently: score 0.44125, interval [0.4154166667, 0.4658333333].
- One final game hit the move cap; assigning it a loss or win yields aggregate scores 0.440833–0.441667, leaving the conclusion unchanged. The training audit independently confirmed the single truncated self-play game was excluded.
- Measured saved-game and update work totals 6.488 hours, excluding preparation/reporting and interrupted unsaved games. The conservative ledger totals 12.195 hours, including interruption downtime; the user authorized removal of the time cap. C$0 paid compute.
- Inspected the completed learning figure and checked report completion, findings and local asset links. Released models and public website assets are unchanged. Negative results are retained; remote CI, pushes and any future release remain user-controlled.
