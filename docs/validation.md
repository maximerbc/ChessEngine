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

Local builds and checks do not prove that GitHub Actions has run remotely or that Render has deployed. Those require the user's push and account authorization. No remote PR, push, public URL, or paid resource has been created by the assistant.
