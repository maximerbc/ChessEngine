# Move-ranking study

Configuration: `configs/move-ranking.json`. This bounded study adds a separate legal-move scorer to the frozen feature-v3 supervised evaluator. The board encoder starts from its seed-42 checkpoint; the value evaluator itself never changes. Three rankers use seeds 42, 43 and 44. Seed 42 is the representative before any results are viewed.

For legal moves A(s), the network emits scores l(s,a). Training minimizes cross-entropy -log(exp(l(s,a*)) / sum over a in A(s) exp(l(s,a))), where a* is Stockfish's first principal-variation move. Padding is masked before softmax. This measures imitation of the teacher's choice, not a win probability. Unlike scalar value regression, it learns which move to examine first.

Whole source games define approximately 50,000 training and 5,000 validation positions. Labels use the pinned Stockfish executable and settings in the data protocol. An initial audit failed on two canonical train/test overlaps in the existing corpus. All four affected source games are quarantined from this new study, documented in its exclusion audit; the original corpus is unchanged. No validation/test labels enter training. Fresh final openings exclude corpus and previous reserved source games and positions.

Checkpoint selection uses minimum validation cross-entropy, with the fixed epoch limit and patience in configuration. Report cross-entropy, top-one/top-three agreement and mean teacher-move rank for every seed. Exact CPU resume preserves optimizer and RNG state.

Search computes ranking once at the root, with its runtime charged to the search budget. Previous-iteration PV stays first. No legal moves are removed; deeper ordering and quiescence are unchanged. All seeds receive 40 development opening pairs at 2,000 nodes and at 100 ms per move. Fixed-depth 2/3 comparisons have 100,000-node and 3-second safety limits; report incomplete searches separately.

The preselected seed 42 receives 200 fresh final pairs at 100 ms per move after development runs complete. Use color swaps, clear caches, paired bootstrap intervals, and report capped-game sensitivity. Final results cannot select a different representative. Compare the new supervised initializer against the current released default on 40 development pairs at 100 ms: only a score of at least 50% permits the new release; otherwise preserve existing public checkpoints and deliver a preview.

Full experiments remain local; paid cost is zero. Benchmark training and matches before full launches, retain raw ignored artifacts and compact committed evidence. Classification accuracy and search efficiency do not by themselves establish stronger chess. No new live RL training is included; the RL page presents the completed negative live-self-play study honestly.
