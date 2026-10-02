# Bounded local study: protocol frozen before outcomes

The full 100,000-position teacher dataset is retained. Both specified network sizes are fitted, selecting the small model if its validation MSE is within 2% of the large model; otherwise use the lower validation error. This selection precedes self-play and uses no final match result.

The measured pilot generated eight games in approximately 20 seconds at 48 nodes/move, with seven completed and one truncated at 256 plies. A two-position search benchmark measured about 2,400-2,800 nodes/second; CPU training updates were about 0.34 ms per 256-position batch. MPS was unavailable to the current Python runtime. These are coarse throughput estimates, not strength measurements.

For an initial completed local study at zero monetary cost, retain **three RL seeds and 200 paired final openings per seed**, but reduce self-play to **60 games per seed**, in three 20-game batches. Use 48 nodes/move, quiescence bound 2, and 256-ply cap for training and primary matches. Evaluate checkpoints after batches 1 and 3 on eight development opening pairs. Select the highest development score, breaking ties in favor of the earlier checkpoint. Final results do not affect selection.

These settings are saved in `configs/local-*.json`. This is explicitly a bounded local study, not the originally targeted 1,000 self-play games per seed / 1,000-node evaluation. The target configurations remain available. Shallow search and limited RL experience materially restrict conclusions, especially transfer to the 2,000-node interactive demo. Do not claim general playing-strength improvement from this experiment alone.

Report each seed's score and paired bootstrap interval. Report standard deviation across seeds separately. For the representative demo/report checkpoint, select the seed with the highest development score (ties: seed order 42,43,44). Publish all final seed results, including regressions. No selection by final score.

Secondary comparisons use the same representative checkpoint against the handcrafted evaluator and a pinned Stockfish reference; also run a small equal-time comparison. They are diagnostics, not alternative primary endpoints. Report truncation rates and uncapped-pair sensitivity. Paid spend remains C$0.
