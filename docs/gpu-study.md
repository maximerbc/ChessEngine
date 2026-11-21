# GPU studies: frozen design and evidence boundary

This is a new named study, `gpu-value-v1`, not a continuation of the original 324-parameter experiment or the closed policy/value study. `configs/gpu-study.json` is its prospective definition. No dataset expansion, paid benchmark, full fit, pilot or main tournament has been executed by implementing this infrastructure. The new training budget is unset until the user reviews a measured forecast and selects a provider/budget. No cloud resources are provisioned by the runner.

## Data and value network

The dataset has **1,000,000 positions overall**, exactly 800,000 train / 100,000 validation / 100,000 test. Whole source games are assigned before sampling. Canonical positions are deduplicated globally; reserved positions and source games are excluded. Only eligible historical training records are retained, without their labels. Previously used research examples cannot enter the new validation/test pools. Source archives are the standard rated Lichess January–June 2013 releases, with downloaded SHA-256 identities frozen in the manifest. Source exhaustion fails rather than silently overshooting quotas or changing sources.

All positions are relabeled consistently: Stockfish native side-to-move WDL `(wins-losses)/total`, executable/version identity, explicit EvalFile/EvalFileSmall NNUE hashes, 10,000 nodes, one thread, 32 MiB hash and a cleared hash per position. Labels/chunks are resumable and independent of old labels. The disk-backed feature cache avoids keeping 3.4 GB of dense features plus complete histories in RAM.

Both new models have 853→128→64→1 layers, ReLU hidden activations and tanh output (117,633 parameters), with feature v3. Supervision starts from scratch for seeds 42/43/44, AdamW lr .001 / decay .0001, batch 2,048, at most 40 epochs, patience six. Validation selects epochs. Test labels never enter fitting or checkpoint selection. Seed 42 is representative, not the best observed seed.

## Search, rules and evolution

PUCT uses `Q + 1.5 * prior * sqrt(max(1,parent_visits))/(1+edge_visits)`, uniform priors on legal actions, legal masking and lowest canonical action label on ties. Values are side-to-move; child scores negate once. Search rebuilds each move without transposition merging or tournament noise. CPU uses the same rule convention and action order. Fixed simulation budgets are deterministic on a fixed backend; floating-point differences can affect near-ties across devices. Equal-allowance CPU evaluation is the qualification evidence, not an assumption of universal bitwise device identity.

PGX's low-level legal transitions are wrapped with complete original-history replay, effective-en-passant repetition hashes, repetition features and automatic fifty-move/threefold claims including intended-next-move claims. Checkmate takes precedence over a current clock draw. Built-in PGX 512-ply termination is bypassed. A separate tournament cap gives rule result `*` and neutral fitness .5, never a fabricated rules draw or a training target. The declared 2,048-ply history capacity fails closed on overflow. Source-level rule/hash changes require fresh parity checks against python-chess.

Separable CMA-ES starts independently and uses no teacher labels, supervised initialization or backpropagation. The augmented-supervised seed-42 model is a frozen opponent only. At the planned default, each candidate plays 16 games (eight color pairs) per opponent group: handcrafted-value PUCT, augmented-supervised PUCT and archived evolutionary PUCT. Group fitness weights are .50/.25/.25. Generation openings are common across all candidates. Archive opponents contain only this run's random initializer and earlier generation means.

The planned main total is `64*40*48*3 = 368,640` games; alternatives are 184,320 and 737,280 at 24/96 games per candidate. No count demonstrates sample sufficiency. The separately frozen seed-20261121 ten-generation pilot is **30,720 games**, even when a different main scenario is selected. It requires ≤20% caps, ≥10% decisive rule outcomes and ≥.05 conservative score gain against handcrafted-value PUCT on the same 40 pilot-check pairs. A failed pilot closes this named study. Its seed (20261121) and 600-position training-opening pool are separate from main seeds 42/43/44 and their 1,000-opening pool, so the main run does not replay a pilot prefix.

Before learning, benchmark 32/64/128 simulations and choose the lowest satisfying the frozen criterion: at least 90% of searched moves visit at least 80% of legal root actions, all requested simulations finish, no illegal actions/rule mismatches/history overflow, and resume/parity checks pass. Adjust batch size to fit memory; it does not change game counts. Pilot checks must preserve the same coverage requirements. A smoke run cannot freeze a workload or support a GPU quote.

## Selection and release

Use 40 fresh development pairs, separate 200-pair qualification and 200-pair final pools, all disjoint by source games and canonical positions from corpus/reservations/each other. Freeze all three selections before opening final position tests or matches. Final comparisons can block promotion; they cannot choose a different checkpoint. Report every seed and preserve negative/inconclusive findings.

Current Supervised remains the default and its checksum is immutable. The new four-model lineup retains Classical/current Supervised alpha-beta and adds Augmented Supervised/GPU-trained Evolutionary with CPU PUCT. New cards are unavailable until qualified. Public gates require ≤5% caps, paired 95% lower bounds >50% against Classical and ≥45% against current Supervised, for seed 42 and the three-seed aggregate. Gate sensitivity treats all candidate caps as losses. Bootstrap whole opening pairs together across seeds; report seed SD separately. Do not select on final results or attach old study evidence to new identities.

Different search families receive equal CPU wall-clock allowances (default 1,500 ms); PUCT simulations and alpha-beta nodes remain separate metrics. Manifest v4 records architecture, feature, search, checksums and evidence. Version 3 remains compatible. Staging qualified artifacts does not merge, publish or deploy them.

## Preservation

`docs/preservation/legacy-20261121.json` lists the verified 1,578-member local archive. It includes old datasets, checkpoints/optimizers, interrupted evolution, configurations, reports, PR #15 material and uncommitted research changes. An actual independent restoration succeeded. The compressed archive stays under the primary checkout's ignored `runs/preservation-20261121/legacy-results.tar.gz`; small inference-only 180k seed bundles and the failed-release report are retained in private Git. The active and public 80k model files are unchanged.

See [GPU operations](gpu-operations.md) for staged execution and [training environment](../training/README.md) for locks. Historical protocols and results are unchanged.
