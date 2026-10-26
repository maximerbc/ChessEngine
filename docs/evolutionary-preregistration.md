# Evolutionary evaluation: frozen implementation protocol

Authorized 2026-10-26. This replaces playable self-play, not its historical evidence.
The complete machine-readable settings are in `configs/evolutionary.json`; the
runner freezes their bytes, scientific source hashes, dependency versions and input
hashes before any pilot. Protocol edits after measurements require an explicit new
experiment, never silent resume. No paid compute, push or deployment is authorized.

## Model and features

Every one of 324 parameters is learned, including material effects. There are no
teacher targets, imported weights, outcome regression, fixed material score or
gradient updates. Chess summaries are explicitly human-designed prior knowledge.
The scalar STM prediction is tanh(u ReLU(Wx+b) + wx+c), with W=4x63 and
parameter layout W[252], b[4], u[4], w[63], c[1]. Inference uses float32,
including during evolutionary candidate evaluation; CMA-ES uses float64 internally.

Feature groups, in order, are counts[10], summed centrality[10], pawn advancement[2],
pawn structure[8], king safety[6], attack coverage[10], phase[1], rights[4], EP[9],
halfmove[1], repetition[2]. Each group's own-side entries precede opponent entries;
piece order is pawn, knight, bishop, rook, queen. Counts divide by 8/2/2/2/1;
centrality is 3.5-(abs(file-3.5)+abs(rank-3.5))/2, summed and divided by
3.5 times starting piece count. Promoted pieces are not clipped. Pawn advancement
sums relative ranks/48. Doubled excess, isolated, passed and pawn-defended pawn
counts divide by 8. Passed pawns have no opposing pawn ahead on their own or
adjacent files; pawn-defended means attacked by a friendly pawn. King features are
friendly pawns on the next rank in three neighboring files/3, attacked adjacent
king squares/8 and centrality/3.5. Attack coverage is the union of each piece type's
attacks excluding friendly occupancy, divided by 64; it is not legal mobility.
Phase is non-pawn/non-king count/14. Rights are own K/Q then opponent K/Q. EP is
file a-h plus absence, relative rank being implicit in a legal position and STM.
Clock is min(halfmove/150,1); repetition-2/3 is computed from original history.

## Data, objective and study

Fresh source-game-separated pools contain 600 training, 120 pilot, 40 pilot-check,
40 development-screen, 200 qualification and 200 final starts. All pools exclude
the supervised corpus as well as every old reserved group: this conservative
implementation strengthens the minimum qualification/final exclusion requirement.
Only source IDs and positions are used from the labeled corpus; labels are not
consumed. Human result headers and continuation moves are never training targets.
Full moves may be hashed solely to identify source games. Train/pilot pools are
balanced by remaining non-king pieces: <=10, 11-22, 23-30. Other pools use 12-ply
starts. No source or starting canonical position may occur twice across pools.

Full-covariance cma 4.5.0, population 16, Normal(0,.05) mean, sigma .1, bounds
[-3,3], 40 generations and seeds 42/43/44 are fixed. Each generation uses six
identical paired starts for all candidates: two each versus Classical, released
Supervised, and one deterministically sampled archived mean. Group fitness weights
are .5/.25/.25. Archive contains mean zero and every fifth completed generation.
Means, not best individuals, are retained for selection at generations 10/20/40.
Curriculum: generations 1-10 endgame; 11-20 half endgame/middle; 21-40 half
middle/opening. Cap=512 extra plies, automatic draw claims, qdepth=4, fresh search
tables. Capped results stay '*'; a neutral .5 tournament adjudication is explicitly
separate from a rule outcome. There are no position-level outcome labels.

Benchmark 512/2048/8192 nodes, four starts per stratum, Classical/Supervised/random
evaluators. Choose smallest passing budget (>=99% depth one overall, >=90% depth
two on middle/opening). Pilot seed=20261026, ten generations: cap fraction<=20%,
decisive fraction>=10%, >=.05 score gain against Classical on 40 pilot-check pairs,
combined conservative peak memory<4 GiB. Failed gates end this study without
retries. Runtime forecasts precede learning pilot and full computation, use measured
p90 game durations and 50% margin, and are not time allowances. At most two game
workers, one numerical thread per worker; paid cost C$0.

Development selects each seed's generation by average score against the two fixed
opponents; ties prefer earlier. All selections freeze before qualification/finals.
Representative seed 42 is fixed, not the best seed. Primary budget=10,000 nodes and
1,500 ms; secondary=node-only selected pilot budget, both qdepth 4. Qualification
and fresh final production comparisons require representative and shared-pair
three-seed aggregate 95% bootstrap lower bounds >.50 versus Classical and >=.45
versus Supervised, with >=2 seed point scores >.50 versus Classical and <=5% caps.
Gates use pessimistic capped-game fitness 0, not just neutral adjudication. Report
ordinary results alongside pessimistic sensitivity and seed SD separately from
opening-pair uncertainty. The final pass/fail may block release but never select
another model. All three seeds get both final budgets even if qualification fails.

## Operations and release

Run `.venv/bin/python -m chessengine evolutionary --stage all`; stages prepare,
benchmark, pilot, train, evaluate, report, release are independently available.
Private `runs/evolutionary` stores resumable optimizer state, pending populations,
match ledgers, hashes and elapsed time. Public checkpoints are non-pickled NumPy;
only private self-generated optimizer state is unpickled. No optimizer updates are
repeated on resume. Report and model publication are independent.

Active manifest version 3 defaults to Supervised and lists required models.
Historical release identities are frozen separately. RL analysis returns 410;
retired weights never enter public assets or containers. Three Demo cards remain;
Evolutionary is disabled until qualified, while Research retains all outcomes.

Main was verified through the connector at 4be7055a746fbd6bc0eb7cb223a771531b22977f.
Its complete file diff against local 3e11f2d was empty (four remote merge commits).
The new local branch starts at that verified equivalent tree; no user branch or
shared main was reset, pushed or rewritten.
