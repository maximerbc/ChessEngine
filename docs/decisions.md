# Decision log

## 001 - Shared engine and lightweight inference
Use one Python search implementation for experiments and the API. Train an MLP in PyTorch and export arrays for NumPy CPU inference. This avoids a second JavaScript engine with subtly different semantics, and keeps the server free of the PyTorch runtime. Exact inference parity is tested.

## 002 - Outcome learning before more complex RL
Use Monte Carlo self-play value regression. No teacher labels enter RL. A frozen acting model generates each batch, then equal samples per completed game train the next actor. This is approximate policy iteration with search, not policy-gradient RL or AlphaZero. Truncated games are excluded, which can bias training toward short games; record that limitation.

## 003 - Draw and cache semantics
Automatically take claimable draws, including a claim possible on the next legal move. Search/API/self-play/matches share this convention. Preserve history, and key transpositions by full FEN and move history. This sacrifices transposition reuse for correctness; optimize only with proof/tests of equivalent draw semantics.

## 004 - Private source, public allowlisted demo
The GitHub repository stays private. Render receives explicitly authorized repository access. Only the static build output is public; allowlisted NumPy model exports and research artifacts are copied by a dedicated command. Secrets and raw training data must never enter `web/public`.

## 005 - Evidence levels and compute gates
Smoke tests prove plumbing, pilots estimate feasibility, and a frozen main study tests hypotheses. Never relabel a small pilot as the planned 400-game, multi-seed experiment. Hardware observations and benchmark-driven scope reductions must be recorded before final testing, not after inspecting results.

## 006 - Budgets are study inputs
Store experimental values in `configs`, not in this log or AGENTS. Use single-threaded NumPy BLAS because threaded tiny matrix multiplies hurt search throughput. Paid compute stays at zero until there is evidence it is useful. The budget ceiling remains C$50 including contingency.
