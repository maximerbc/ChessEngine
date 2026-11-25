# Research map

The current demo presents Classical, Augmented Supervised, and Direct-Move CNN. Experiments below have distinct datasets, checkpoint identities, search budgets, and evidence levels; their metrics are not interchangeable. The [active manifest](../artifacts/models/manifest.json) determines playable identities.

| Study | Status | Finding and next reading |
| --- | --- | --- |
| Original supervised/self-play study | Completed; improvement inconclusive | RL seed scores 50.00%, 49.25%, and 50.25%; all paired intervals include 50%. [Protocol](local-study-preregistration.md), [report](../web/public/research/report.html). |
| Supervised diagnostics and data expansion | Completed development evidence | Audits, regularization, and larger datasets measure prediction error; they do not demonstrate stronger play. [Diagnostics](supervised-diagnostics.md), [200k report](experiments/supervised-200k/index.html). |
| Fixed-data representations and averaging | Completed development evidence | Prediction averaging improved MSE at higher inference cost; weight averaging missed its adoption gate. [Fixed-data study](fixed-data-improvements.md), [averaging](supervised-averaging.md). |
| Stronger Monte Carlo self-play | Completed; negative | 44.13% against the initializer, paired 95% interval 41.54–46.58%. [Report](experiments/rl-improvement/index.html). |
| RL regression diagnosis | Off-policy development diagnostic | Fixed-trajectory replay and prediction anchoring are separate from fresh on-policy evidence. [Methods](rl-update-diagnosis.md). |
| Root move ordering | Completed; release gate failed | Learned ordering only changes root order; its study does not establish stronger play. [Evidence](experiments/move-ranking/README.md). |
| Policy/value PUCT | Feasibility pilot completed; full comparison deferred | 60 attempts establish feasibility only; the two-treatment study is untested. [Scope and audit](policy-value-release.md). |
| Augmented supervised value learning | Training completed; experimental playable | 800k training positions; mean three-seed validation MSE 0.21476 on its own validation pool. [Evidence](experiments/gpu-value/index.html). |
| Augmented search ablation | Completed development comparison | Alpha-beta improved the screen relative to the failed PUCT screen; reused openings and different node ceilings do not establish exact demo strength. [Results](experiments/gpu-value/alpha-beta-screen-result.json). |
| Elite human-move CNN | Baseline completed; experimental playable | Fixed seed-42 checkpoint passed validation/export checks; direct human-move prediction is distinct from Stockfish labeling. [Protocol](elite-policy.md), [evidence](experiments/elite-policy/index.html). |
| Stockfish move-policy CNN | Completed; unstable across seeds | Seeds 42/44 passed strict sampled export checks; seed 43 collapsed and failed. [Full report](experiments/stockfish-move-policy-1m-20261124/README.md). |
| Evolutionary learning | Historical interrupted work and separate GPU infrastructure | No evolutionary model is active in manifest v7. Pilots and planned workloads are not completed strength studies. [Original protocol](evolutionary-preregistration.md), [GPU operations](gpu-operations.md). |

Research preserves negative results and failures. Original self-play, later regression, replay diagnostics, policy/value feasibility, and evolutionary experiments must retain their own identities. Historical operational costs and authorizations apply to their dated runs.

For the implementation, read [mathematics](mathematics.md), [development](development.md), or the [interactive research reports](../web/public/research/studies.json). Reports are committed artifacts; this documentation cleanup does not regenerate or alter them.
