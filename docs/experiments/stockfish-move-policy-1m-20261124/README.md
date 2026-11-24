# Million-position Stockfish move-policy results

The authorized full Runpod study completed on 2026-11-24, with no separate pilot. The split is exactly **800,000 training / 100,000 validation / 100,000 test positions**. All three seeds completed the frozen early-stopping protocol. **Seed 43 collapsed and fails strict CPU export probability parity; this is an unstable research result, with no model promotion.** Seed 42 remains the predefined representative.

## Held-out results

Each bracket is a 95% whole-source-game bootstrap interval from 1,000 resamples, clustered over 22,603 test games. All three checkpoint hashes were sealed before test evaluation.

| Metric | Seed 42 (representative) | Seed 43 (failed) | Seed 44 |
| --- | ---: | ---: | ---: |
| Test cross-entropy | 2.4044 [2.3961, 2.4133] | 3.2244 [3.2192, 3.2292] | 2.3593 [2.3503, 2.3684] |
| Test top-one agreement | 31.01% [30.71, 31.30] | 6.86% [6.70, 7.01] | 32.13% [31.83, 32.40] |
| Test top-three agreement | 54.05% [53.74, 54.37] | 17.59% [17.35, 17.82] | 55.70% [55.40, 55.98] |
| Mean teacher-move rank | 6.1174 [6.0740, 6.1653] | 15.8517 [15.7778, 15.9222] | 5.8498 [5.8072, 5.8948] |
| Selected epoch | 4 | 1 | 7 |
| Completed epochs | 10 | 7 | 13 |
| Frozen-checkpoint training CE | 2.30282 | 3.22437 | 2.15104 |
| Online loss at selected epoch | 2.43377 | 3.21174 | 2.31352 |
| Strict CPU export check | pass | FAIL | pass |

Across all three seeds, mean test top-one agreement is 23.33%, with sample standard deviation 14.28 percentage points. This seed variability is separate from the whole-game intervals. Seed 44 does not replace seed 42 after seeing test results.

The table distinguishes frozen-checkpoint training error from online loss measured while weights were changing. Full precision, all metric intervals and checkpoint hashes are in [results.json](results.json); every epoch is retained in [selections.json](selections.json). Agreement with a bounded teacher does not establish playing strength, Elo or win probability.

## Data and execution

- Reused the immutable million-position value corpus as position inputs only; stripped scalar labels and generated new Stockfish first-PV move targets. Retained 999,187 positions, quarantined 142 whole source games implicated in conflicts, and refilled 813 positions from the audited fresh pool. Original datasets remain unchanged.
- Preserved complete histories, source-game ownership, reserved-position exclusions and global canonical deduplication. Encoded 2,599 bounded shards; every shard hash passed preflight.
- Pinned Stockfish 19 binary/NNUE, 10,000 nodes, fresh 32 MiB hash and one thread per position; two independent local labeling workers. No Linux teacher substitution.
- Local preparation: 164.0 s; labeling: 8,773.4 s; encoding: 853.6 s. These are wall times, not the sum of parallel worker seconds.
- One Secure Cloud RTX 4090 in EU-RO-1, PyTorch 2.8.0+cu128 / CUDA 12.8, BF16, fresh weights for seeds 42/43/44. AdamW 0.0003, decay 0.0001, batch 256, maximum 40 epochs, patience six. First-epoch updates were retained (18.4 s for training plus validation).
- Full train/evaluate job wall time: 691.2 s. GPU rental uptime: 1,368 s, including setup and backups. The independent guard stopped the Pod after the verified byte-backup marker.
- Downloaded archive and all selected/last/resume artifact hashes verified. All 1,000 teacher chunks, the million-position input, source archive and encoded data are backed up locally. The temporary Pod and new volume were deleted; read-backs returned 404. Unrelated evolutionary resources were untouched.

Training source is frozen at `7b43f0478d89c15562d28f48cdd235538430e752`; [frozen-source.json](frozen-source.json) records the packaged file hashes. [execution.json](execution.json) records exact data, runtime, hashes and cleanup evidence. Private weights, optimizer/RNG states and raw data are excluded from Git and remain in `runs/stockfish-move-policy-1m-20261123` in the original checkout.

## Numerical failure and export qualification

Seed 43 stopped after seven epochs without improvement beyond epoch one. Its test CE (3.22436218) matches the uniform-legal-move baseline (3.22436215); its 6.858% top-one score reflects deterministic legal-move tie breaking, not a successful learned preference. A posthoc read-only audit found constant scores on all 32 inspected positions and absolute logits up to 8.35 billion. The cause of the optimization collapse was not established.

The original relative-logit-only verifier incorrectly accepted large absolute errors. A stricter probability check exposed a 0.1722 maximum probability discrepancy. NumPy now reduces each output row identically, preventing artificial preferences between identical large activations; synthetic regression tests cover this invariant. No weight, checkpoint selection, Torch training forward pass or scientific setting was changed, and no additional fit was run.

After that correction, seeds 42 and 44 pass probability and chosen-move checks on 24 positions each across train/validation/test. Maximum probability errors are 0.0000151 and 0.00000763. Seed 43 still fails probability parity against the original single-position Torch CPU calculation at two of 24 positions (maximum discrepancy 0.1429), despite matching chosen moves. It remains unqualified; tolerances were not relaxed. The verifier writes all failures and exits nonzero instead of declaring the whole export set successful.

All checkpoint hashes pass. [export-verification.json](export-verification.json) is the authoritative functional result. [initial-export-failure.json](initial-export-failure.json) preserves the initial failure; [superseded-relative-only-verification.json](superseded-relative-only-verification.json) is historical and must not be treated as qualification. [numerical-audit.json](numerical-audit.json) explicitly labels the posthoc checks; none selected a replacement checkpoint.

Corrected NumPy inference median latency on the local macOS arm64 machine, one thread, first 32 encoded test positions: seed 42: 0.272 ms, seed 43: 0.267 ms, seed 44: 0.267 ms. These measurements exclude encoding/loading and do not qualify the failed seed. Original frozen-build Runpod-host timings remain in results.json and refer to the pre-correction implementation.

Historical resume requires the original frozen source/runtime: the scientific-file identity guard correctly rejects changed inference-source hashes. The post-training numerical correction does not rewrite checkpoint metadata or the original test measurements.

## Costs and validation

Conservative total estimate: **C$0.70**, including aborted setup, rounded GPU time, storage and 5% conversion / 25% tax allowances at USD/CAD 1.4369. This is below the C$8 operating plan and C$10 inclusive cap. The first aborted Pod has posted US$0.01443172; the main Pod and both volumes had not posted at the final billing read. Zero records are not claimed as free. [costs.json](costs.json) separates posted billing from estimates.

All four CI workflows passed for training source 7b43f04. The final inference/verifier regression set passed cleanly: 20 tests, including the independent guard, plus Ruff and diff checks. Final review-branch CI is tracked in PR #20. The actual downloaded-model verifier intentionally exits 1 for the retained seed-43 functional failure; that is a negative study result, not a passing check.

Other retained failures: initial manual MultiPV configuration was rejected before labeling; the superseded 60,007-position preparation never trained a CNN; its brief GPU rental was canceled before fitting; foreground cloud unpacking timed out before detached setup succeeded. Earlier local combined tests exited 137 after printing results and were not counted as clean exits.

No merge, deployment, public publication or production-model change.
