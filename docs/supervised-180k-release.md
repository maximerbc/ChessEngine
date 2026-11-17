# Supervised 180k release check

The requested candidate is the existing feature-v1, 128/64 MLP trained with seed
42 on 180,302 positions. It is not the later feature-v3 combined candidate, which
failed its separate development release screen. No new training or seed selection
is part of this change.

The baseline is the supervised model declared by main at
`ba08f1822285676c33f3dd7c98bf664ab1105896`, trained on 80,299 positions.
The candidate's source checkpoint SHA256 is
`05d37bdb0ecd6717c719608e66c21fd30b19e6dd6ce27d72763d6d982d848ba5`;
the baseline release SHA256 is
`ff3436d07d18eb4953e24e6c83b55d25bbb46b44094ed347f99320148ae04c4b`.

Before measurement, freeze 40 development opening pairs from the existing
evolutionary opening pool, which excludes the expanded supervised corpus and old
reserved source games and canonical positions. Use the existing supervised
release-screen settings: 100 ms per move, 1,000,000,000-node ceiling, qdepth two,
512 additional plies, automatic draw claims, fresh search tables, and swapped
colors for each opening. Both evaluators receive identical settings. Use two
local workers and record elapsed time; paid cost is C$0.

The candidate is eligible only after all 80 games complete, with score at least
50%, at most 5% capped games, and uncapped-pair score at least 50%. The latter two
conditions strengthen the prior screen's point-score rule. Preserve unfinished
rule results and report neutral-adjudication sensitivity separately. Bootstrap
opening pairs, rather than treating games as independent. A pass is a bounded
development release decision, not a claim of statistically established stronger
play or fresh final evidence. Production search remains at 10,000 nodes, 1,500 ms
and qdepth four; this screen does not measure those production settings.

Do not replace the release if this screen fails. Original local-study identities,
historical reports, recorded matches and evolutionary training inputs remain
frozen. Export only inference weights and allowlisted metadata; update both
release manifests and their public model copy together. The active release's
evidence must identify the new checkpoint rather than reuse the original study.

Reproduce the screen with the original private dataset/checkpoints and frozen
opening pool present:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/check_supervised_180k.py
PYTHONPATH=src .venv/bin/python -m chessengine.supervised_release runs/supervised-200k/baseline-all-seed-42/best.npz runs/supervised-180k-release
```

The second command stages the sanitized candidate and audited evidence, without
changing the active manifests. Adding `--promote` is permitted only after a
passing screen and refuses this failed candidate. The
screen can resume from its saved baseline copy after promotion. Public compact
protocols and PGNs reproduce the summary and allow an independent rules audit;
private training files and local paths are not included in public assets.

Existing evolutionary protocols pin the previous supervised checksum. This
release does not migrate or restart them: their hash checks must reject the new
active weights. A future cloud study must explicitly pin its baseline and
preregister its amended inputs and settings.

## Completed result

All 80 games completed in approximately 568 seconds of local execution: 27 wins,
24 draws and 29 losses. Candidate score 48.75%, paired 95% bootstrap interval
40.0–58.125%, with zero capped games. Every saved legal history and outcome was
independently replayed. Paid cost was C$0.

This misses the frozen 50% point-score threshold. The result is inconclusive,
not evidence that the candidate is worse. Preserve the original active weights
and both manifests. The private-repository candidate bundle and this draft PR
make the requested replacement reviewable, but do not promote it or change the
threshold after observing results. Any explicit user override would be a new
release-policy decision and must retain this negative gate outcome.
