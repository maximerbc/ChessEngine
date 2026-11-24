# Elite human-move baseline handoff

Branch: `codex/elite-pytorch-baseline`, based on the existing augmented alpha-beta
integration at `b33405895c78e94c873687d5155dfdf59d2f3e5b`. Approved Runpod training
is running. No push, merge or deployment has occurred.

## Ready locally

- The independently implemented reference CNN, complete 4208-class vocabulary,
  bounded direct NumPy inference, resumable FP32 training and sealed test/report
  pipeline are implemented.
- The fresh November 2025 elite corpus is prepared: exactly 800,000 training,
  100,000 validation and 100,000 test examples. Input archive SHA-256:
  `1ffa577aa0a1352fa04c9a890a6706049a3a79edc52810b136758d1e2a1a2eab`.
- Dataset SHA-256:
  `093938622f8d8903020ee4afc25c23d787bd5f47607bc479f3f2eeec4d558d49`.
  Counts, exclusions, source games and the independent audit are recorded in
  `docs/experiments/elite-policy/preparation.json`. All million raw/encoded labels,
  positions and split owners were checked; full history/encoding/draw checks were
  sampled at 1004 positions, not claimed as exhaustive replays.
- Dataset preparation took 663.48 seconds after a 31.41-second exclusion-index
  build. The preliminary 100-game pass is part of the same resumable preparation,
  not a trained pilot or playing-strength result.
- The chooser contains Classical, Supervised, Augmented Supervised and Move
  Prediction. The last card remains unavailable until real seed-42 evidence passes
  the gate. Existing supervised checkpoint identities and default are preserved.

## Verification

- Full Python run: 144 passed, 2 skipped. Ten focused new tests passed separately,
  including two additional evidence-gate tests added after full-suite collection.
- Browser suite: 47 passed, 6 skipped; the four direct-policy flows also passed
  separately. Skips require production/live/local-preview services.
- TypeScript/Vite production build, Ruff, CLI help and diff checks passed.
- Docker/container execution is **not verified**: this machine has no Docker
  executable. The existing container CI remains required before deployment.

Retained verification failures: the first broad run exposed a stale public v4
manifest after the private v5 update; the manifests were synchronized. A subsequent
run correctly rejected a source change made while its freeze/resume test was
running; the stable-source run passed. Browser checks initially lacked Chromium,
then hit macOS sandbox Mach-port restrictions; Chromium was installed in a task
cache and browser checks ran with approved execution outside that sandbox. Old
three-card expectations and a strict selector in the new policy test were fixed.
An early audit/loader implementation repeatedly decompressed each shard; it was
corrected before any real fit. No dataset labels or frozen quotas were changed.
Some early verification commands exited 137; clean native Ruff runs were completed afterward.

## Durable private backup

The original project checkout contains
`runs/elite-policy-20261124/pretraining-backup.tar.gz` and the separate runnable
`training-package.tar.gz`. The 203,767,547-byte backup has 39 verified members,
including the runnable data/source bundle, original elite ZIP, indexes, audit
receipts and verification logs. Its checksum receipt is
`docs/experiments/elite-policy/backup.json`. This preserves prepared inputs, not
trained weights. The runnable source is frozen at the recorded pre-receipt commit;
this later backup receipt does not change the scientific source.

## Approved launch and measured forecast

The user approved the bounded run and explicitly requested its launch. RTX 4090
placement attempts were refused for capacity; its queued request was cancelled
and verified cancelled. The single replacement is Secure Cloud RTX PRO 4500
Blackwell in EU-RO-1 at US$0.72/hour. This changes only the execution hardware,
within the approved C$12 inclusive cap; the frozen scientific protocol is unchanged.
The pinned image is `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, digest
`sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.

The pod started at 18:13:22 UTC on November 24, 2026. Independent guards enforce
23:13:22 UTC (five hours including setup), or stop after a verified-backup marker.
Checkpoints live on this run's separate 10 GB network volume. The old evolutionary
pod and volume remain outside this run's cleanup scope.

The discarded GPU benchmark processed 800,000 examples in 24.4773 seconds,
32,683 examples/second. Its 7,343-second training-only forecast becomes 11,915
seconds after the frozen 1.5 overhead factor and 900-second allowance. This passed
the remaining-deadline gate. Seeds 42/43/44 then began the full 100-epoch recipe.
The runnable source is frozen at `a7a142e0a5313a5538749a42a5c6d45aa4843991`;
subsequent report-only edits do not alter the live training package.

The conservative ledger reserves C$10 for prior/pending project costs and C$12 for
this run, C$22 planned cumulative operating cost versus the C$40 allowance. The
USD/CAD 1.60 bound and fee/tax multipliers are budget allowances, not actual rates.
Live billing, final elapsed time and verified trained backups remain pending.
Private launch receipts are under `runs/elite-launch-20261124` and are backed up
with the final outputs. Test metrics remain sealed until every seed attempt and
checkpoint selection is complete.

## Reproduction and final staging

Use the commands and operations in `docs/elite-policy.md`. Run Python commands in
the review worktree with `PYTHONPATH=src` when using the shared local virtualenv.
The package builder includes only allowlisted source, requirements and the audited
dataset; no credentials or existing released weights are uploaded.

After all three seed attempts finish, seal selections before test opening, publish
every outcome and verify downloads locally. `chessengine.elite_release` can stage
only seed 42 after validation/export checks. It must not choose another seed using
test scores. Rebuild the report and both manifests, rerun release/browser/container
checks and prepare the final local release commit. Then verify backups, remove only
this run's temporary pod/volume and confirm their removal through fresh reads.
No remote push, merge or deployment is included.

## Checks during the live run

A stable-source full pytest run passed 146 tests with 2 skips. The updated backup
verifier also passed its corruption-rejection check. A separate clean environment
with only `requirements-runtime.txt` started the API successfully and confirmed
PyTorch is absent. The real three-model browser flows passed (2 production-style
tests, 16.6 seconds), including Stockfish comparison, mobile research during an API
outage, and reconnect. This is native-process evidence; it does not replace Docker
image checks. Repeat the real-model checks after any successful seed-42 staging.
