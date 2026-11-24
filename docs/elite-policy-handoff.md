# Elite human-move baseline handoff

Branch: `codex/elite-pytorch-baseline`, based on the existing augmented alpha-beta
integration at `b33405895c78e94c873687d5155dfdf59d2f3e5b`. No push, merge,
deployment, GPU provisioning, gradient benchmark or real fit has occurred.

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

## Exact launch gate

Read `docs/experiments/elite-policy/launch-proposal.json`: it is **not an approval**.
It proposes one Secure Cloud RTX 4090 at the observed US$0.89/hour, the pinned
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` image, one new 10 GB network volume,
SSH only, a five-hour independent deadline and a C$12 inclusive new-run cap.
The four-hour planning estimate is unmeasured; the first authorized GPU operation
is a discarded 30-second benchmark. Full fitting starts only if its measured
forecast plus overhead fits the remaining deadline and cumulative budget.

The proposal reserves C$10 for prior/pending project costs, giving C$22 planned
cumulative operating cost with this run, below C$40 plus C$10 contingency.
The USD/CAD 1.60 bound and fee/tax multipliers are conservative budget allowances,
not a claim about the actual exchange rate or tax rate. Reconcile live bills and
refresh capacity at launch. Stop before provisioning if the ledger no longer fits.

The user must explicitly confirm before any billable GPU benchmark or training,
as required by the accepted plan and the dated `AGENTS.md` amendment. Once approved,
the operator creates the receipt binding the exact source/data, actions, deadline
and cumulative forecast; the job wrapper rejects absent or mismatched approval.

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
