# GitHub handoff

The repository remains private. The assistant creates local commits/branches only; the user pushes them. No GitHub PR is claimed to exist until actually created remotely.

The intended local stack is:

- `main`: initial project contract and licensing.
- `feat/foundation`: shared engine, supervised pipeline, API and initial demo.
- `feat/rl-study`: self-play, frozen evaluation, study runner and evidence tooling.
- `feat/showcase`: tested release assets, public report, deployment and final documentation.

After all local verification is complete, push the stack in one command:

```bash
git push origin main feat/foundation feat/rl-study feat/showcase
```

Create three PRs on GitHub: foundation into main; RL study into foundation; showcase into RL study. Merge in order using merge commits (not squash/rebase), retargeting each later PR to main after its predecessor merges. This preserves the shared ancestry and avoids duplicate diffs. The PR descriptions are prepared in `docs/pr-descriptions.md`.

Do not push `data/`, `runs/`, `.venv/`, `tools/`, local credentials or caches. The tracked inference exports and public report files are intentional release assets. The repository itself does not become public when Railway hosts the static build.

If the initial local commits could not be created because of filesystem permission restrictions, do not execute this command blindly: inspect `git branch` and the final handoff first.

## Fourth PR: Railway deployment

`feat/railway-deployment` builds on `feat/showcase`. After the showcase PR is merged, target this PR at `main`; merge only after **Production containers** and existing checks pass. If showcase is still pending, use it as the temporary base and retarget after its merge. Preserve merge-commit ancestry.

After the local implementation handoff, the user pushes:

```bash
git push -u origin feat/railway-deployment
```

Use the fourth PR description in `docs/pr-descriptions.md`. Railway account authorization and production setup are grouped in `docs/deployment.md` and happen after merge.

## Supervised diagnostics PR

The completed study is on `feat/supervised-diagnostics`, following the merged Railway deployment PR. Push this branch once:

```bash
git push -u origin feat/supervised-diagnostics
```

Open a PR targeting `main`, titled **Diagnose supervised generalization with controlled experiments**, using the corresponding draft in `docs/pr-descriptions.md`. Wait for GitHub checks before merging. No training run is needed in CI beyond the tiny smoke study. This PR preserves released models and website assets; its diagnostic report is separate from the public deployment.

## Follow-up: 200k training-data extension

`feat/supervised-200k` builds on `feat/supervised-diagnostics`. The diagnostics branch remains ready for its original push. Merge that PR first; then push the extension when its result handoff is ready:

```bash
git push -u origin feat/supervised-200k
```

Target `main` after diagnostics merges (or temporarily target `feat/supervised-diagnostics`). Preserve merge-commit ancestry to avoid repeating the diagnostics diff. Title: **Expand supervised training to 200k positions**. Full datasets and trial checkpoints stay local; the private repository receives code, configs and compact evidence only.

## Follow-up: fixed-data improvements

`feat/fixed-data-improvements` follows `feat/supervised-200k`. Merge the 200k PR first, then target this PR at `main` (or temporarily use the 200k branch as its base). Preserve merge-commit ancestry. User push:

```bash
git push -u origin feat/fixed-data-improvements
```

Title: **Improve supervised fitting with fixed training data**. The seventh draft in `docs/pr-descriptions.md` records the actual experiment results. No checkpoint or deployment replacement is included. Wait for remote checks before merging.
