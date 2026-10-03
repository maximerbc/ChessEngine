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
