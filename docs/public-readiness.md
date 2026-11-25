# Public-readiness audit — 2026-11-25

**Decision: preparation complete; publication remains conditional.** No credential requiring rotation was found in the scanned repository contents/history. Resolve the tracked-weight/privacy choices and hosted-data review below before making the existing repository public. A clean scan cannot prove that arbitrary binary data or external copies contain no secrets.

## Audited version and method

GitHub confirmed a private repository, default branch `main`, head **`6f9fedba2b5359e622395074b41229eb8a1470f5`** (PR #23 merged). Cleanup branch: `codex/public-readiness`. The original working checkout was on an older evolutionary branch and was left untouched.

HTTPS Git authentication was unavailable. An isolated local clone used the byte-identical PR #23 tree already on disk; missing GitHub commits/trees/blobs were read through the connected GitHub API, reconstructed and checked against their Git object IDs. A separate audit object database combined cached local history and verified missing objects; `git fsck --connectivity-only --no-dangling` passed. This is not a history rewrite. The cleanup checkout has full main ancestry, not a shallow publication history.

Coverage:

- **44 captured local branch/remote refs**, plus the verified current main and current evolutionary review head: **46 audit refs, 116 reachable commits, 1,051 unique blob versions**. All six live GitHub branch heads were present in that audit graph. No local tags were present; GitHub's tag-ref namespace returned 404.
- **506 files** tracked at the base commit. Every reachable unique blob was separately exported and scanned, including deleted/older file versions. Git history scanning also covered diffs across the captured graph (Gitleaks reports 90 commits with scanned changes; merge traversal accounts for the larger reachable count).
- **Gitleaks 8.30.1**, downloaded from the [official release](https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1) and checksum-verified. History scan passed with the reviewed exception below. An unfiltered all-blob directory scan reported only the same two harmless checksum matches. Tracked snapshots use archive traversal depth three and default recursive decoding.
- **26 unique NPZ/archive versions** inspected for member names/metadata. Current checkpoint metadata was read using `allow_pickle=False`; no personal paths, email addresses, or authenticated URLs were found in that metadata. Random binary-string regex matches were not treated as decoded metadata.
- The one notebook version (`notebooks/paid-colab.ipynb`) has **zero output cells**. Its source was included in scanning.
- The committed four-page research PDF was text-extracted and scanned; its metadata uses anonymous/default author fields. The first page was rendered for inspection. This is a privacy/content check, not a new scientific or full PDF-layout validation. Existing preview images have no EXIF fields; a fresh local-demo screenshot was captured and inspected.
- Additional inventory checks looked for personal filesystem paths, email-like strings, infrastructure references, and URLs containing username/password authentication. No authenticated URLs were found. See findings below for personal paths and commit emails.

## Findings and publication decisions

| Classification | Location | Finding and required disposition |
| --- | --- | --- |
| Publication blocker under the existing private-weight policy | `artifacts/models/*.npz`, `artifacts/research/**/*.npz`, previous commits | **27 NPZ paths are tracked at current main**, including the direct-move CNN, old self-play model, supervised variants and seed-level research exports. Public Git access exposes them regardless of static deployment allowlists. Decide explicitly whether these exports may be public. If not, prepare reviewed history removal or a separate curated public repository, preserving the private original. No model files were removed or newly promoted here. |
| Privacy decision | Commit author/committer metadata | Historical commits expose the owner's student email; two distinct email identities occur, including GitHub's noreply committer. Future commits use the verified account's noreply address locally. This does not alter old history. Accept this exposure or separately authorize historical cleanup. |
| Privacy decision | `docs/preservation/legacy-20261121.json`; historical `docs/preservation/evolutionary-gpu-native-20261122.json` versions | Preservation inventories contain personal absolute paths. Other `/Users/` occurrences in code/tests are sanitizers or examples. Preserve frozen receipts in this cleanup; accept their path metadata or separately approve a sanitized publication copy with provenance mapping. |
| Reviewed false positive | `docs/experiments/stockfish-move-policy-1m-20261124/frozen-source.json:17` and its `web/public/research/` copy | A SHA-256 keyed by `src/chessengine/api.py` triggered the generic API-key rule. It is an immutable source checksum. `.gitleaks.toml` requires both an exact published path and that exact checksum line. No whole-directory, hash-wide, or provider-rule exclusion was added. Synthetic checks exposed an unsafe global-path exception during preparation; the final exception is attached only to the generic API-key rule, and canaries verify both published paths. |
| Harmless operational references | `.env.example`, Runpod guard scripts, tests, research receipts, PR bodies | Environment placeholders, provider endpoints, run/resource identities, source hashes and measured costs are operational/reproducibility information, not credentials. They can still be removed from a curated public copy if the owner prefers. |
| Publication review still required | Hosted Actions logs and older retained artifacts | Repository Git scans do not clear all hosted logs and attachments. The sampled artifact review is described below; remaining accessible material needs a final owner review or a dedicated audit before visibility changes. |

Ignore rules now cover common keys/credential files, local infrastructure state, backups, training checkpoints, new research NPZ exports, and scanner findings. They do not untrack existing files or erase earlier commits. Existing model/data/evidence identities remain unchanged.

## GitHub-hosted surfaces

Read-only inspection found **23 PRs**, no ordinary issues, no issue comments, no inline PR comments, no submitted PR reviews, and no releases. PR descriptions were reviewed: they describe research results, failed gates, costs and operational handoffs; no credentials were identified. Historical private-publication wording remains part of those remote records and was not edited. PR bodies, reviews, and attachments can contain material beyond Git files.

The Actions API reported **358 workflow runs**. The latest 100 run records were inspected as metadata, not a complete log-content audit. The current main production run `45559244305` has artifact `14996299572`, `production-browser-results` (476,413 bytes). Its downloaded archive matched the GitHub SHA-256 receipt and contains three PNG demo screenshots, with no text logs or raw training data. Automated text scanning reads no secret-bearing text from these PNGs; binary-image scanning is not proof of image-content safety. Older artifacts and full logs remain outside this review. The repository-wide artifact endpoint was unsupported by the connector; per-run artifact reads worked.

Private vulnerability reporting was not verified as enabled. `SECURITY.md` uses GitHub's private reporting option only if available and otherwise an existing private maintainer contact. Before publication, enable a working private reporting route or supply a contact you want public. No hosted demo URL was recorded in repository metadata or verified during this task; the README links to local setup rather than inventing one.

## Verification

- Python: **240 passed, 2 skipped** in 250.02 seconds, with `PYTHONPATH=src` and one BLAS/OpenMP thread.
- Browser: **58 passed, 8 skipped** in 34.5 seconds. Skips need live services, production containers or optional previews. The first sandboxed Chromium launch failed; the approved unsandboxed run passed. A real local API and Vite server also produced the current README screenshot.
- CLI help, TypeScript/Vite production build, Ruff and whitespace checks passed.
- New documentation links, snapshot secret scans, the exact checksum exception and synthetic generic/provider credential canaries were checked separately. The new GitHub workflow has not run remotely; it uses read-only contents permission, a pinned checkout commit, a checksum-pinned scanner, full fetched history, tracked-file snapshots, redacted output, and no uploaded secret findings.
- No engine/API/search/model code, release manifests, existing checkpoint bytes, frozen research evidence, or deployment definitions changed. No training, history deletion/rewrite, remote push, visibility change, deployment, or paid activity occurred.

## Recheck and publish separately

Before publication, resolve the weight/email/path choices, review remaining hosted logs/artifacts, establish a private security-reporting route, and rerun scans on the exact proposed public commit and all branches/tags that will remain accessible. Re-audit any changes made after this dated snapshot.

With a full clone containing the intended refs and Gitleaks 8.30.1 installed:

```bash
git fsck --connectivity-only --no-dangling
gitleaks git --config .gitleaks.toml --redact --log-opts="--all" .
# Use an existing empty directory outside the repository:
git archive HEAD | tar -x -C /path/to/an/empty/audit-directory
(cd /path/to/an/empty/audit-directory && gitleaks dir --config /absolute/path/to/ChessEngine/.gitleaks.toml --redact --max-archive-depth 3 .)
```

Review author metadata with `git log --all --format='%h %ae %ce'` privately. Do not paste suspected credentials into issues or reports. If a later scan discovers a real secret, revoke/rotate it before any publication, then assess history and hosted copies; merely deleting its current file is insufficient.
