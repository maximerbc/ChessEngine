# Security

This is an experimental portfolio/research project. Security reports should identify the affected commit, component, impact, and minimal reproduction without including live credentials or private data.

## Reporting a vulnerability

Use GitHub's **Report a vulnerability** option if it is available on this repository. Private vulnerability reporting has not been verified as enabled. If the option is absent, contact the maintainer through an existing private channel; do not disclose sensitive details in a public issue or pull request. Ordinary bugs without sensitive content can use GitHub issues. No response-time guarantee is promised.

If a credential has been exposed, revoke or rotate it first. Removing the current file or adding an ignore rule does not remove earlier commits, PRs, logs, artifacts, or existing copies.

## Scan before publication

Use the pinned Gitleaks CLI described in the [automated workflow](.github/workflows/secrets.yml), from a full clone with all intended branches and tags:

```bash
gitleaks git --redact --log-opts="--all" .
git archive HEAD | tar -x -C /path/to/an/empty/audit-directory
gitleaks dir --redact --max-archive-depth 3 /path/to/an/empty/audit-directory
```

The archive scan covers tracked files, including nested NPZ metadata, without scanning ignored datasets or local credentials. Use an existing empty directory outside the checkout. The workflow performs both scans and does not upload findings. Reviewed exceptions must identify a specific harmless match; do not exclude whole evidence directories or all hashes. Reports should redact values and still identify file, commit, and rule.

See the [public-readiness report](docs/public-readiness.md) for the actual audit coverage, privacy decisions, and remaining publication blockers. Automated scans cannot certify that arbitrary artifacts or GitHub-hosted discussions contain no sensitive information.
