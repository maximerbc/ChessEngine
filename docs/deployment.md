# Railway deployment

The owner handles pushes, merging, deployment, and repository visibility. Public-source preparation follows the separate [readiness audit](public-readiness.md); no deployment or paid activation is implied by these files.

The current GPU amendment keeps serving on CPU, with one API replica/one Uvicorn worker and no training dependencies. Existing manifest v3 remains supported. Future qualified `supervised-1m` and `evolutionary-gpu` artifacts use staged manifest v4 with explicit architecture, feature, search, checksums and evidence. Classical/current Supervised retain alpha-beta and exact release identities; current Supervised stays default. New models use CPU PUCT, report simulations separately from nodes, and retain the default 1,500-ms allowance. See [GPU operations](gpu-operations.md); do not deploy staged artifacts without the user's separate decision.

## Two-service setup

The API image also packages checksum-verified Stockfish 19 for the on-demand
[current-position comparison](position-comparison.md). It runs as a bounded
reference subprocess, not as a playing opponent or an additional service.

Authorize Railway's GitHub integration for **only ChessEngine**. Create one project with two repository-backed services, both tracking `main` with repository root `/`. Use Dockerfile builds and set `RAILWAY_DOCKERFILE_PATH` to the path below. Do not override the image start command.

| Setting | chessengine-demo | chessengine-api |
| --- | --- | --- |
| Dockerfile | `deploy/Dockerfile.web` | `deploy/Dockerfile.api` |
| Replicas | 1 | 1 |
| Serverless | Disabled | Enabled |
| Deployment health path | `/health` | `/health` |
| Health timeout | 120 seconds | 120 seconds |
| Restart policy | On failure, maximum 3 retries | On failure, maximum 3 retries |
| Persistent volumes | None | None |

Disable automatic PR environments. Enable waiting for GitHub CI before automatic deployments. Use Railway-generated domains initially; no custom-domain purchase is required. Both images bind to Railway's `PORT` (8080 locally if omitted). Leave the public domain target port on the service's configured port.

Generate both HTTPS domains, then set:

- Demo: `VITE_API_URL=https://<api-domain>` without trailing slash. This is a public **build argument**, provided through Railway service variables; changing it requires a frontend rebuild.
- API: `ALLOWED_ORIGINS=https://<demo-domain>` without trailing slash. No wildcard. Changing it requires redeployment.
- API image defaults: `REQUIRE_MODELS=1`, `MODEL_MANIFEST=/app/artifacts/models/manifest.json`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`. Keep these defaults.

Deploy both services after configuration. API serverless changes take effect on deployment. Keep the website running so reports remain available when only the API is asleep. No keep-alive polling is needed.

## Cost and account boundary

Start with available trial/free credits. Railway currently lists $1/month free usage and a US$5/month Hobby subscription including $5 usage. An always-running website is **not guaranteed to fit free credits**. Paid activation requires the user's explicit approval; the C$50 project ceiling includes hosting, currency conversion and taxes, with 20% contingency retained.

Before activation, inspect the account's current plan and remaining credits. Set a compute usage email alert and hard limit within the available allocation in Workspace Usage; do not treat that monthly limit as a lifetime budget or as cancelling subscription fees. Record actual charges in CAD and stop services before exhausting the remaining project budget. A paid monthly allowance must be agreed before any upgrade. Do not activate Railway Agent or extra services.

A hard usage limit can stop **both** services, including the report. Monitor measured memory, CPU, egress and projected costs after the first deployment. Record the deployed commit, date, URLs, plan, current resource usage and actual cost in the validation handoff. Do not invent estimates or assume unused credits roll over.

## Verification and rollback

1. Confirm both `/health` endpoints succeed and API `/models` matches the released manifest's two neural checkpoint hashes.
2. Play using each neural model, compare a position, export PGN, and check mobile layout. Download the PDF and replay a saved game.
3. Let the API become inactive and sleep. Confirm the report still loads, then reconnect/play to verify real wake-up. Do not claim a specific cold-start duration until measured.
4. Inspect Railway logs and metrics for startup failures, excessive memory or repeated restarts. `/health` is a deployment readiness check, not an ongoing monitoring service.
5. If a release fails, redeploy the last verified commit for each affected service using the same domain variables. Use Railway rollback if that deployment is still retained; otherwise rebuild the recorded commit. No data migration is required.

Only the built website is served by Nginx. The backend image contains runtime source but never serves its filesystem. Allowlisted inference weights and research exports are intentionally public. No `data/`, `runs/`, credentials, or resume checkpoints belong in either image.

## Reproduce container checks

GitHub Actions' **Production containers** workflow builds both images, verifies non-root execution, tests missing/corrupt models, checks HTTP responses and artifact hashes, and runs real-backend browser tests. No Railway credentials are needed in CI.

With Docker available, build from repository root:

```bash
docker build -f deploy/Dockerfile.api -t chess-api .
docker build -f deploy/Dockerfile.web --build-arg VITE_API_URL=http://127.0.0.1:8000 -t chess-web .
docker run --rm --name chess-api -p 8000:8080 -e ALLOWED_ORIGINS=http://127.0.0.1:8080 chess-api
```

In another terminal:

```bash
docker run --rm --name chess-web -p 8080:8080 chess-web
```

Then run `.venv/bin/python scripts/check_deployment.py` and, from `web`, `PLAYWRIGHT_BASE_URL=http://127.0.0.1:8080 DEPLOYMENT_API_URL=http://127.0.0.1:8000 npx playwright test tests/deployment.spec.ts`.

Docker is not available in the current local environment; successful local Python/frontend tests do not establish container success. Require the remote container workflow to pass before merge.

Provider references: [pricing](https://docs.railway.com/pricing/plans), [cost controls](https://docs.railway.com/pricing/cost-control), [Dockerfiles](https://docs.railway.com/builds/dockerfiles), [serverless](https://docs.railway.com/deployments/serverless). Use Dockerfiles and the dashboard settings above; the legacy config-as-code format is not introduced.
