# Deploy without making the repository public

Local smoke tests must pass first. The user pushes the reviewed branch/merge; the assistant does not push or change repository visibility.

1. In Render, connect GitHub and grant access to **only the private ChessEngine repository**.
2. Create a Blueprint from `render.yaml`. Keep the engine on the Free instance type. The second service is a static site.
3. Set `VITE_API_URL` on the static site to the API service's HTTPS URL, without a trailing slash. This is a public endpoint, not a secret.
4. Set `ALLOWED_ORIGINS` on the API to the exact static site HTTPS origin, without a trailing slash. Redeploy the frontend after changing its build-time variable.
5. Check `/health` and `/models`, then play a complete game, switch models, compare a position, export PGN, and test on mobile.
6. Let the API sleep and verify that research/saved games remain usable and reconnect recovers live play.

No database is needed. The API loads versioned compact model files from the repository and processes one search at a time. Extra concurrent requests receive HTTP 429. Each request has a maximum 3-second search budget and 10,000 nodes. Request histories are bounded and validated. Do not expose training commands as endpoints.

`publish_artifacts.py` is the only release-copy step. It exports inference weights, the report, figures, and selected public game records. Never copy raw `data/`, `runs/`, `.env`, or full resume checkpoints to the static site. Check the build output before deployment.

Free Render services may sleep and have traffic/build limits. Do not add a card or upgrade automatically; inspect current provider terms when activating the services. There is no keep-alive workaround. If a service is unavailable, the frontend displays the limitation honestly.

## Account boundary

Deployment cannot be completed until the user has pushed and authorized Render to read the private repository. `render.yaml` and the built site are deployable artifacts, not proof that a public URL exists.
