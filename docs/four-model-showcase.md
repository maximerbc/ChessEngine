# Four-model showcase

The overview leads to `/lab/classical`, `/lab/supervised`, `/lab/move-guided`, and `/lab/rl`. Switching pages carries the initial position and complete move history. Each page explains its inputs, computation, output and limits, with expandable mathematics and live position inspection. Checkpoint identities in the static evidence must match the API before play is enabled.

The guided approach combines a separate cross-entropy-trained move scorer with the same frozen supervised value network. Its probabilities rank legal choices; they are not win probabilities. The RL page uses the completed live-self-play seed-42 batch-10 checkpoint and explicitly reports the regression. Anchored replay is not relabeled as successful live RL.

## Local preview

After all ranker fits finish:

```sh
.venv/bin/python -m chessengine.showcase configs/move-ranking.json runs/showcase-preview
MODEL_MANIFEST=runs/showcase-preview/models/manifest.json .venv/bin/python -m uvicorn chessengine.api:app --port 8000
SHOWCASE_PUBLIC_DIR=../runs/showcase-preview/public npm --prefix web run dev
```

The preview includes only explicitly exported checkpoints and report assets. Public weight metadata omits local training paths. Original historical research remains separately identified as an archive in the existing Research view.

## Release gate

Complete the frozen ranking evaluation first. Only a passing development comparison of the new supervised model against the released default authorizes publishing this checkpoint collection:

```sh
.venv/bin/python -m chessengine.showcase configs/move-ranking.json runs/showcase-release --publish
```

The command refuses publication without completed evidence and a passing gate. It updates the explicit manifest, model downloads and designated report assets. A failed gate leaves the current deployment intact; the four-model preview remains reviewable locally. The user handles remote pushes and Railway deployment.

Railway continues to use the two existing Dockerfiles. Nginx explicitly serves the four `/lab/` routes; missing research/model files still return 404. The API requires checksummed value and ranking components for a version-2 showcase manifest, one worker and one bounded search. No training dependency enters the runtime image.
