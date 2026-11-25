from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from threadpoolctl import threadpool_limits
from .common import read_json
from .engine import Search, Handcrafted, MATE
from .evolution_model import load_evaluator
from .ranking import NumpyRanker, GuidedEvaluator
from .comparison import ComparisonSettings, model_assessments, stockfish_assessment
from .puct_serving import ValueSearch, STRATEGY
from .elite_policy import NumpyElitePolicy, STRATEGY as ELITE_STRATEGY, validate_release
from .gpu.release import validate_entry
from .gpu.protocol import MODEL_IDS, LEGACY_SHA
from .demo_models import load_catalogue, MODEL_IDS as CATALOGUE_IDS
from .release_lineup import load_three_approaches


class PositionRequest(BaseModel):
    initial_fen: str = Field(default=chess.STARTING_FEN, max_length=100)
    moves: list[str] = Field(default_factory=list, max_length=1024)
    model_id: str = Field(default="handcrafted", max_length=40)


class AnalysisRequest(PositionRequest):
    nodes: int = Field(default=10000, ge=1, le=10000)
    time_ms: int = Field(default=1500, ge=10, le=3000)
    simulations: int | None = Field(default=None, ge=1, le=128)


def create_app(manifest_path=None, comparison_settings=None):
    comparison_settings = ComparisonSettings.model_validate(
        comparison_settings or read_json(os.getenv("COMPARISON_CONFIG", "configs/comparison.json"))
    )
    models = {
        "handcrafted": {
            "name": "Classical",
            "description": "Handcrafted material and positional evaluation with alpha-beta search.",
            "group": "Classical",
            "experimental": False,
            "kind": "handcrafted",
            "evidence": "baseline",
            "checkpoint_id": "handcrafted-v1",
            "evaluator": Handcrafted(),
            "strategy": "alpha-beta",
        }
    }
    manifest_path = Path(manifest_path or os.getenv("MODEL_MANIFEST", "artifacts/models/manifest.json"))
    require_models = os.getenv("REQUIRE_MODELS", "0") == "1"
    if require_models and not manifest_path.is_file():
        raise ValueError("Production model manifest is missing")
    default_model = "handcrafted"
    manifest = {}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get('release_version') == 7:
            models.update(load_three_approaches(manifest, manifest_path.parent))
            default_model = manifest['default']
        elif manifest.get('release_version') == 6:
            models.update(load_catalogue(manifest, manifest_path.parent))
            default_model = manifest['default']
        else:
            if manifest.get('release_version') in {4, 5} and manifest.get('default') != 'supervised':
                raise ValueError('Manifest v4 must preserve the current Supervised default')
            if len({r['id'] for r in manifest['models']}) != len(manifest['models']):
                raise ValueError('Duplicate model identity in manifest')
            if require_models:
                required = manifest.get('required_model_ids')
                if (manifest.get('release_version') not in {3, 4, 5} or not isinstance(required, list)
                        or 'supervised' not in required or len(set(required)) != len(required)
                        or not set(required).issubset({'supervised', 'evolutionary', 'elite-policy', *MODEL_IDS})
                        or {r['id'] for r in manifest['models']} != set(required)):
                    raise ValueError('Production requires a v3 manifest or v4 manifest with explicit required models')
            for row in manifest["models"]:
                if row['id'] == 'rl' or row.get("kind") == "move-guided":
                    continue
                if Path(row['file']).name != row['file']:
                    raise ValueError('Model file must be inside the manifest directory')
                if row['id'] == 'elite-policy':
                    if manifest.get('release_version') != 5:
                        raise ValueError('Elite policy requires manifest v5')
                    evaluator = NumpyElitePolicy(manifest_path.parent / row['file'])
                    if evaluator.checkpoint_id != row['sha256']:
                        raise ValueError('Elite model manifest checksum mismatch')
                    validate_release(row, evaluator, manifest_path.parent)
                    models[row['id']] = dict(name=row['name'], kind='human-move-policy',
                        evidence=row['evidence_id'], checkpoint_id=evaluator.checkpoint_id, evaluator=evaluator,
                        strategy=ELITE_STRATEGY, search=row['search'], experimental=True, qualified=False,
                        strength_status='not-established', documentation_path=row.get('documentation_path'),
                        components={'policy': evaluator.checkpoint_id}, evidence_id=row['evidence_id'])
                    continue
                evaluator = load_evaluator(manifest_path.parent / row["file"])
                expected_kind = {'supervised': 'supervised', 'evolutionary': 'evolutionary', **MODEL_IDS}.get(row['id'])
                if expected_kind is None or evaluator.metadata['kind'] != expected_kind:
                    raise ValueError('Model kind does not match its playable identity')
                if evaluator.checkpoint_id != row["sha256"]:
                    raise ValueError("Model manifest checksum mismatch")
                if row['id'] not in MODEL_IDS and row.get('search',{'strategy':'alpha-beta'}) != {'strategy':'alpha-beta'}:
                    raise ValueError('Existing model identities must preserve alpha-beta search')
                if manifest.get('release_version') in {4, 5} and row['id']=='supervised' and row['sha256'] != LEGACY_SHA:
                    raise ValueError('Manifest v4 must preserve the released 80k Supervised identity')
                if row['id'] in MODEL_IDS:
                    if manifest.get('release_version') not in {4, 5}:
                        raise ValueError('GPU study models require manifest v4')
                    validate_entry(row, evaluator)
                if manifest.get('release_version') in {4, 5}:
                    architecture = [853 if evaluator.feature_version == 3 else 841,
                                    *evaluator.metadata['hidden'], 1]
                    if (row.get('architecture') != architecture
                            or row.get('feature_version') != evaluator.feature_version
                            or not row.get('evidence_id') or not row.get('search')):
                        raise ValueError('Manifest architecture, feature, search or evidence identity mismatch')
                    if row['id'] not in MODEL_IDS and row['search'] != {'strategy': 'alpha-beta'}:
                        raise ValueError('Existing released search must remain alpha-beta')
                models[row["id"]] = {
                    "name": row["name"],
                    "kind": evaluator.metadata["kind"],
                    "evidence": evaluator.metadata.get("evidence", "unknown"),
                    "checkpoint_id": evaluator.checkpoint_id,
                    "evaluator": evaluator,
                    "documentation_path": row.get("documentation_path"),
                    "experimental": row.get("experimental", row["id"] == "rl"),
                    "components": {"value": evaluator.checkpoint_id},
                    "strategy": row.get('search', {}).get('strategy', 'alpha-beta'),
                    "search": row.get('search', {'strategy': 'alpha-beta'}),
                    "evidence_id": row.get('evidence_id'),
                    "qualified": row.get("qualified"),
                    "strength_status": row.get("strength_status"),
                }
            for row in manifest["models"]:
                if row.get("kind") != "move-guided":
                    continue
                value_id = row["value_model_id"]
                if value_id not in models:
                    raise ValueError("Guided value component is missing")
                ranker = NumpyRanker(manifest_path.parent / row["ranker_file"])
                if ranker.checkpoint_id != row["ranker_sha256"]:
                    raise ValueError("Ranker checksum mismatch")
                evaluator = GuidedEvaluator(models[value_id]["evaluator"], ranker)
                if evaluator.checkpoint_id != row["sha256"]:
                    raise ValueError("Guided component identity mismatch")
                models[row["id"]] = dict(
                    name=row["name"],
                    kind="move-guided",
                    evidence="supervised-root-ordering",
                    checkpoint_id=evaluator.checkpoint_id,
                    evaluator=evaluator,
                    documentation_path=row.get("documentation_path"),
                    experimental=row.get("experimental", True),
                    components={"value": models[value_id]["checkpoint_id"], "ranker": ranker.checkpoint_id},
                )
            if manifest.get('release_version') == 5:
                allowed = {'supervised', 'supervised-1m', 'elite-policy'}
                if not {'supervised', 'supervised-1m'}.issubset(models) or not {r['id'] for r in manifest['models']}.issubset(allowed):
                    raise ValueError('Manifest v5 requires the authorized four-model lineup')
            default_model = manifest.get("default", default_model)
            if default_model not in models:
                raise ValueError("Default model is missing")
    if require_models and not ({'supervised-1m'} if manifest.get('release_version') == 7 else {'supervised'}).issubset(models):
        raise ValueError('Production requires supervised model')
    for model in models.values():
        policy = model.get('strategy') == ELITE_STRATEGY
        model['capabilities'] = dict(position_value=not policy, search=not policy, move_probabilities=policy)
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        with threadpool_limits(limits=1):
            yield

    app = FastAPI(title="ChessEngine Lab", version="0.1.0", lifespan=lifespan)
    origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"]
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "models": len(models), "busy": lock.locked()}

    @app.get("/models")
    def list_models():
        return {
            "default": default_model,
            "models": [
                {"id": key, **{k: v for k, v in value.items() if k != "evaluator"}}
                for key, value in models.items()
            ],
        }

    def request_board(request: PositionRequest):
        if (manifest.get('release_version') == 7
                and request.model_id in (CATALOGUE_IDS | {'evolutionary', 'evolutionary-gpu', 'move-guided'}) - models.keys()):
            raise HTTPException(410, 'This opponent is retired from live play. See /research')
        if request.model_id == 'rl':
            raise HTTPException(410, 'Self-play is retired from live play. See /research?study=self-play-archive')
        if request.model_id not in models:
            raise HTTPException(404, "Model is unavailable; no fallback was substituted")
        try:
            board = chess.Board(request.initial_fen)
            if not board.is_valid():
                raise ValueError("Invalid board")
            for uci in request.moves:
                if board.is_game_over(claim_draw=True):
                    raise ValueError("Moves supplied after game termination")
                board.push_uci(uci)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return board

    async def acquire_worker():
        if lock.locked():
            raise HTTPException(
                429, "Engine is busy. Please try again shortly.", headers={"Retry-After": "2"}
            )
        await lock.acquire()

    @app.post("/evaluate-comparison")
    async def evaluate_comparison(request: PositionRequest):
        board = request_board(request)
        await acquire_worker()

        def compare_position():
            selected = models[request.model_id]
            return {
                "initial_fen": request.initial_fen,
                "moves": request.moves,
                "fen": board.fen(),
                "perspective": "white",
                "model_id": request.model_id,
                "checkpoint_id": selected["checkpoint_id"],
                "components": selected.get("components", {}),
                "terminal": board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else None,
                **model_assessments(board, selected["evaluator"], comparison_settings,
                                    selected.get('strategy', 'alpha-beta'),
                                    selected.get('search', {}).get('simulations', 64)),
                "stockfish": stockfish_assessment(board, comparison_settings),
            }

        async def work():
            try:
                return await asyncio.to_thread(compare_position)
            finally:
                lock.release()

        return await asyncio.shield(asyncio.create_task(work()))

    @app.post("/analyze")
    async def analyze(request: AnalysisRequest):
        board = request_board(request)
        await acquire_worker()

        async def work():
            try:
                selected = models[request.model_id]
                strategy = selected.get('strategy', 'alpha-beta')
                if strategy == ELITE_STRATEGY:
                    return await asyncio.to_thread(selected['evaluator'].analyze, board)
                simulations = min(request.simulations or 128, selected.get('search', {}).get('simulations', 64))
                search = (ValueSearch(selected['evaluator'], simulations, request.time_ms)
                          if strategy == STRATEGY else Search(selected['evaluator'], request.nodes, request.time_ms))
                return await asyncio.to_thread(
                    search.run, board
                )
            finally:
                lock.release()

        # Shield ensures disconnects do not release the worker while its thread is still searching.
        result = await asyncio.shield(asyncio.create_task(work()))
        if isinstance(result, dict):
            return {**result, 'model_id': request.model_id,
                    'checkpoint_id': models[request.model_id]['checkpoint_id'],
                    'components': models[request.model_id].get('components', {})}
        resulting = board.copy(stack=True)
        if result.move:
            resulting.push_uci(result.move)
        terminal = resulting.outcome(claim_draw=True)
        return {
            **result.to_dict(),
            "best_move": result.move,
            "resulting_fen": resulting.fen(),
            "terminal": resulting.result(claim_draw=True) if terminal else None,
            "value": max(-1.0, min(1.0, result.value)),
            "mate_score": abs(result.value) > MATE - 1000,
            "value_perspective": "side_to_move_at_request",
            "model_id": request.model_id,
            "checkpoint_id": models[request.model_id]["checkpoint_id"],
            "components": models[request.model_id].get("components", {}),
            "strategy": models[request.model_id].get('strategy', 'alpha-beta'),
            "explanation": {
                "handcrafted": Handcrafted().explain(board) if request.model_id == "handcrafted" else None,
                "feature_version": getattr(models[request.model_id]["evaluator"], "feature_version", None),
                "raw_value": float(models[request.model_id]["evaluator"](board)),
                "side_to_move": "white" if board.turn else "black",
                "piece_counts": {
                    ("own" if color == board.turn else "opponent"): {
                        chess.piece_name(piece): len(board.pieces(piece, color)) for piece in range(1, 6)
                    }
                    for color in [board.turn, not board.turn]
                },
            },
        }

    return app


app = create_app()
