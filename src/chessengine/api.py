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
from .model import NumpyEvaluator


class AnalysisRequest(BaseModel):
    initial_fen: str = Field(default=chess.STARTING_FEN, max_length=100)
    moves: list[str] = Field(default_factory=list, max_length=1024)
    model_id: str = Field(default="handcrafted", max_length=40)
    nodes: int = Field(default=1000, ge=1, le=10000)
    time_ms: int = Field(default=1200, ge=10, le=3000)


def create_app(manifest_path=None):
    models = {
        "handcrafted": {
            "name": "Classical baseline",
            "kind": "handcrafted",
            "evidence": "baseline",
            "checkpoint_id": "handcrafted-v1",
            "evaluator": Handcrafted(),
        }
    }
    manifest_path = Path(manifest_path or os.getenv("MODEL_MANIFEST", "artifacts/models/manifest.json"))
    require_models = os.getenv("REQUIRE_MODELS", "0") == "1"
    if require_models and not manifest_path.is_file():
        raise ValueError("Production model manifest is missing")
    default_model = "handcrafted"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for row in manifest["models"]:
            evaluator = NumpyEvaluator(manifest_path.parent / row["file"])
            if evaluator.checkpoint_id != row["sha256"]:
                raise ValueError("Model manifest checksum mismatch")
            models[row["id"]] = {
                "name": row["name"],
                "kind": evaluator.metadata["kind"],
                "evidence": evaluator.metadata.get("evidence", "unknown"),
                "checkpoint_id": evaluator.checkpoint_id,
                "evaluator": evaluator,
            }
        default_model = manifest.get("default", default_model)
        if default_model not in models:
            raise ValueError("Default model is missing")
    if require_models and not {"supervised", "rl"}.issubset(models):
        raise ValueError("Production requires supervised and RL models")
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

    @app.post("/analyze")
    async def analyze(request: AnalysisRequest):
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
        if lock.locked():
            raise HTTPException(
                429, "Engine is busy. Please try again shortly.", headers={"Retry-After": "2"}
            )
        await lock.acquire()

        async def work():
            try:
                return await asyncio.to_thread(
                    Search(models[request.model_id]["evaluator"], request.nodes, request.time_ms).run, board
                )
            finally:
                lock.release()

        # Shield ensures disconnects do not release the worker while its thread is still searching.
        result = await asyncio.shield(asyncio.create_task(work()))
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
        }

    return app


app = create_app()
