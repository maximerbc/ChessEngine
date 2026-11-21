"""On-demand reference analysis. Never supplies a move to the playing engine."""
from __future__ import annotations

import logging
import os
import time
from typing import Literal

import chess
import chess.engine
from pydantic import BaseModel, ConfigDict, Field

from .common import digest
from .engine import MATE, Search
from .puct_serving import ValueSearch, STRATEGY

log = logging.getLogger(__name__)


class ComparisonSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_nodes: int = Field(ge=1, le=10000)
    model_time_ms: int = Field(ge=10, le=3000)
    stockfish_nodes: int = Field(ge=1, le=10000)
    stockfish_time_ms: int = Field(ge=10, le=3000)
    protocol_timeout_seconds: float = Field(gt=0, le=5)
    threads: Literal[1]
    hash_mb: int = Field(ge=1, le=32)
    binary: str


def winner(value):
    return "white" if value > 0 else "black"


def model_assessments(board, evaluator, settings, strategy='alpha-beta', simulations=64):
    sign = 1 if board.turn == chess.WHITE else -1
    position_value = float(evaluator(board)) * sign
    search = (ValueSearch(evaluator, simulations, settings.model_time_ms) if strategy == STRATEGY
              else Search(evaluator, settings.model_nodes, settings.model_time_ms))
    result = search.run(board)
    is_mate = abs(result.value) > MATE - 1000
    return {
        "position_value": position_value,
        "search": {
            "strategy": strategy,
            "simulations": getattr(result, 'simulations', None),
            "value": None if is_mate else result.value * sign,
            "mate_winner": winner(result.value * sign) if is_mate else None,
            "terminal": result.terminal,
            "depth": result.depth,
            "nodes": result.nodes,
            "elapsed_ms": result.elapsed_ms,
            "pv": result.pv,
            "limits": ({"simulations": simulations, "time_ms": settings.model_time_ms} if strategy == STRATEGY
                       else {"nodes": settings.model_nodes, "time_ms": settings.model_time_ms}),
        },
    }


def stockfish_assessment(board, settings):
    """Fresh process/hash per request, original history, mandatory native WDL."""
    limits = {
        "nodes": settings.stockfish_nodes,
        "time_ms": settings.stockfish_time_ms,
        "threads": settings.threads,
        "hash_mb": settings.hash_mb,
    }
    base = dict(value=None, pawn_score=None, mate=None, mate_winner=None,
                terminal=None, version=None, binary_sha256=None, depth=0,
                nodes=0, elapsed_ms=0, limits=limits, error=None)
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return {
            **base, "status": "terminal", "terminal": outcome.result(),
            "value": 0.0 if outcome.winner is None else None,
            "mate_winner": None if outcome.winner is None else winner(1 if outcome.winner else -1),
        }
    engine = None
    start = time.perf_counter()
    try:
        binary = os.getenv("STOCKFISH_PATH", settings.binary)
        binary_hash = digest(binary)
        engine = chess.engine.SimpleEngine.popen_uci(binary, timeout=settings.protocol_timeout_seconds)
        version = engine.id.get("name", "")
        if version != "Stockfish 19":
            raise ValueError("Comparison requires the pinned Stockfish 19 release")
        engine.configure({"Threads": settings.threads, "Hash": settings.hash_mb, "UCI_ShowWDL": True})
        info = engine.analyse(
            board.copy(stack=True),
            chess.engine.Limit(nodes=settings.stockfish_nodes, time=settings.stockfish_time_ms / 1000),
            game=object(),
        )
        if "wdl" not in info or "score" not in info:
            raise ValueError("Stockfish did not return native WDL and score")
        wdl = info["wdl"].pov(chess.WHITE)
        if wdl.total() <= 0:
            raise ValueError("Stockfish returned invalid WDL estimates")
        score = info["score"].pov(chess.WHITE)
        mate = score.mate()
        cp = score.score()
        return {
            **base, "status": "ok", "version": version, "binary_sha256": binary_hash,
            "value": None if score.is_mate() else (wdl.wins - wdl.losses) / wdl.total(),
            "pawn_score": None if cp is None else cp / 100,
            "mate": mate,
            "mate_winner": ("white" if score > chess.engine.Cp(0) else "black") if score.is_mate() else None,
            "depth": info.get("depth", 0), "nodes": info.get("nodes", 0),
            "elapsed_ms": (time.perf_counter() - start) * 1000,
        }
    except (OSError, ValueError, TimeoutError, chess.engine.EngineError):
        log.warning("Stockfish comparison unavailable", exc_info=True)
        return {**base, "status": "unavailable",
                "error": "Stockfish analysis is unavailable. Please retry shortly."}
    finally:
        if engine is not None:
            # close() closes pipes and terminates a stalled subprocess; unlike an
            # unbounded quit handshake it also works after analysis timed out.
            engine.close()
            try:
                engine.returncode.result(timeout=settings.protocol_timeout_seconds)
            except (TimeoutError, chess.engine.EngineError):
                log.warning("Stockfish subprocess did not finish closing promptly")
