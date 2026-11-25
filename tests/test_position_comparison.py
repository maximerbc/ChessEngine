import asyncio
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import threading

import chess
import chess.engine
from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest

from chessengine.api import create_app
from chessengine.common import digest, read_json
from chessengine.comparison import ComparisonSettings, model_assessments, stockfish_assessment
from chessengine.engine import Search, SearchResult, MATE
from chessengine.evolution_model import export


@pytest.fixture
def settings():
    return ComparisonSettings.model_validate({
        **read_json("configs/comparison.json"), "model_nodes": 32, "model_time_ms": 100,
        "stockfish_nodes": 64, "stockfish_time_ms": 100,
    })


@pytest.fixture
def fake_stockfish(monkeypatch):
    class Reference:
        def __init__(self):
            self.id = {"name": "Stockfish 19"}
            self.closed = False
            self.returncode = concurrent.futures.Future()
            self.returncode.set_result(0)
            self.result = {"wdl": chess.engine.PovWdl(chess.engine.Wdl(600, 300, 100), chess.BLACK),
                           "score": chess.engine.PovScore(chess.engine.Cp(123), chess.BLACK),
                           "depth": 3, "nodes": 64}

        def configure(self, options):
            self.options = options

        def analyse(self, board, limit, game):
            self.board, self.limit = board, limit
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

        def close(self):
            self.closed = True

    reference = Reference()
    monkeypatch.setattr("chessengine.comparison.digest", lambda path: "a" * 64)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **k: reference)
    return reference


def test_direct_and_searched_values_have_independent_white_perspectives(settings, monkeypatch):
    monkeypatch.setattr(Search, "run", lambda self, board: SearchResult(None, .7, 2, 12, 3, []))
    for turn, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
        board = chess.Board()
        board.turn = turn
        result = model_assessments(board, lambda b: .25, settings)
        assert result["position_value"] == .25 * sign
        assert result["search"]["value"] == .7 * sign
        assert result["search"]["mate_winner"] is None
    monkeypatch.setattr(Search, "run", lambda self, board: SearchResult(None, -MATE + 3, 2, 12, 3, []))
    result = model_assessments(chess.Board(), lambda b: .25, settings)
    assert result["search"]["value"] is None
    assert result["search"]["mate_winner"] == "black"


def test_reference_uses_native_wdl_history_limits_and_cleans_up(settings, fake_stockfish):
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push_uci(uci)
    result = stockfish_assessment(board, settings)
    assert result["value"] == -.5 and result["pawn_score"] == -1.23
    assert result["binary_sha256"] == "a" * 64
    assert fake_stockfish.board.move_stack == board.move_stack
    assert fake_stockfish.options == {"Threads": 1, "Hash": 32, "UCI_ShowWDL": True}
    assert fake_stockfish.limit.nodes == 64 and fake_stockfish.limit.time == .1
    assert fake_stockfish.closed


@pytest.mark.parametrize("failure", ["wdl", "invalid_wdl", "version", "timeout", "crash", "missing_binary"])
def test_reference_errors_do_not_fabricate_values(settings, fake_stockfish, monkeypatch, failure):
    if failure == "wdl":
        del fake_stockfish.result["wdl"]
    elif failure == "invalid_wdl":
        fake_stockfish.result["wdl"] = chess.engine.PovWdl(chess.engine.Wdl(0, 0, 0), chess.WHITE)
    elif failure == "version":
        fake_stockfish.id["name"] = "Stockfish 18"
    elif failure == "missing_binary":
        def missing(*a, **k):
            raise FileNotFoundError("missing")
        monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", missing)
    else:
        fake_stockfish.result = TimeoutError() if failure == "timeout" else chess.engine.EngineTerminatedError()
    result = stockfish_assessment(chess.Board(), settings)
    assert result["status"] == "unavailable"
    assert result["value"] is None and result["pawn_score"] is None
    assert fake_stockfish.closed == (failure != "missing_binary")


def test_reference_mate_is_not_clamped_to_numeric_advantage(settings, fake_stockfish):
    fake_stockfish.result["score"] = chess.engine.PovScore(chess.engine.Mate(3), chess.BLACK)
    fake_stockfish.result["wdl"] = chess.engine.PovWdl(chess.engine.Wdl(1000, 0, 0), chess.BLACK)
    result = stockfish_assessment(chess.Board(), settings)
    assert result["value"] is None and result["pawn_score"] is None
    assert result["mate"] == -3 and result["mate_winner"] == "black"


def test_comparison_supports_every_current_evaluator_without_releasing_candidates(tmp_path, settings, fake_stockfish):
    shutil.copyfile("artifacts/models/supervised.npz", tmp_path / "supervised.npz")
    export(np.zeros(324), tmp_path / "evolutionary.npz")
    manifest = {"default": "supervised", "models": [
        {"id": name, "name": name, "file": name + ".npz", "sha256": digest(tmp_path / (name + ".npz"))}
        for name in ["supervised", "evolutionary"]
    ]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with TestClient(create_app(tmp_path / "manifest.json", settings.model_dump())) as client:
        for model in ["handcrafted", "supervised", "evolutionary"]:
            payload = {"model_id": model, "moves": ["e2e4"]}
            response = client.post("/evaluate-comparison", json=payload)
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["moves"] == payload["moves"]
            assert result["model_id"] == model and result["perspective"] == "white"
            assert result["fen"] == fake_stockfish.board.fen()
            assert result["search"]["nodes"] <= settings.model_nodes
            assert -1 <= result["position_value"] <= 1
            assert result["stockfish"]["value"] == -.5
            assert "best_move" not in result and "resulting_fen" not in result
        assert client.post("/evaluate-comparison", json={"model_id": "rl"}).status_code == 410
        assert client.post("/evaluate-comparison", json={"model_id": "missing"}).status_code == 404
        assert client.post("/evaluate-comparison", json={"moves": ["e2e5"]}).status_code == 422
        assert client.post("/evaluate-comparison", json={"initial_fen": "bad"}).status_code == 422


def test_partial_result_and_history_sensitive_terminal_draw(tmp_path, settings, monkeypatch):
    monkeypatch.setenv("STOCKFISH_PATH", str(tmp_path / "missing"))
    with TestClient(create_app(tmp_path / "manifest.json", settings.model_dump())) as client:
        partial = client.post("/evaluate-comparison", json={}).json()
        assert partial["stockfish"]["status"] == "unavailable"
        assert isinstance(partial["position_value"], float) and partial["search"]["nodes"] <= 32
        # The automatic claim convention ends the game one move before the
        # third occurrence, when a repetition claim becomes available.
        moves = (["g1f3", "g8f6", "f3g1", "f6g8"] * 2)[:-1]
        drawn = client.post("/evaluate-comparison", json={"moves": moves}).json()
        assert drawn["terminal"] == "1/2-1/2"
        assert drawn["search"]["terminal"] == "1/2-1/2"
        assert drawn["stockfish"]["status"] == "terminal"
        assert drawn["stockfish"]["value"] == 0
        assert client.post("/evaluate-comparison", json={"moves": moves + ["e2e4"]}).status_code == 422
        # Same FEN without repetition history is not an automatic draw.
        bare = client.post("/evaluate-comparison", json={"initial_fen": drawn["fen"]}).json()
        assert bare["terminal"] is None
        mate = client.post("/evaluate-comparison", json={"initial_fen": "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"}).json()
        assert mate["terminal"] == "1-0"
        assert mate["stockfish"]["mate_winner"] == mate["search"]["mate_winner"] == "white"
        assert mate["stockfish"]["value"] is None and mate["search"]["value"] is None


def test_comparison_keeps_shared_worker_until_disconnected_work_finishes(tmp_path, settings, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def delayed(board, config):
        started.set()
        assert release.wait(3)
        return {"status": "unavailable"}

    monkeypatch.setattr("chessengine.api.stockfish_assessment", delayed)

    async def run():
        transport = httpx.ASGITransport(app=create_app(tmp_path / "missing", settings.model_dump()))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/evaluate-comparison", json={}))
            try:
                assert await asyncio.to_thread(started.wait, 2)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                for endpoint in ["/analyze", "/evaluate-comparison"]:
                    assert (await client.post(endpoint, json={})).status_code == 429
                assert (await client.get("/health")).json()["busy"]
            finally:
                release.set()
            for _ in range(100):
                if not (await client.get("/health")).json()["busy"]:
                    break
                await asyncio.sleep(.01)
            assert not (await client.get("/health")).json()["busy"]
    asyncio.run(run())


def test_live_comparison_budget(fake_stockfish, monkeypatch):
    live = ComparisonSettings.model_validate(read_json("configs/comparison.json"))
    seen = []

    def probe(self, board):
        seen.append((self.limit, self.time_ms, self.qdepth))
        return SearchResult("e2e4", 0, 1, 1, 1, ["e2e4"])

    monkeypatch.setattr(Search, "run", probe)
    result = model_assessments(chess.Board(), lambda _: 0, live)
    reference = stockfish_assessment(chess.Board(), live)
    assert seen == [(30000, 3000, 4)]
    assert result["search"]["limits"] == {"nodes": 30000, "time_ms": 3000}
    assert reference["limits"]["nodes"] == 30000
    assert reference["limits"]["time_ms"] == 3000
    assert fake_stockfish.limit.nodes == 30000
    assert fake_stockfish.limit.time == 3
    for key, value in [("model_nodes", 30001), ("stockfish_nodes", 30001),
                       ("model_time_ms", 3001), ("stockfish_time_ms", 3001)]:
        with pytest.raises(ValueError):
            ComparisonSettings.model_validate({**live.model_dump(), key: value})


def test_real_local_stockfish_comparison():
    binary = Path(os.getenv("STOCKFISH_PATH", "tools/stockfish/stockfish-macos-universal"))
    if not binary.is_file():
        pytest.skip("Real Stockfish integration requires a local Stockfish 19 binary")
    with TestClient(create_app()) as client:
        response = client.post("/evaluate-comparison", json={"model_id": "supervised-1m", "moves": ["e2e4"]})
        assert response.status_code == 200
        result = response.json()
        assert result["stockfish"]["status"] == "ok", result
        assert result["stockfish"]["version"] == "Stockfish 19"
        assert result["stockfish"]["binary_sha256"] == digest(binary)
        assert -1 <= result["stockfish"]["value"] <= 1
        assert result["stockfish"]["nodes"] <= 31000  # UCI nodes can overshoot by a search batch.
        assert result["search"]["nodes"] <= 30000
