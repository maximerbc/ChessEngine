import json
import chess
import numpy as np
import pytest
from fastapi.testclient import TestClient
from chessengine.api import create_app
from chessengine.common import write_json
from chessengine.engine import Handcrafted, Search
from chessengine.evaluate import freeze_protocol, select_checkpoint, paired_summary
from chessengine.features import encode, FEATURE_SIZE, restore, record_board
from chessengine.model import NumpyEvaluator, build_model, export


def test_value_perspective_and_piece_square():
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    white = Handcrafted()(board)
    board.turn = chess.BLACK
    assert Handcrafted()(board) == -white
    assert encode(board)[768] == 0
    assert encode(board)[64 * 4 + chess.D1] == 1
    assert encode(board).shape == (FEATURE_SIZE,)


def test_search_matches_unpruned_minimax():
    board = chess.Board("4k3/8/8/8/3q4/8/3R4/4K3 w - - 0 1")
    ev = Handcrafted()

    def brute(b, d, ply):
        terminal = Search.terminal(b, ply)
        if terminal is not None:
            return terminal
        if not d:
            return ev(b)
        vals = []
        for move in list(b.legal_moves):
            b.push(move)
            vals.append(-brute(b, d - 1, ply + 1))
            b.pop()
        return max(vals)

    expected = brute(board, 2, 0)
    result = Search(ev, nodes=100000, max_depth=2, qdepth=0).run(board)
    assert result.value == pytest.approx(expected)
    assert result.move == "d2d4"


def test_history_round_trip():
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push_uci(move)
    reconstructed = restore(record_board(board))
    assert reconstructed.move_stack == board.move_stack
    np.testing.assert_array_equal(encode(reconstructed), encode(board))
    wrong = record_board(board)
    wrong["fen"] = chess.STARTING_FEN
    with pytest.raises(ValueError):
        restore(wrong)


def test_invalid_checkpoint_schema_and_nonfinite(tmp_path):
    p = tmp_path / "test.npz"
    export(build_model([4]), p, {"hidden": [4]})
    with np.load(p) as arrays:
        data = {k: arrays[k] for k in arrays}
    meta = json.loads(str(data["metadata"]))
    meta["feature_version"] = 999
    data["metadata"] = np.array(json.dumps(meta))
    np.savez(p, **data)
    with pytest.raises(ValueError, match="schema"):
        NumpyEvaluator(p)
    meta["feature_version"] = 1
    data["metadata"] = np.array(json.dumps(meta))
    data["w0"][0, 0] = float("nan")
    np.savez(p, **data)
    with pytest.raises(ValueError, match="weights"):
        NumpyEvaluator(p)


def test_protocol_cannot_be_overwritten(tmp_path):
    openings = tmp_path / "openings.json"
    write_json(openings, {"test": [{"id": "one", "fen": chess.STARTING_FEN}]})
    path = tmp_path / "protocol.json"
    freeze_protocol("handcrafted", "handcrafted", openings, {"pairs": 1}, path)
    with pytest.raises(ValueError, match="overwrite"):
        freeze_protocol("handcrafted", "handcrafted", openings, {"pairs": 1}, path)


def test_selection_rejects_test_set(tmp_path):
    write_json(tmp_path / "matches.json", {"protocol": {"split": "test"}, "games": []})
    with pytest.raises(ValueError, match="final-test"):
        select_checkpoint([tmp_path], tmp_path / "selection.json")


def test_bootstrap_pairing_and_cap_sensitivity():
    games = [
        {"opening_id": "a", "candidate_color": "white", "score": 1, "capped": False},
        {"opening_id": "a", "candidate_color": "black", "score": 0, "capped": False},
        {"opening_id": "b", "candidate_color": "white", "score": 0.5, "capped": True},
        {"opening_id": "b", "candidate_color": "black", "score": 1, "capped": False},
    ]
    s = paired_summary(games, repetitions=1000)
    assert s["score"] == 0.625 and s["ci95"] == [0.5, 0.75]
    assert s["uncapped_pairs"] == 1 and s["uncapped_score"] == 0.5


def test_api_finishes_a_game(tmp_path):
    with TestClient(create_app(tmp_path / "missing")) as client:
        result = client.post("/analyze", json={"moves": ["f2f3", "e7e5", "g2g4"], "nodes": 2000}).json()
        assert result["best_move"] == "d8h4"
        assert result["terminal"] == "0-1"
        assert result["mate_score"]
        invalid = client.post("/analyze", json={"initial_fen": "8/8/8/8/8/8/8/8 w - - 0 1"})
        assert invalid.status_code == 422
