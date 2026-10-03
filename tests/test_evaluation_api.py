import pytest
from fastapi.testclient import TestClient
from chessengine.api import create_app
from chessengine.evaluate import paired_summary


def test_pairs_and_ci():
    games = [
        dict(opening_id=str(i), candidate_color=c, score=0.5, capped=False)
        for i in range(4)
        for c in ["white", "black"]
    ]
    result = paired_summary(games, repetitions=100)
    assert result["score"] == 0.5 and result["ci95"] == [0.5, 0.5]
    assert result["conclusion"] == "inconclusive"
    with pytest.raises(ValueError):
        paired_summary(games[:-1])


def test_api_limits_history_and_moves(tmp_path):
    with TestClient(create_app(tmp_path / "missing")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/models").json()["default"] == "handcrafted"
        response = client.post("/analyze", json={"nodes": 30}).json()
        assert response["best_move"] and response["checkpoint_id"] == "handcrafted-v1"
        assert response["nodes"] <= 30
        assert client.post("/analyze", json={"moves": ["e2e5"]}).status_code == 422
        assert client.post("/analyze", json={"nodes": 10001}).status_code == 422
        assert client.post("/analyze", json={"model_id": "missing"}).status_code == 404
        moves = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2
        terminal = client.post("/analyze", json={"moves": moves[:-1]}).json()
        assert terminal["terminal"] == "1/2-1/2" and terminal["best_move"] is None
