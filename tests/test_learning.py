import json
import random
import chess
import numpy as np
import pytest
import torch
from chessengine.features import encode, record_board
from chessengine.model import build_model, export, NumpyEvaluator
from chessengine.train import fit, seed_all
from chessengine.rl import outcome_value, play_game, selfplay
from chessengine.engine import Handcrafted
from chessengine.data import validate_splits


def dataset(path):
    rows = []
    for i, square in enumerate(["a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"]):
        board = chess.Board()
        board.remove_piece_at(chess.parse_square(square))
        rows.append(
            {
                **record_board(board),
                "split": "validation" if i > 5 else "train",
                "game_id": str(i),
                "target": -0.4,
                "phase": "opening",
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def config(epochs):
    return dict(
        seed=42,
        hidden=[16, 8],
        learning_rate=0.01,
        weight_decay=0,
        batch_size=4,
        epochs=epochs,
        patience=100,
        device="cpu",
        evidence="smoke",
    )


def test_export_parity(tmp_path):
    seed_all(42)
    model = build_model([16, 8])
    path = tmp_path / "model.npz"
    export(model, path, {"hidden": [16, 8]})
    evaluator = NumpyEvaluator(path)
    board = chess.Board()
    with torch.no_grad():
        expected = model(torch.from_numpy(encode(board))).item()
    assert evaluator(board) == pytest.approx(expected, abs=1e-6)


def test_overfit_and_exact_resume(tmp_path):
    path = tmp_path / "data.jsonl"
    dataset(path)
    fit(path, tmp_path / "full", config(8))
    fit(path, tmp_path / "resumed", config(3))
    fit(path, tmp_path / "resumed", config(8), resume=True)
    a = NumpyEvaluator(tmp_path / "full/last.npz")
    b = NumpyEvaluator(tmp_path / "resumed/last.npz")
    for (w1, b1), (w2, b2) in zip(a.layers, b.layers):
        np.testing.assert_array_equal(w1, w2)
        np.testing.assert_array_equal(b1, b2)
    metrics = json.loads((tmp_path / "full/metrics.json").read_text())["history"]
    assert metrics[-1]["train_loss"] < metrics[0]["train_loss"]


def test_leakage_rejected(tmp_path):
    rows = dataset(tmp_path / "data")
    rows.append({**rows[0], "split": "test"})
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(rows)


def test_outcomes_and_truncation():
    assert outcome_value(chess.WHITE, chess.WHITE) == 1
    assert outcome_value(chess.WHITE, chess.BLACK) == -1
    assert outcome_value(None, chess.BLACK) == 0
    cfg = dict(max_plies=1, exploration_plies=20, epsilon=1, nodes=1)
    result, sample = play_game(Handcrafted(), {}, cfg, random.Random(42))
    assert result["truncated"] and sample is None
    result, sample = play_game(
        Handcrafted(),
        {"fen": "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"},
        {**cfg, "epsilon": 0, "nodes": 2000},
        random.Random(42),
    )
    assert not result["truncated"]
    assert sample[1].tolist() == [1]


def test_rl_resume_and_frozen_actor(tmp_path):
    seed_all(42)
    model_path = tmp_path / "initial.npz"
    export(build_model([16, 8]), model_path, {"hidden": [16, 8]})
    openings = tmp_path / "openings.json"
    openings.write_text(
        json.dumps({"train": [{"fen": "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"}], "development": [], "test": []})
    )
    cfg = dict(
        seed=42,
        learning_rate=0.001,
        batch_size=8,
        batches=2,
        games_per_batch=2,
        positions_per_game=4,
        nodes=100,
        max_plies=2,
        epsilon=0,
        exploration_plies=0,
        device="cpu",
    )
    selfplay(model_path, openings, tmp_path / "full", cfg)
    selfplay(model_path, openings, tmp_path / "resume", {**cfg, "batches": 1})
    selfplay(model_path, openings, tmp_path / "resume", cfg, resume=True)
    for (a, b), (c, d) in zip(
        NumpyEvaluator(tmp_path / "full/last.npz").layers, NumpyEvaluator(tmp_path / "resume/last.npz").layers
    ):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
    games = json.loads((tmp_path / "full/games-001.json").read_text())
    assert len({g["actor_sha256"] for g in games}) == 1
