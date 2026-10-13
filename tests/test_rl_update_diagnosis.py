import chess
import chess.pgn
import numpy as np
import pytest
import torch
from chessengine.features import encode
from chessengine.model import build_model, export, NumpyEvaluator
from chessengine.rl_update_diagnosis import sample_game, fit
from chessengine.rl_updates import value_loss


def test_anchor_loss_and_detached_target():
    p = torch.tensor([0.5, -0.2], requires_grad=True)
    y = torch.tensor([1.0, -1.0])
    anchor = torch.tensor([0.2, -0.1], requires_grad=True)
    loss = value_loss(p, y, anchor, 1)
    assert torch.allclose(loss, ((p - y) ** 2).mean() + ((p - anchor) ** 2).mean())
    loss.backward()
    assert anchor.grad is None
    assert torch.allclose(p.grad, (p.detach() - y) + (p.detach() - anchor.detach()))
    assert torch.equal(value_loss(p, y), ((p - y) ** 2).mean())
    with pytest.raises(ValueError):
        value_loss(p, y, anchor, -1)


def test_replay_preserves_history_sign_and_excludes_opening():
    board = chess.Board()
    for move in ["f2f3", "e7e5", "g2g4"]:
        board.push_uci(move)
    expected = encode(board, 3)
    board.push_uci("d8h4")

    game = chess.pgn.Game.from_board(board)
    row = dict(pgn=str(game), plies=1, result="0-1", truncated=False)
    x, y = sample_game(row, 32, np.random.default_rng(4))
    assert x.shape == (32, 853) and np.array_equal(x[0], expected)
    assert np.all(y == 1)  # Black is to move before its mating move.
    assert sample_game({**row, "truncated": True}, 32, np.random.default_rng(4)) is None
    with pytest.raises(ValueError):
        sample_game({**row, "result": "1-0"}, 32, np.random.default_rng(4))


def test_fixed_trajectory_resume_identical(tmp_path):
    torch.set_num_threads(1)
    checkpoint = tmp_path / "base.npz"
    export(build_model([8, 4], feature_version=3), checkpoint, {"hidden": [8, 4]})
    config = dict(
        baseline=str(checkpoint),
        treatments={"anchor": dict(learning_rate=0.0001, anchor_weight=1.0)},
        batches=2,
        batch_size=4,
    )
    roots = [tmp_path / "full", tmp_path / "resume"]
    for root in roots:
        root.mkdir()
        for batch in [1, 2]:
            rng = np.random.default_rng(batch)
            np.savez(
                root / f"cache-42-{batch}.npz",
                x=rng.normal(size=(12, 853)).astype(np.float32),
                y=rng.choice([-1.0, 0.0, 1.0], 12).astype(np.float32),
                initial=np.zeros(12, dtype=np.float32),
            )
    fit(config, roots[0], "anchor", 42, "same")
    fit(config, roots[1], "anchor", 42, "same", stop_after=1)
    fit(config, roots[1], "anchor", 42, "same")
    for (wa, ba), (wb, bb) in zip(
        NumpyEvaluator(roots[0] / "anchor-42/batch-002.npz").layers,
        NumpyEvaluator(roots[1] / "anchor-42/batch-002.npz").layers,
    ):
        assert np.array_equal(wa, wb) and np.array_equal(ba, bb)
    with pytest.raises(ValueError, match="mismatch"):
        fit(config, roots[1], "anchor", 42, "changed")


def test_tiny_complete_study_and_mismatch(tmp_path):
    from chessengine.common import write_json
    from chessengine.features import record_board
    from chessengine.rl_update_diagnosis import run

    base = tmp_path / "base.npz"
    export(build_model([8, 4], feature_version=3), base, {"hidden": [8, 4]})
    source = tmp_path / "source"
    source.mkdir()
    board = chess.Board()
    for move in ["f2f3", "e7e5", "g2g4"]:
        board.push_uci(move)
    opening = {**record_board(board), "id": "mate"}
    write_json(source / "openings.json", {"train": [], "development": [opening], "test": []})
    board.push_uci("d8h4")
    game = chess.pgn.Game.from_board(board)
    record = dict(pgn=str(game), plies=1, result="0-1", truncated=False)
    for batch in [1, 2]:
        write_json(source / f"seed-42/training/games-{batch:03}.json", [record])
    data = tmp_path / "data.jsonl"
    import json

    data.write_text(json.dumps({**opening, "target": 1.0, "split": "validation", "game_id": "v"}) + "\n")
    config = dict(
        source=str(source),
        baseline=str(base),
        development_data=str(data),
        output=str(tmp_path / "out"),
        seeds=[42],
        batches=2,
        positions_per_game=4,
        sampling_seed=123,
        batch_size=4,
        treatments={
            "reference": {"learning_rate": 0.0001, "anchor_weight": 0.0},
            "anchored": {"learning_rate": 0.0001, "anchor_weight": 1.0},
        },
        evaluation=dict(pairs=1, nodes=128, qdepth=2, max_plies=4, seed=2026),
    )
    results = run(config)
    assert len(results) == 2 and all(r["development"]["games"] == 2 for r in results)
    assert run(config)[0]["development"] == results[0]["development"]
    with pytest.raises(ValueError, match="mismatch"):
        run({**config, "sampling_seed": 999})
    (tmp_path / "out/cache-42-1.npz").write_bytes(b"bad")
    with pytest.raises(ValueError, match="Cached"):
        run(config)
