import chess
import numpy as np
import pytest
import torch
from chessengine.features import encode
from chessengine.model import build_model, export
from chessengine.ranking import move_features, build_ranker, export_ranker, NumpyRanker
from chessengine.engine import Search, Handcrafted
from chessengine.ranking_data import select_games


def baseline(tmp_path):
    p = tmp_path / "value.npz"
    export(build_model([128, 64], feature_version=3), p, {"hidden": [128, 64], "kind": "supervised"})
    return p


def test_move_orientation_promotions_and_special_moves():
    white = chess.Board()
    black = white.mirror()
    assert np.array_equal(
        move_features(white, [chess.Move.from_uci("e2e4")]),
        move_features(black, [chess.Move.from_uci("e7e5")]),
    )
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    moves = [m for m in board.legal_moves if m.promotion]
    assert set(move_features(board, moves)[:, 2]) == {129, 130, 131, 132}
    for fen, uci in [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("7k/8/8/3pP3/8/8/8/7K w - d6 0 1", "e5d6"),
    ]:
        b = chess.Board(fen)
        m = chess.Move.from_uci(uci)
        assert m in b.legal_moves
        assert move_features(b, [m]).shape == (1, 3)


def test_ranker_parity_mask_and_corruption(tmp_path):
    model = build_ranker(baseline(tmp_path))
    board = chess.Board()
    moves = list(board.legal_moves)
    x = encode(board, 3)
    mi = move_features(board, moves)
    mask = torch.ones((1, len(moves)), dtype=torch.bool)
    mask[0, -1] = False
    logits = model(torch.tensor(x[None]), torch.tensor(mi[None]), mask).detach().numpy()[0]
    assert np.isneginf(logits[-1])
    path = tmp_path / "rank.npz"
    export_ranker(model, path, {})
    ranker = NumpyRanker(path)
    np.testing.assert_allclose(logits[:-1], ranker.logits(x, mi)[:-1], atol=2e-6)
    ranked = ranker.rank(board)
    assert {r["move"] for r in ranked} == {m.uci() for m in moves}
    assert abs(sum(r["probability"] for r in ranked) - 1) < 1e-6
    with np.load(path) as source:
        arrays = {k: source[k] for k in source.files}
    arrays["hw"] = np.full_like(arrays["hw"], np.nan)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Invalid"):
        NumpyRanker(path)


def test_root_only_once_budget_and_unguided_compatibility():
    class Ranker:
        calls = 0

        def rank(self, board):
            self.calls += 1
            moves = list(board.legal_moves)[::-1]
            return [dict(move=m.uci(), probability=1 / len(moves)) for m in moves]

    b = chess.Board()
    r = Ranker()
    result = Search(Handcrafted(), nodes=500, ranker=r).run(b)
    assert r.calls == 1 and result.nodes <= 500 and result.move in {m.uci() for m in b.legal_moves}
    first = Search(Handcrafted(), nodes=200).run(b)
    second = Search(Handcrafted(), nodes=200, ranker=None).run(b)
    assert (first.move, first.value, first.depth, first.nodes, first.pv) == (
        second.move,
        second.value,
        second.depth,
        second.nodes,
        second.pv,
    )

    class Invalid:
        def rank(self, board):
            return [dict(move="a1a8", probability=1)]

    with pytest.raises(ValueError):
        Search(Handcrafted(), ranker=Invalid()).run(b)


def test_whole_games_nested():
    rows = [dict(game_id=str(i), row=j) for i in range(6) for j in range(3)]
    small = select_games(rows, 5, 123)
    large = select_games(rows, 11, 123)
    assert large[: len(small)] == small and len(small) == 6
    assert all(sum(r["game_id"] == g for r in small) == 3 for g in {r["game_id"] for r in small})


def test_ranking_resume_and_target_validation(tmp_path):
    from chessengine.common import write_json, digest
    from chessengine.features import record_board
    from chessengine.ranking_train import dataset, train

    value = baseline(tmp_path)
    boards = [chess.Board()]
    b = chess.Board()
    b.push_uci("e2e4")
    boards.append(b)
    rows = [
        {**record_board(b), "split": s, "teacher_move": list(b.legal_moves)[0].uci()}
        for s in ["train", "validation"]
        for b in boards
    ]
    results = []
    for name in ["full", "resume"]:
        root = tmp_path / name
        write_json(root / "label-progress.json", {"complete": True})
        write_json(root / "labels/0000.json", rows)
        write_json(root / "labels/0000.manifest.json", {"sha256": digest(root / "labels/0000.json")})
        config = dict(
            output=str(root),
            baseline=str(value),
            training=dict(learning_rate=0.0003, weight_decay=0.0001, batch_size=2, epochs=2, patience=6),
        )
        data, sha = dataset(config)
        if name == "resume":
            train(config, 42, data, sha, stop_after=1)
        train(config, 42, data, sha)
        results.append(NumpyRanker(root / "seed-42/last.npz"))
        with pytest.raises(ValueError, match="mismatch"):
            train({**config, "other": True}, 42, data, sha)
    for key in results[0].arrays:
        assert np.array_equal(results[0].arrays[key], results[1].arrays[key])


def test_move_label_resume_and_corruption(tmp_path, monkeypatch):
    from chessengine.common import write_json
    from chessengine.features import record_board
    from chessengine.ranking_data import label

    class Teacher:
        id = {"name": "test"}
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def configure(self, config):
            pass

        def analyse(self, board, *args, **kwargs):
            self.calls += 1
            return {"pv": [list(board.legal_moves)[0]]}

    teacher = Teacher()
    manifest = tmp_path / "teacher.json"
    write_json(manifest, {"teacher": teacher.id})
    write_json(tmp_path / "positions.json", [record_board(chess.Board()) for _ in range(3)])
    monkeypatch.setattr("chessengine.ranking_data.prepare", lambda config: tmp_path)
    monkeypatch.setattr("chessengine.ranking_data.chess.engine.SimpleEngine.popen_uci", lambda path: teacher)
    config = dict(label_chunk_size=2, binary="test", teacher_manifest=str(manifest), label_nodes=10)
    assert label(config, pilot=True)["positions"] == 2
    assert label(config)["complete"] and teacher.calls == 3
    label(config)
    assert teacher.calls == 3
    write_json(tmp_path / "labels/0000.json", [])
    with pytest.raises(ValueError, match="mismatch"):
        label(config)


def test_guided_bundle_and_api_identity(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from chessengine.common import write_json, digest
    from chessengine.ranking import GuidedEvaluator, load_guided
    from chessengine.model import NumpyEvaluator
    from chessengine.api import create_app

    value = baseline(tmp_path)
    rank = tmp_path / "rank.npz"
    export_ranker(build_ranker(value), rank, {})
    guided = GuidedEvaluator(NumpyEvaluator(value), NumpyRanker(rank))
    bundle = tmp_path / "bundle.json"
    write_json(
        bundle,
        {
            "value": value.name,
            "ranker": rank.name,
            "value_sha256": digest(value),
            "ranker_sha256": digest(rank),
        },
    )
    assert load_guided(bundle).checkpoint_id == guided.checkpoint_id
    manifest = tmp_path / "manifest.json"
    rows = [dict(id=i, name=i, file=value.name, sha256=digest(value)) for i in ["supervised", "rl"]]
    rows.append(
        dict(
            id="move-guided",
            name="guided",
            kind="move-guided",
            value_model_id="supervised",
            ranker_file=rank.name,
            ranker_sha256=digest(rank),
            sha256=guided.checkpoint_id,
        )
    )
    write_json(manifest, {"showcase_version": 2, "default": "supervised", "models": rows})
    monkeypatch.setenv("REQUIRE_MODELS", "1")
    with TestClient(create_app(manifest)) as client:
        result = client.post("/analyze", json={"model_id": "move-guided", "nodes": 100}).json()
        assert result["checkpoint_id"] == guided.checkpoint_id
        assert len(result["root_rankings"]) == 20
        assert result["components"]["value"] == digest(value)
        classical = client.post("/analyze", json={"model_id": "handcrafted", "nodes": 10}).json()
        terms = classical["explanation"]["handcrafted"]
        assert (
            abs(
                np.tanh((terms["material"] + terms["position"]) / terms["scale"])
                - classical["explanation"]["raw_value"]
            )
            < 1e-7
        )
    write_json(manifest, {"showcase_version": 2, "default": "supervised", "models": rows[:2]})
    with pytest.raises(ValueError, match="move-guided"):
        create_app(manifest)
    rank.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_guided(bundle)


def test_guided_matches_resume_and_pairing(tmp_path):
    from chessengine.common import write_json, digest
    from chessengine.features import record_board
    from chessengine.evaluate import freeze_protocol, run_matches

    value = baseline(tmp_path)
    rank = tmp_path / "rank.npz"
    export_ranker(build_ranker(value), rank, {})
    bundle = tmp_path / "bundle.json"
    write_json(
        bundle,
        dict(value=value.name, ranker=rank.name, value_sha256=digest(value), ranker_sha256=digest(rank)),
    )
    openings = tmp_path / "openings.json"
    write_json(openings, {"development": [dict(id="tiny", **record_board(chess.Board()))]})
    protocol = tmp_path / "protocol.json"
    freeze_protocol(
        "guided:" + str(bundle),
        str(value),
        openings,
        dict(pairs=1, nodes=10, max_plies=2, seed=42),
        protocol,
        "development",
    )
    result = run_matches(protocol, tmp_path / "matches")
    assert result["games"] == 2 and result["capped_games"] == 2
    assert run_matches(protocol, tmp_path / "matches") == result
    rank.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run_matches(protocol, tmp_path / "matches")
