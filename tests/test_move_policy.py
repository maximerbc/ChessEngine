"""Correctness and research-integrity checks; tiny fixtures are not study evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import chess
import numpy as np
import pytest
import torch
from chessengine.common import digest, json_hash, write_json
from chessengine.features import encode, record_board
from chessengine.move_policy import build_policy, export_policy, NumpyMovePolicy
from chessengine.move_policy_data import canonical, encode_shards, ShardDataset, collate, label, prepare
from chessengine.move_policy_train import train, evaluate_all, bootstrap, ShuffledShards
from chessengine.ranking import move_features


def fixture_data(root, seeds=None, epochs=2):
    config = {"positions": {"train": 2, "validation": 2, "test": 2}, "seeds": seeds or [42],
              "shard_size": 1000,
              "teacher": {"binary": "unused", "sha256": "unused", "nodes": 10000},
              "training": {"epochs": epochs, "patience": 6, "batch_size": 2,
                           "learning_rate": 0.0003, "weight_decay": 0.0001}}
    openings = [("e2e4", "e7e5"), ("d2d4", "d7d5"), ("c2c4", "e7e5")]
    rows = []
    for split, opening in zip(("train", "validation", "test"), openings):
        board = chess.Board()
        for i, uci in enumerate(opening):
            board.push_uci(uci)
            rows.append({**record_board(board), "game_id": split, "split": split,
                         "teacher_move": list(board.legal_moves)[0].uci()})
    root.mkdir(parents=True, exist_ok=True)
    (root / "positions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    sha = digest(root / "positions.jsonl")
    write_json(root / "positions.manifest.json", {"identity": {"config": config}, "sha256": sha,
                                                "counts": config["positions"]})
    source = root / "labels/00000.jsonl"
    source.parent.mkdir()
    source.write_text((root / "positions.jsonl").read_text())
    write_json(root / "labels.manifest.json", {"complete": True, "positions": sha,
               "identity": json_hash({"positions": sha, "teacher": config["teacher"]}),
               "chunks": [{"path": "labels/00000.jsonl", "sha256": digest(source)}]})
    encode_shards(config, root)
    return config


def test_numpy_export_parity_and_special_moves(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = build_policy().eval()
    path = tmp_path / "policy.npz"
    export_policy(model, path, {})
    policy = NumpyMovePolicy(path)
    boards = [chess.Board(), chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
              chess.Board("7k/8/8/3pP3/8/8/8/7K w - d6 0 1"),
              chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")]
    for board in boards:
        for b in (board, board.mirror()):
            moves = list(b.legal_moves)
            x, features = encode(b, 3), move_features(b, moves)
            with torch.no_grad():
                logits = model(torch.from_numpy(x[None]), torch.from_numpy(features[None]),
                               torch.ones((1, len(moves)), dtype=torch.bool))[0].numpy()
            np.testing.assert_allclose(policy.logits(x, features), logits, atol=3e-6, rtol=2e-5)
            ranked = policy.rank(b)
            assert {r["move"] for r in ranked} == {m.uci() for m in moves}
            assert sum(r["probability"] for r in ranked) == pytest.approx(1.0)
            assert policy.choose(b) in {m.uci() for m in moves}
    assert {r["move"] for r in policy.rank(boards[-1]) if len(r["move"]) == 5} == {
        "a7a8q", "a7a8r", "a7a8b", "a7a8n"}


def test_claimed_draw_history_and_schema(tmp_path):
    path = tmp_path / "p.npz"
    export_policy(build_policy(), path, {})
    policy = NumpyMovePolicy(path)
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        board.push_uci(move)
    assert policy.choose(board) is None
    assert policy.rank(chess.Board(board.fen()))  # Same FEN, missing original repetition history.
    assert policy.choose(chess.Board("7k/8/6K1/8/8/8/8/8 w - - 0 1")) is None
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["feature_version"] = 1
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="schema"):
        NumpyMovePolicy(path)
    metadata["feature_version"] = 3
    arrays["metadata"] = np.array(json.dumps(metadata))
    arrays["c0w"][0, 0, 0, 0] = np.nan
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="weights"):
        NumpyMovePolicy(path)


def test_mask_and_perspective():
    board = chess.Board()
    mirror = board.mirror()
    np.testing.assert_array_equal(encode(board, 3), encode(mirror, 3))
    assert canonical(board) == canonical(mirror)
    x, m, mask, y, _ = collate([(encode(board, 3), move_features(board, list(board.legal_moves)), 0, "a"),
                              (encode(board, 3), move_features(board, [chess.Move.from_uci("e2e4")]), 0, "b")])
    logits = build_policy()(x, m, mask)
    assert torch.isneginf(logits[1, 1:]).all()
    assert torch.isfinite(torch.nn.functional.cross_entropy(logits, y))


def test_shards_integrity_and_sampling(tmp_path):
    fixture_data(tmp_path)
    data = ShardDataset(tmp_path, "train")
    assert sorted(list(ShuffledShards(data))) == list(range(len(data)))
    assert data[0][3] == "train"
    with pytest.raises(IndexError):
        data[-1]
    part = tmp_path / data.parts[0]["path"]
    part.write_bytes(part.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Corrupt"):
        ShardDataset(tmp_path, "train")


def test_exact_cpu_resume_and_test_gate(tmp_path):
    config = fixture_data(tmp_path / "data", seeds=[42, 43])
    root = tmp_path / "data"
    first = train(config, root, 42, stop_after=1)
    assert not first["complete"]
    # Testing must be rejected before even opening a test shard.
    test_shard = next((root / "encoded").glob("test-*.npz"))
    hidden = test_shard.with_suffix(".hidden")
    test_shard.rename(hidden)
    with pytest.raises(ValueError, match="finish selection"):
        evaluate_all(config, root)
    assert not (root / "test-protocol.json").exists()
    hidden.rename(test_shard)
    resumed = train(config, root, 42, resume=True)
    arrays = NumpyMovePolicy(root / "seed-42/best.npz").arrays
    import shutil
    shutil.rmtree(root / "seed-42")
    full = train(config, root, 42)
    for name, value in NumpyMovePolicy(root / "seed-42/best.npz").arrays.items():
        np.testing.assert_array_equal(value, arrays[name])
    assert [h["validation"] for h in full["history"]] == [h["validation"] for h in resumed["history"]]
    train(config, root, 43)
    result = evaluate_all(config, root)
    assert len(result["results"]) == 2 and (root / "test-protocol.json").exists()
    assert "frozen_checkpoint_train" in result["results"][0]
    changed = copy.deepcopy(config)
    changed["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="Dataset/config"):
        train(changed, root, 42, resume=True)


def test_tiny_overfit():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    board = chess.Board()
    row = (encode(board, 3), move_features(board, list(board.legal_moves)), 3, "tiny")
    x, moves, mask, y, _ = collate([row])
    model = build_policy()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = float(torch.nn.functional.cross_entropy(model(x, moves, mask), y).detach())
    for _ in range(80):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x, moves, mask), y)
        loss.backward()
        opt.step()
    after = float(torch.nn.functional.cross_entropy(model(x, moves, mask), y).detach())
    assert after < before * 0.2
    assert model(x, moves, mask).argmax().item() == 3


def test_label_teacher_hash_and_required_inputs(tmp_path):
    config = fixture_data(tmp_path / "data")
    with pytest.raises(FileNotFoundError):
        label(config, tmp_path / "data", tmp_path)
    (tmp_path / "unused").write_text("wrong executable")
    with pytest.raises(ValueError, match="Pinned"):
        label(config, tmp_path / "data", tmp_path)
    with pytest.raises(FileNotFoundError):
        prepare({"source": "missing.pgn", "exclusions": ["missing.json"]}, tmp_path / "new", tmp_path)


def test_bootstrap_and_cloud_guard():
    stats = bootstrap({"a": np.array([4, 1, 2, 3, 2]), "b": np.array([4, 1, 2, 3, 2])})
    assert stats["top1"] == [0.5, 0.5]
    script = Path(__file__).resolve().parents[1] / "scripts/run_move_policy.sh"
    env = {k: v for k, v in os.environ.items() if k != "CHESS_POLICY_RUNPOD_APPROVED"}
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "approval" in result.stderr


def test_numpy_inference_does_not_import_torch(tmp_path):
    path = tmp_path / "policy.npz"
    export_policy(build_policy(), path, {})
    code = ('import sys; from chessengine.move_policy import NumpyMovePolicy; '
            f'p=NumpyMovePolicy({str(path)!r}); '
            'assert "torch" not in sys.modules; import chess; assert p.choose(chess.Board())')
    subprocess.run([sys.executable, "-c", code], check=True)


def test_preparation_exclusions_and_whole_games(tmp_path):
    import random
    import chess.pgn
    from chessengine.data import split_game
    rng = random.Random(18)
    games, excluded = [], None
    for i in range(100):
        board = chess.Board()
        for _ in range(22):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
        game = chess.pgn.Game.from_board(board)
        game.headers["Event"] = str(i)
        games.append(str(game))
        if excluded is None and len(board.move_stack) == 22:
            gid = json_hash({"initial": board.root().fen(), "moves": [m.uci() for m in board.move_stack]})
            excluded = {**record_board(board), "game_id": gid, "split": split_game(gid)}
    (tmp_path / "source.pgn").write_text("\n\n".join(games))
    write_json(tmp_path / "excluded.json", [excluded])
    config = {"source": "source.pgn", "exclusions": ["excluded.json"], "sampling_seed": 9,
              "max_games": 100, "positions": {"train": 3, "validation": 3, "test": 3}}
    result = prepare(config, tmp_path / "out", tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "out/positions.jsonl").read_text().splitlines()]
    assert all(result["counts"][s] >= 3 for s in config["positions"])
    assert excluded["game_id"] not in {r["game_id"] for r in rows}
    assert len({canonical(chess.Board(r["fen"])) for r in rows}) == len(rows)
    assert prepare(config, tmp_path / "out", tmp_path) == result
    (tmp_path / "excluded.json").write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        prepare(config, tmp_path / "out", tmp_path)


def test_encoded_leakage_and_illegal_target(tmp_path):
    from chessengine.common import read_json
    for corruption in ("game", "position", "target"):
        root = tmp_path / corruption
        config = fixture_data(root)
        path = root / "labels/00000.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if corruption == "game":
            rows[2]["game_id"] = rows[0]["game_id"]
        elif corruption == "position":
            rows[2] = {**rows[0], "split": "validation", "game_id": "another-game"}
        else:
            rows[0]["teacher_move"] = "a1a8"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest = read_json(root / "labels.manifest.json")
        manifest["chunks"][0]["sha256"] = digest(path)
        write_json(root / "labels.manifest.json", manifest)
        with pytest.raises(ValueError, match="leakage|Duplicate|illegal"):
            encode_shards(config, root)


def test_label_resume_and_history(tmp_path, monkeypatch):
    from chessengine.common import read_json
    config = fixture_data(tmp_path / "data")
    root = tmp_path / "data"
    # Replace already encoded fixture labels with a fresh fake-teacher labeling run.
    for path in (root / "labels").glob("*"):
        path.unlink()
    (tmp_path / "teacher").write_text("fake executable for deterministic integration test")
    config["teacher"].update(binary="teacher", sha256=digest(tmp_path / "teacher"))
    manifest = read_json(root / "positions.manifest.json")
    manifest["identity"]["config"] = config
    write_json(root / "positions.manifest.json", manifest)
    calls = []

    class Teacher:
        id = {"name": "test-only"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def configure(self, options):
            pass

        def analyse(self, board, limit, game):
            assert board.move_stack and limit.nodes == 10000
            calls.append(board.fen())
            return {"pv": [list(board.legal_moves)[0]]}

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _: Teacher())
    first = label(config, root, tmp_path)
    assert len(calls) == 6 and first["complete"]
    assert label(config, root, tmp_path) == first and len(calls) == 6
    path = root / first["chunks"][0]["path"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="resume"):
        label(config, root, tmp_path)


def test_private_package_allowlist(tmp_path):
    import tarfile
    fixture_data(tmp_path / "data")
    script = Path(__file__).resolve().parents[1] / "scripts/package_move_policy.py"
    archive_path = tmp_path / "private.tar.gz"
    subprocess.run([sys.executable, str(script), "--dataset", str(tmp_path / "data"),
                    "--output", str(archive_path)], check=True, capture_output=True)
    with tarfile.open(archive_path) as archive:
        names = archive.getnames()
        assert "source-revision.json" in names
        assert "configs/stockfish-move-policy.json" in names
        assert "runs/stockfish-move-policy/dataset.json" in names
        assert not any(".git" in n or "artifacts/models" in n or n.endswith(".pt") for n in names)
        stamp = json.load(archive.extractfile("source-revision.json"))
        assert len(stamp["commit"]) == 40
        assert "src/chessengine/move_policy.py" in stamp["files"]


def test_reserved_puzzle_trajectory(tmp_path):
    from chessengine.move_policy_data import exclusion_rows
    path = tmp_path / "puzzles.csv"
    board = chess.Board()
    path.write_text('FEN,Moves,GameUrl\n"' + board.fen() + '",e2e4 e7e5,https://lichess.org/abcdefgh/black#2\n')
    rows = list(exclusion_rows(path))
    assert len(rows) == 3 and rows[-1]["moves"] == ["e2e4", "e7e5"]
    assert rows[-1]["source_site_id"] == "abcdefgh"
