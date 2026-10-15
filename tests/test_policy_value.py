import json
import chess
import numpy as np
import pytest
import torch
from chessengine.common import read_json, write_json
from chessengine.features import encode, record_board
from chessengine.model import build_model, export as export_value, NumpyEvaluator
from chessengine.policy_value import build, export, PolicyValue, objective
from chessengine.puct import PUCT, Node
from chessengine.ranking import move_features
from chessengine.pv_data import validate_groups


@pytest.fixture
def initial(tmp_path):
    path = tmp_path / "value.npz"
    torch.manual_seed(7)
    export_value(build_model([128, 64], feature_version=3), path, {"hidden": [128, 64]})
    return path


def test_initial_parity_schema_promotions_and_export(initial, tmp_path):
    model = build(initial)
    path = tmp_path / "dual.npz"
    export(model, path, {})
    runtime = PolicyValue(path)
    boards = [chess.Board(), chess.Board("7k/P7/6K1/8/8/8/8/8 w - - 0 1")]
    for board in boards:
        moves, p, v = runtime.predict(board)
        x = torch.from_numpy(encode(board, 3)[None])
        m = torch.from_numpy(move_features(board, moves)[None])
        logits, value = model(x, m, torch.ones((1, len(moves)), dtype=torch.bool))
        np.testing.assert_allclose(p, torch.softmax(logits, dim=-1).detach().numpy()[0], atol=1e-7)
        assert v == pytest.approx(value.item(), abs=1e-7)
        assert v == pytest.approx(NumpyEvaluator(initial)(board), abs=1e-7)
    assert len([m for m in boards[1].legal_moves if m.promotion]) == 4
    assert (
        len(set(tuple(v) for v in move_features(boards[1], list(boards[1].legal_moves)) if v[2] != 128)) == 4
    )
    with pytest.raises((KeyError, ValueError)):
        NumpyEvaluator(path)
    with np.load(path) as a:
        arrays = {k: a[k] for k in a.files}
    meta = json.loads(str(arrays["metadata"]))
    meta["move_schema"] = 999
    arrays["metadata"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="schema"):
        PolicyValue(path)


def test_anchor_detached_arithmetic_and_masked_gradients():
    for weight in (0.0, 0.1):
        logits = torch.tensor([[0.0, 0.0, -float("inf")]], requires_grad=True)
        v = torch.tensor([0.5], requires_grad=True)
        initial = torch.tensor([-0.5], requires_grad=True)
        loss = objective(logits, v, torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([0.0]), initial, weight)
        assert loss.item() == pytest.approx(np.log(2) + 0.25 + weight)
        loss.backward()
        assert v.grad.item() == pytest.approx(1 + 2 * weight)
        assert initial.grad is None
        assert torch.isfinite(logits.grad).all() and logits.grad[0, 2] == 0


class Uniform:
    def predict(self, board):
        moves = list(board.legal_moves)
        return moves, np.ones(len(moves)) / len(moves), 0.0


def test_puct_mate_sign_budget_history_and_draws():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    before = board.fen()
    r = PUCT(Uniform(), 128).run(board)
    board.push_uci(r.move)
    assert board.is_checkmate() and r.value == 1
    assert r.visits.sum() == r.simulations == 128 and r.policy.sum() == pytest.approx(1.0)
    assert r.terminal_hits and r.network_calls == r.expanded_leaves
    history = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        history.push_uci(uci)
    original = list(history.move_stack)
    draw = PUCT(Uniform(), 4).run(history)
    assert draw.move is None and draw.value == 0 and draw.network_calls == 0
    assert history.move_stack == original
    fifty = chess.Board("7k/8/6K1/8/8/8/R7/8 w - - 100 1")
    assert PUCT(Uniform(), 4).run(fifty).value == 0
    terminal = PUCT(Uniform(), 4).run(board)
    assert terminal.value == -1 and terminal.simulations == 0
    assert before == "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"


def test_selection_priors_noise_and_deterministic_ties():
    node = Node([1, 2], [0.1, 0.9])
    assert node.select(1.5) == 1
    node.n[:] = [10, 10]
    node.w[:] = [10, -10]
    assert node.select(1.5) == 0
    board = chess.Board()
    a = PUCT(Uniform(), 32).run(board)
    b = PUCT(Uniform(), 32).run(board)
    assert a.move == b.move and np.array_equal(a.visits, b.visits)
    with pytest.raises(ValueError, match="RNG"):
        PUCT(Uniform(), 4, noise_fraction=0.25).run(board)
    a = PUCT(Uniform(), 32, noise_fraction=0.25).run(board, np.random.default_rng(9))
    b = PUCT(Uniform(), 32, noise_fraction=0.25).run(board, np.random.default_rng(9))
    np.testing.assert_array_equal(a.visits, b.visits)


def test_legal_castling_en_passant_and_history_preserved():
    boards = [chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")]
    board = chess.Board()
    for m in ["e2e4", "a7a6", "e4e5", "d7d5"]:
        board.push_uci(m)
    boards.append(board)
    for b in boards:
        history = list(b.move_stack)
        r = PUCT(Uniform(), 8).run(b)
        assert set(r.moves) == set(b.legal_moves) and chess.Move.from_uci(r.move) in b.legal_moves
        assert b.move_stack == history
    assert any(boards[0].is_castling(m) for m in PUCT(Uniform(), 4).run(boards[0]).moves)
    assert any(board.is_en_passant(m) for m in PUCT(Uniform(), 4).run(board).moves)


def test_opening_split_audit():
    row = {**record_board(chess.Board()), "id": "a", "source_game_id": "source"}
    with pytest.raises(ValueError, match="leakage"):
        validate_groups({"train": [row], "final": [row]})


def tiny_config(initial, output):
    from chessengine.common import read_json

    cfg = read_json("configs/policy-value.json")
    return {
        **cfg,
        "initializer": str(initial),
        "output": str(output),
        "seeds": [42, 43, 44],
        "batches": 1,
        "games_per_batch": 1,
        "minimum_completed_per_batch": 1,
        "positions_per_game": 2,
        "batch_size": 1,
        "simulations": 128,
        "max_plies": 4,
        "development_batches": [1],
        "bootstrap_resamples": 20,
        "equal_time_ms": 2,
        "reference_nodes": 128,
        "development_pairs": 1,
        "final_pairs": 1,
        "control_pairs": 1,
        "reference_pairs": 1,
    }


def mate_opening(identity):
    return {
        **record_board(chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")),
        "id": identity,
        "source_game_id": identity,
    }


def test_uniform_game_sampling_truncation_and_resume(initial, tmp_path):
    from chessengine.pv_train import train, dataset
    from chessengine.pv_games import generate
    from chessengine.train import trusted_load
    from chessengine.common import digest

    cfg = tiny_config(initial, tmp_path / "full")
    opening = mate_opening("train")
    groups = {"train": [opening]}
    a = train(cfg, groups, "anchored", 42, "study")
    assert a["history"][0]["samples"] == 2 and a["history"][0]["completed"] == 1
    resumed = {**cfg, "output": str(tmp_path / "resumed")}
    assert train(resumed, groups, "anchored", 42, "study", stop_after_steps=1)["status"] == "interrupted"
    train(resumed, groups, "anchored", 42, "study")
    x = trusted_load(tmp_path / "full/anchored-42/resume.pt")
    y = trusted_load(tmp_path / "resumed/anchored-42/resume.pt")
    for key in x["model"]:
        torch.testing.assert_close(x["model"][key], y["model"][key], rtol=0, atol=0)
    assert x["optimizer_steps"] == y["optimizer_steps"] == 2
    with pytest.raises(ValueError, match="identity"):
        train({**resumed, "learning_rate": 0.01}, groups, "anchored", 42, "study")
    actor = PolicyValue(tmp_path / "full/anchored-42/actor-001.npz")
    game = generate(actor, opening, cfg, 77)
    assert not game["truncated"]
    d = dataset([game, {**game, "truncated": True}], cfg, 42, 1)
    assert len(d[0]) == cfg["positions_per_game"]
    with pytest.raises(ValueError, match="quality gate"):
        dataset([{**game, "truncated": True}], cfg, 42, 1)
    source_sha = digest(initial)
    train({**cfg, "output": str(tmp_path / "baseline")}, groups, "baseline", 42, "study")
    assert digest(initial) == source_sha


def test_tiny_six_run_selection_final_and_report(initial, tmp_path):
    from chessengine.pv_train import train
    from chessengine.pv_study import initialize, evaluate, report

    root = tmp_path / "study"
    root.mkdir()
    cfg = tiny_config(initial, root)
    groups = {key: [mate_opening(key)] for key in ["train", "development", "final", "control", "reference"]}
    for index, (key, rows) in enumerate(groups.items()):
        b = chess.Board(rows[0]["initial_fen"])
        b.set_piece_at(chess.A2 + index, chess.Piece(chess.PAWN, chess.WHITE))
        groups[key] = [{**record_board(b), "id": key, "source_game_id": key}]
    validate_groups(groups)
    init = initialize(cfg, root)
    for treatment in cfg["anchor_weights"]:
        for seed in cfg["seeds"]:
            train(cfg, groups, treatment, seed, "tiny-study")
    result = evaluate(cfg, groups, root, init, "tiny-study")
    assert len(result["selections"]) == 6 and len(result["seeds"]) == 3
    from chessengine.pv_audit import completed

    assert completed(cfg, groups, root)["complete"]
    assert result["aggregate"]["games"] == 6
    assert all(len(read_json(root / f"final-{s}/primary/matches.json")["games"]) == 2 for s in cfg["seeds"])
    write_json(root / "status.json", {"status": "complete"})
    evidence = report(cfg, tmp_path / "report")
    assert evidence["results"]["status"] == "complete"
    assert "Primary final result" in (tmp_path / "report/index.html").read_text()
    old = read_json(root / "selection-lock.json")
    old["anchored-42"]["selected"]["batch"] = 999
    write_json(root / "selection-lock.json", old)
    with pytest.raises(ValueError, match="identity"):
        evaluate(cfg, groups, root, init, "tiny-study")
