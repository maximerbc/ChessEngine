import chess
import pytest
from chessengine.engine import Search, Handcrafted, MATE
from chessengine.features import encode, FEATURE_SIZE


def test_mate_and_sign():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    result = Search(Handcrafted(), nodes=2000).run(board)
    board.push_uci(result.move)
    assert board.is_checkmate()
    assert result.value > MATE - 10
    assert Search(Handcrafted()).run(board).value == -MATE


def test_budget_and_no_mutation():
    board = chess.Board()
    result = Search(Handcrafted(), nodes=1).run(board)
    assert result.nodes <= 1
    assert chess.Move.from_uci(result.move) in board.legal_moves
    assert not board.move_stack
    assert Search(Handcrafted(), time_ms=0).run(board).move is not None


def test_deadline_returns_last_completed_iteration(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("chessengine.engine.time.perf_counter", lambda: now[0])
    search = Search(Handcrafted(), nodes=30000, time_ms=3000)
    original = search.negamax

    def expire_next_iteration(board, depth, alpha, beta, ply):
        if ply == 0 and depth == 2:
            now[0] = 3.0
        return original(board, depth, alpha, beta, ply)

    monkeypatch.setattr(search, "negamax", expire_next_iteration)
    board = chess.Board()
    result = search.run(board)
    assert result.depth == 1
    assert result.elapsed_ms == 3000
    assert result.nodes < 30000
    assert chess.Move.from_uci(result.move) in board.legal_moves
    assert not board.move_stack


def test_repetition_history():
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        board.push_uci(move)
    plain = chess.Board(board.fen())
    assert Search.key(board) != Search.key(plain)
    assert Search(Handcrafted()).run(board).terminal == "1/2-1/2"
    assert Search(Handcrafted()).run(plain).terminal is None
    assert encode(board)[840] == 1


@pytest.mark.parametrize(
    "fen,move",
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
    ],
)
def test_special_moves(fen, move):
    board = chess.Board(fen)
    assert chess.Move.from_uci(move) in board.legal_moves
    board.push_uci(move)
    assert board.is_valid()
    assert encode(board).shape == (FEATURE_SIZE,)


def test_stalemate_and_move_clock():
    assert Search(Handcrafted()).run(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")).terminal == "1/2-1/2"
    assert Search(Handcrafted()).run(chess.Board("7k/8/6K1/8/8/8/8/R7 w - - 100 70")).terminal == "1/2-1/2"
