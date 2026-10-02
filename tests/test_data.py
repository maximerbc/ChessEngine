import chess
import chess.pgn
from chessengine.data import prepare, records
from chessengine.common import json_hash, write_json
from chessengine.features import record_board


def test_reserved_games_are_excluded_before_sampling(tmp_path):
    board = chess.Board()
    moves = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8".split()
    for move in moves:
        board.push_uci(move)
    pgn = tmp_path / "source.pgn"
    pgn.write_text(str(chess.pgn.Game.from_board(board)) + "\n\n")
    output = tmp_path / "positions.jsonl"
    manifest = prepare(str(pgn), output, limit=10, reserved=None)
    assert manifest["positions"] > 0
    game_id = json_hash({"initial": chess.STARTING_FEN, "moves": moves})
    reserved = tmp_path / "reserved.json"
    write_json(
        reserved,
        {"train": [], "development": [], "test": [{**record_board(board), "source_game_id": game_id}]},
    )
    manifest = prepare(str(pgn), output, limit=10, reserved=reserved)
    assert manifest["positions"] == 0


def test_repeated_source_game_is_not_duplicated(tmp_path):
    board = chess.Board()
    for uci in "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5".split():
        board.push_uci(uci)
    game = str(chess.pgn.Game.from_board(board))
    pgn = tmp_path / "source.pgn"
    pgn.write_text(game + "\n\n" + game + "\n\n")
    out = tmp_path / "out.jsonl"
    prepare(str(pgn), out, limit=100, reserved=None)
    rows = list(records(out))
    keys = [" ".join(r["fen"].split()[:4]) for r in rows]
    assert len(set(keys)) == len(keys)
    assert len({r["game_id"] for r in rows}) == 1
