"""Schema v1: absolute piece planes, turn, rights, EP, clock, repetition."""

import chess
import numpy as np

FEATURE_VERSION = 1
FEATURE_SIZE = 841  # 12*64 + turn + 4 rights + 65 EP + clock + 2 repetition


def encode(board: chess.Board) -> np.ndarray:
    x = np.zeros(FEATURE_SIZE, dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = (0 if piece.color == chess.WHITE else 6) + piece.piece_type - 1
        x[plane * 64 + square] = 1
    x[768] = float(board.turn)
    x[769:773] = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]
    x[773 + (board.ep_square if board.ep_square is not None else 64)] = 1
    x[838] = min(board.halfmove_clock / 150, 1)
    x[839] = board.is_repetition(2)
    x[840] = board.is_repetition(3)
    return x


def position_key(board: chess.Board) -> str:
    """Conservative leakage key: ignore clocks, retain effective position state."""
    return " ".join(board.fen().split()[:4])


def restore(record: dict) -> chess.Board:
    board = chess.Board(record.get("initial_fen", chess.STARTING_FEN))
    for uci in record.get("moves", []):
        board.push_uci(uci)
    if "fen" in record and board.fen() != record["fen"]:
        if "moves" in record:
            raise ValueError("Stored FEN disagrees with move history")
        board = chess.Board(record["fen"])
    return board


def record_board(board: chess.Board) -> dict:
    return {
        "initial_fen": board.root().fen(),
        "moves": [m.uci() for m in board.move_stack],
        "fen": board.fen(),
    }
