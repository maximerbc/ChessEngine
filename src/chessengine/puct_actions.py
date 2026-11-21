"""Pinned PGX action labels, usable by CPU inference without importing PGX/JAX."""
from functools import lru_cache
import json
from pathlib import Path
import chess


@lru_cache(maxsize=1)
def tables():
    value = json.loads(Path(__file__).with_name('puct_action_tables.json').read_text())
    if value['schema'] != 'pgx-chess-2.6.0-actions-v1':
        raise ValueError('Unsupported action mapping schema')
    return value


def action_map(board):
    table = tables()
    mapping = {}
    for move in board.legal_moves:
        def relative(square):
            file, rank = chess.square_file(square), chess.square_rank(square)
            return file * 8 + (rank if board.turn else 7 - rank)
        origin, target = relative(move.from_square), relative(move.to_square)
        if move.promotion and move.promotion != chess.QUEEN:
            under = [chess.ROOK, chess.BISHOP, chess.KNIGHT].index(move.promotion)
            candidates = [p for p in range(under * 3, under * 3 + 3)
                          if table['from_plane'][origin][p] == target]
            if len(candidates) != 1:
                raise ValueError('Invalid underpromotion action')
            plane = candidates[0]
        else:
            plane = table['to_plane'][origin][target]
        label = origin * 73 + plane
        if plane < 0 or label in mapping:
            raise ValueError('Invalid or duplicate action mapping')
        mapping[label] = move
    return mapping
