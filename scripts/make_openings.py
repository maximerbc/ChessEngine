"""Reserve opening source games before preparing supervised positions."""

import argparse
import random
import chess.pgn
from chessengine.data import open_pgn
from chessengine.common import json_hash, write_json, digest
from chessengine.features import record_board, position_key

p = argparse.ArgumentParser()
p.add_argument("source")
p.add_argument("output")
p.add_argument("--seed", type=int, default=2026)
a = p.parse_args()
rows = []
seen = set()
with open_pgn(a.source) as stream:
    while len(rows) < 480:
        game = chess.pgn.read_game(stream)
        if game is None:
            raise ValueError("Not enough opening games")
        moves = list(game.mainline_moves())
        if game.errors or len(moves) < 16:
            continue
        board = game.board()
        game_id = json_hash({"initial": board.fen(), "moves": [m.uci() for m in moves]})
        for move in moves[:12]:
            board.push(move)
        key = position_key(board)
        if key in seen or board.is_game_over(claim_draw=True):
            continue
        seen.add(key)
        rows.append({**record_board(board), "id": game_id[:12], "source_game_id": game_id})
random.Random(a.seed).shuffle(rows)
write_json(a.output, {"train": rows[:240], "development": rows[240:280], "test": rows[280:]})
write_json(
    a.output + ".manifest.json",
    {
        "source": a.source,
        "source_sha256": digest(a.source),
        "seed": a.seed,
        "note": "All source games excluded from supervised data; 12-ply starts.",
    },
)
print("Reserved 240 training, 40 development, 200 test openings")
