"""Measured CPU/MPS throughput and game-outcome calibration, separate from model selection."""

import argparse
import time
import io
import chess
import chess.pgn
import numpy as np
from threadpoolctl import threadpool_limits
from chessengine.common import read_json, write_json, digest
from chessengine.model import NumpyEvaluator

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("output")
p.add_argument("--matches")
a = p.parse_args()
evaluator = NumpyEvaluator(a.checkpoint)
with threadpool_limits(limits=1):
    board = chess.Board()
    t = time.perf_counter()
    for _ in range(1000):
        evaluator(board)
    ms = time.perf_counter() - t
result = {"cpu_ms_per_position": ms, "checkpoint_sha256": digest(a.checkpoint)}
if a.matches:
    games = read_json(a.matches)["games"]
    pred = []
    out = []
    for row in games:
        if row["capped"]:
            continue
        game = chess.pgn.read_game(io.StringIO(row["pgn"]))
        board = game.board()
        candidate_color = row["candidate_color"] == "white"
        own_turns = 0
        for move in game.mainline_moves():
            sample = board.turn == candidate_color and own_turns % 2 == 0
            if board.turn == candidate_color:
                own_turns += 1
            if sample:
                pred.append((evaluator(board) + 1) / 2)
                out.append(row["score"])
            board.push(move)
    bins = []
    for lo in np.arange(0, 1, 0.1):
        ids = [i for i, x in enumerate(pred) if lo <= x < (lo + 0.1 if lo < 0.9 else 1.000001)]
        if ids:
            bins.append(
                {
                    "prediction": float(np.mean([pred[i] for i in ids])),
                    "observed": float(np.mean([out[i] for i in ids])),
                    "count": len(ids),
                }
            )
    result.update(
        {
            "bins": bins,
            "positions": len(pred),
            "note": "Expected-score calibration, not win probability; positions correlated within games.",
        }
    )
write_json(a.output, result)
print(result)
