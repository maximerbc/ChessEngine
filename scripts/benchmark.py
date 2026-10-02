"""Measure search and training devices before fixing a compute budget."""

import argparse
import time
import chess
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from chessengine.common import write_json, provenance
from chessengine.features import encode
from chessengine.model import NumpyEvaluator, load_export
from chessengine.engine import Search

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("output")
a = p.parse_args()
rows = []
with threadpool_limits(limits=1):
    torch.set_num_threads(1)
    ev = NumpyEvaluator(a.checkpoint)
    for nodes in [48, 128, 1000]:
        t = time.perf_counter()
        total = 0
        for fen in [chess.STARTING_FEN, "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"]:
            result = Search(ev, nodes=nodes, qdepth=2).run(chess.Board(fen))
            total += result.nodes
        elapsed = time.perf_counter() - t
        rows.append(
            {"nodes_budget": nodes, "seconds_per_move": elapsed / 2, "nodes_per_second": total / elapsed}
        )
    devices = []
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        model, _ = load_export(a.checkpoint)
        model.to(device)
        x = torch.from_numpy(np.stack([encode(chess.Board())] * 256)).to(device)
        for _ in range(3):
            model(x).sum().backward()
            model.zero_grad()
        if device == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            model(x).sum().backward()
            model.zero_grad()
        if device == "mps":
            torch.mps.synchronize()
        devices.append({"device": device, "batch_forward_backward_ms": (time.perf_counter() - t) * 1000 / 20})
write_json(a.output, {"search": rows, "training": devices, "provenance": provenance()})
print(rows, devices)
