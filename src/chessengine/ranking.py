"""Versioned legal-move scorer; NumPy runtime has no training dependency."""

import json
from pathlib import Path
import chess
import numpy as np
from .common import digest
from .features import encode

PROMOTIONS = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def move_features(board, moves):
    flip = 0 if board.turn else 56
    return np.array(
        [
            [m.from_square ^ flip, 64 + (m.to_square ^ flip), 128 + PROMOTIONS.index(m.promotion)]
            for m in moves
        ],
        dtype=np.int64,
    ).reshape(-1, 3)


class NumpyRanker:
    def __init__(self, path):
        self.checkpoint_id = digest(path)
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            if (
                self.metadata.get("format") != "move-ranker-v1"
                or self.metadata.get("feature_version") != 3
                or self.metadata.get("move_schema") != 1
            ):
                raise ValueError("Unsupported move ranker schema")
            h = self.metadata["board_hidden"]
            m = self.metadata["move_hidden"]
            shapes = {
                "e0w": (h[0], 853),
                "e0b": (h[0],),
                "e1w": (h[1], h[0]),
                "e1b": (h[1],),
                "hw": (m, h[1] + 133),
                "hb": (m,),
                "ow": (1, m),
                "ob": (1,),
            }
            self.arrays = {}
            for name, shape in shapes.items():
                x = data[name].copy()
                if x.shape != shape or not np.isfinite(x).all():
                    raise ValueError("Invalid ranker weights")
                self.arrays[name] = x

    def logits(self, x, moves):
        a = self.arrays
        context = np.maximum(x @ a["e0w"].T + a["e0b"], 0)
        context = np.maximum(context @ a["e1w"].T + a["e1b"], 0)
        width = context.shape[-1]
        base = context @ a["hw"][:, :width].T + a["hb"]
        hidden = np.maximum(base[..., None, :] + a["hw"][:, width:].T[moves].sum(axis=-2), 0)
        return (hidden @ a["ow"].T + a["ob"]).squeeze(-1)

    def rank(self, board):
        moves = list(board.legal_moves)
        if not moves:
            return []
        scores = self.logits(encode(board, 3), move_features(board, moves))
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        order = np.argsort(-scores, kind="stable")
        return [{"move": moves[i].uci(), "probability": float(probabilities[i])} for i in order]


def build_ranker(baseline):
    import torch
    from torch import nn
    from .model import load_export

    value, metadata = load_export(baseline)
    hidden = metadata["hidden"]
    if getattr(value, "feature_version", None) != 3:
        raise ValueError("Move ranker requires feature-v3 initializer")

    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(853, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU()
            )
            self.head = nn.Linear(hidden[1] + 133, 32)
            self.output = nn.Linear(32, 1)
            self.board_hidden = hidden
            with torch.no_grad():
                for target, source in zip([self.encoder[0], self.encoder[2]], [value[0], value[2]]):
                    target.weight.copy_(source.weight)
                    target.bias.copy_(source.bias)

        def forward(self, x, moves, mask):
            context = self.encoder(x)
            width = context.shape[-1]
            base = torch.nn.functional.linear(context, self.head.weight[:, :width], self.head.bias)
            embedded = torch.nn.functional.embedding(moves, self.head.weight[:, width:].T).sum(dim=-2)
            logits = self.output(torch.relu(base[:, None, :] + embedded)).squeeze(-1)
            return logits.masked_fill(~mask, -float("inf"))

    return Ranker()


def export_ranker(model, path, metadata):
    arrays = {}
    for prefix, layer in [
        ("e0", model.encoder[0]),
        ("e1", model.encoder[2]),
        ("h", model.head),
        ("o", model.output),
    ]:
        arrays[prefix + "w"] = layer.weight.detach().cpu().numpy()
        arrays[prefix + "b"] = layer.bias.detach().cpu().numpy()
    meta = {
        **metadata,
        "format": "move-ranker-v1",
        "feature_version": 3,
        "move_schema": 1,
        "board_hidden": model.board_hidden,
        "move_hidden": 32,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=np.array(json.dumps(meta)), **arrays)


class GuidedEvaluator:
    """Compose a frozen value evaluator with a separate root move scorer."""

    def __init__(self, value, ranker):
        from .common import json_hash

        self.value = value
        self.ranker = ranker
        self.checkpoint_id = json_hash({"value": value.checkpoint_id, "ranker": ranker.checkpoint_id})
        self.metadata = {**value.metadata, "kind": "move-guided"}
        self.feature_version = value.feature_version

    def __call__(self, board):
        return self.value(board)


def load_guided(path):
    from .common import read_json
    from .model import NumpyEvaluator

    path = Path(path)
    bundle = read_json(path)
    value_path = path.parent / bundle["value"]
    rank_path = path.parent / bundle["ranker"]
    if digest(value_path) != bundle["value_sha256"] or digest(rank_path) != bundle["ranker_sha256"]:
        raise ValueError("Guided component checksum mismatch")
    return GuidedEvaluator(NumpyEvaluator(value_path), NumpyRanker(rank_path))
