"""Research-only direct legal-move CNN; inference requires NumPy, never torch."""
from pathlib import Path
import json
import chess
import numpy as np
from .features import encode
from .ranking import move_features

FORMAT = "direct-move-cnn-v1"
SCHEMA = {"format": FORMAT, "feature_version": 3, "move_schema": 1,
          "channels": [64, 128], "context": 256, "move_hidden": 128}
SHAPES = {"c0w": (64, 12, 3, 3), "c0b": (64,), "c1w": (128, 64, 3, 3),
          "c1b": (128,), "fw": (256, 8192 + 85), "fb": (256,),
          "hw": (128, 256 + 133), "hb": (128,), "ow": (1, 128), "ob": (1,)}


def build_policy():
    import torch
    from torch import nn

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.c0 = nn.Conv2d(12, 64, 3, padding=1)
            self.c1 = nn.Conv2d(64, 128, 3, padding=1)
            self.context = nn.Linear(8192 + 85, 256)
            self.head = nn.Linear(256 + 133, 128)
            self.output = nn.Linear(128, 1)

        def forward(self, x, moves, mask):
            planes = x[:, :768].reshape(-1, 12, 8, 8)
            planes = torch.relu(self.c1(torch.relu(self.c0(planes)))).flatten(1)
            context = torch.relu(self.context(torch.cat((planes, x[:, 768:]), dim=1)))
            base = nn.functional.linear(context, self.head.weight[:, :256], self.head.bias)
            embedded = nn.functional.embedding(moves, self.head.weight[:, 256:].T).sum(-2)
            logits = self.output(torch.relu(base[:, None, :] + embedded)).squeeze(-1)
            return logits.masked_fill(~mask, -float("inf"))

    return Policy()


def export_policy(model, path, metadata):
    arrays = {}
    for prefix, layer in [("c0", model.c0), ("c1", model.c1), ("f", model.context),
                          ("h", model.head), ("o", model.output)]:
        arrays[prefix + "w"] = layer.weight.detach().cpu().numpy()
        arrays[prefix + "b"] = layer.bias.detach().cpu().numpy()
    meta = {**metadata, **SCHEMA, "kind": "research-direct-move-policy"}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    temp.replace(path)


def convolution(x, weight, bias):
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(x, ((0, 0), (1, 1), (1, 1))), (3, 3), axis=(1, 2))
    return np.einsum("cyxij,ocij->oyx", windows, weight, optimize=True) + bias[:, None, None]


class NumpyMovePolicy:
    def __init__(self, path):
        from .common import digest
        self.checkpoint_id = digest(path)
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            if any(self.metadata.get(k) != v for k, v in SCHEMA.items()):
                raise ValueError("Unsupported direct-move policy schema")
            if self.metadata.get("kind") != "research-direct-move-policy":
                raise ValueError("Invalid policy kind")
            if set(data.files) != {"metadata", *SHAPES}:
                raise ValueError("Invalid policy arrays")
            self.arrays = {}
            for key, shape in SHAPES.items():
                array = data[key]
                if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError("Invalid policy weights")
                self.arrays[key] = array.copy()

    def logits(self, x, moves):
        x = np.asarray(x, dtype=np.float32)
        moves = np.asarray(moves)
        if x.shape != (853,) or not np.isfinite(x).all():
            raise ValueError("Expected feature-v3 position")
        if (moves.ndim != 2 or moves.shape[1] != 3 or moves.dtype.kind not in "iu"
                or np.any(moves < [0, 64, 128]) or np.any(moves >= [64, 128, 133])):
            raise ValueError("Invalid move features")
        a = self.arrays
        z = np.maximum(convolution(x[:768].reshape(12, 8, 8), a["c0w"], a["c0b"]), 0)
        z = np.maximum(convolution(z, a["c1w"], a["c1b"]), 0)
        context = np.maximum(a["fw"] @ np.concatenate((z.ravel(), x[768:])) + a["fb"], 0)
        base = a["hw"][:, :256] @ context + a["hb"]
        hidden = np.maximum(base[None, :] + a["hw"][:, 256:].T[moves].sum(1), 0)
        # Reduce each row identically: BLAS row tiling can invent move preferences
        # when large, identical activations differ only through rounding.
        return np.einsum("mh,h->m", hidden, a["ow"][0], optimize=False) + a["ob"][0]

    def rank(self, board: chess.Board):
        if board.is_game_over(claim_draw=True):
            return []
        moves = list(board.legal_moves)
        logits = self.logits(encode(board, 3), move_features(board, moves))
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return [{"move": moves[i].uci(), "probability": float(probabilities[i])}
                for i in np.argsort(-logits, kind="stable")]

    def choose(self, board):
        ranked = self.rank(board)
        return ranked[0]["move"] if ranked else None
