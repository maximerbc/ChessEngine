"""Skripkon-style human-move CNN. Torch trains; NumPy serves direct legal moves."""
from pathlib import Path
import json
import time
import chess
import numpy as np
from .common import digest, json_hash

REFERENCE = "97238ead1892adca5e8626c568cd8129d5993ee9"
STRATEGY = "direct-policy-v1"
MODEL_ID = "elite-policy"


def vocabulary():
    moves = {chess.square_name(a) + chess.square_name(b)
             for a in chess.SQUARES for b in chess.SQUARES if a != b}
    for rank, target in [(6, 7), (1, 0)]:
        for file in range(8):
            for end in range(max(0, file - 1), min(8, file + 2)):
                for promotion in "qrbn":
                    moves.add(chess.square_name(chess.square(file, rank))
                              + chess.square_name(chess.square(end, target)) + promotion)
    return tuple(sorted(moves))


MOVES = vocabulary()
MOVE_IDS = {move: i for i, move in enumerate(MOVES)}
SCHEMA = dict(format="elite-human-cnn-v1", encoder="absolute-13-legal-destinations-v1",
              classes=4208, vocabulary_sha256=json_hash(MOVES), channels=[64, 128],
              hidden=256, reference_commit=REFERENCE, label_source="human-played-move")
SHAPES = {"c0w": (64, 13, 3, 3), "c0b": (64,), "c1w": (128, 64, 3, 3), "c1b": (128,),
          "fw": (256, 8192), "fb": (256,), "ow": (4208, 256), "ob": (4208,)}


def encode_board(board, legal=None):
    """Exactly the reference plane layout: a1=(0,0), white before black."""
    x = np.zeros((13, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        x[piece.piece_type - 1 + (0 if piece.color else 6), square // 8, square % 8] = 1
    for move in board.legal_moves if legal is None else legal:
        x[12, move.to_square // 8, move.to_square % 8] = 1
    return x


def build_policy():
    import torch.nn as nn

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.c0 = nn.Conv2d(13, 64, 3, padding=1)
            self.c1 = nn.Conv2d(64, 128, 3, padding=1)
            self.context = nn.Linear(8192, 256)
            self.output = nn.Linear(256, len(MOVES))
            nn.init.kaiming_uniform_(self.c0.weight, nonlinearity="relu")
            nn.init.kaiming_uniform_(self.c1.weight, nonlinearity="relu")
            nn.init.xavier_uniform_(self.context.weight)
            nn.init.xavier_uniform_(self.output.weight)

        def forward(self, x):
            x = self.c1(self.c0(x).relu()).relu().flatten(1)
            return self.output(self.context(x).relu())

    return Policy()


def export_policy(model, path, metadata):
    arrays = {}
    for prefix, layer in [("c0", model.c0), ("c1", model.c1), ("f", model.context), ("o", model.output)]:
        arrays[prefix + "w"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[prefix + "b"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Nonfinite elite policy weights")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps({**metadata, **SCHEMA,
                              "kind": "human-move-policy"}, sort_keys=True)), **arrays)
    path.with_suffix(".tmp").replace(path)


def convolution(x, weight, bias):
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(x, ((0, 0), (1, 1), (1, 1))), (3, 3), axis=(1, 2))
    columns = windows.transpose(0, 3, 4, 1, 2).reshape(weight.shape[1] * 9, 64)
    return (weight.reshape(len(weight), -1) @ columns + bias[:, None]).reshape(-1, 8, 8)


class NumpyElitePolicy:
    def __init__(self, path):
        self.checkpoint_id = digest(path)
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            if any(self.metadata.get(k) != v for k, v in SCHEMA.items()):
                raise ValueError("Unsupported elite policy schema/vocabulary")
            if self.metadata.get("kind") != "human-move-policy" or set(data.files) != {"metadata", *SHAPES}:
                raise ValueError("Invalid elite policy identity/arrays")
            self.arrays = {}
            for key, shape in SHAPES.items():
                a = data[key]
                if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                    raise ValueError("Invalid elite policy weights")
                self.arrays[key] = a.copy()

    def logits(self, x):
        x = np.asarray(x)
        if x.shape != (13, 8, 8) or not np.isfinite(x).all() or not np.isin(x, [0, 1]).all():
            raise ValueError("Expected binary 13-plane elite input")
        a = self.arrays
        z = np.maximum(convolution(x.astype(np.float32), a["c0w"], a["c0b"]), 0)
        z = np.maximum(convolution(z, a["c1w"], a["c1b"]), 0)
        z = np.maximum(a["fw"] @ z.ravel() + a["fb"], 0)
        logits = np.einsum("mh,h->m", a["ow"], z, optimize=False) + a["ob"]
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite elite inference")
        return logits

    def rank(self, board):
        if board.is_game_over(claim_draw=True):
            return []
        moves = sorted(board.legal_moves, key=lambda move: move.uci())
        logits = self.logits(encode_board(board, moves))[[MOVE_IDS[m.uci()] for m in moves]]
        probabilities = np.exp(logits.astype(np.float64) - float(logits.max()))
        probabilities /= probabilities.sum()
        return [dict(move=moves[i].uci(), probability=float(probabilities[i]))
                for i in np.argsort(-logits, kind="stable")]

    def choose(self, board):
        ranked = self.rank(board)
        return ranked[0]["move"] if ranked else None

    def analyze(self, board):
        start = time.perf_counter()
        ranked = self.rank(board)
        move = ranked[0]["move"] if ranked else None
        resulting = board.copy(stack=True)
        if move:
            resulting.push_uci(move)
        return dict(best_move=move, move=move, resulting_fen=resulting.fen(),
                    terminal=resulting.result(claim_draw=True) if resulting.is_game_over(claim_draw=True) else None,
                    value=None, position_value=None, mate_score=False, value_perspective=None,
                    depth=None, nodes=None, simulations=None, search=None, pv=[],
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    strategy=STRATEGY, root_rankings=ranked,
                    explanation=dict(raw_value=None, feature_version=None,
                                     side_to_move="white" if board.turn else "black", piece_counts={},
                                     label_source="elite human games", encoder=SCHEMA["encoder"]))


def validate_release(row, policy, directory):
    """Experimental access requires real validation/export evidence for fixed seed 42."""
    if (row.get("kind") != "human-move-policy" or row.get("search") != {"strategy": STRATEGY}
            or row.get("architecture") != SCHEMA or row.get("experimental") is not True
            or row.get("qualified") is not False or row.get("strength_status") != "not-established"
            or row.get("authorization_id") != "user-20261124-elite-play"
            or row.get("evidence_id") != "elite-pytorch-baseline-20261124"
            or policy.metadata.get("seed") != 42
            or policy.metadata.get('study') != 'elite-pytorch-baseline-20261124'
            or policy.metadata.get('training_positions') != 800000
            or row.get('dataset_sha256') != policy.metadata.get('dataset_sha256')):
        raise ValueError("Elite experimental release identity mismatch")
    name = row.get("gate_file", "")
    if not name or Path(name).name != name:
        raise ValueError("Elite release needs a local validation gate")
    path = Path(directory) / name
    if digest(path) != row.get("gate_sha256"):
        raise ValueError("Elite gate checksum mismatch")
    gate = json.loads(path.read_text())
    if (gate.get("passed") is not True or gate.get("checkpoint_sha256") != policy.checkpoint_id
            or gate.get("dataset_sha256") != policy.metadata.get("dataset_sha256")
            or gate.get("seed") != 42 or gate.get("export_passed") is not True
            or gate.get("validation_top1_difference_ci", [0])[0] <= 0):
        raise ValueError("Elite policy did not pass validation/export gate")
