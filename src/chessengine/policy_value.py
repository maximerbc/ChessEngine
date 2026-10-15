"""Research-only shared-trunk policy/value model; NumPy inference, explicit schema."""

import json
from pathlib import Path
import numpy as np
from .common import digest
from .features import encode
from .ranking import move_features

FORMAT = "policy-value-v1"


def build(initializer, policy_seed=20261015):
    import torch
    from torch import nn
    from .model import load_export

    initial, metadata = load_export(initializer)
    if metadata["feature_version"] != 3 or metadata["hidden"] != [128, 64]:
        raise ValueError("Policy/value initialization requires the 128/64 feature-v3 model")

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(853, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
            self.value = nn.Linear(64, 1)
            self.policy = nn.Linear(64 + 133, 32)
            self.policy_out = nn.Linear(32, 1)

        def forward(self, x, moves, mask):
            context = self.encoder(x)
            base = torch.nn.functional.linear(context, self.policy.weight[:, :64], self.policy.bias)
            embedded = torch.nn.functional.embedding(moves, self.policy.weight[:, 64:].T).sum(dim=-2)
            logits = self.policy_out(torch.relu(base[:, None, :] + embedded)).squeeze(-1)
            return logits.masked_fill(~mask, -float("inf")), torch.tanh(self.value(context)).squeeze(-1)

    with torch.random.fork_rng():
        torch.manual_seed(policy_seed)
        model = Network()
    with torch.no_grad():
        for target, source in zip(
            [model.encoder[0], model.encoder[2], model.value], [initial[0], initial[2], initial[4]]
        ):
            target.weight.copy_(source.weight)
            target.bias.copy_(source.bias)
    return model


def export(model, path, metadata):
    arrays = {}
    for name, layer in [
        ("e0", model.encoder[0]),
        ("e1", model.encoder[2]),
        ("v", model.value),
        ("p", model.policy),
        ("o", model.policy_out),
    ]:
        arrays[name + "w"] = layer.weight.detach().cpu().numpy()
        arrays[name + "b"] = layer.bias.detach().cpu().numpy()
    meta = {
        **metadata,
        "format": FORMAT,
        "feature_version": 3,
        "move_schema": 1,
        "hidden": [128, 64],
        "policy_hidden": 32,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    temp.replace(path)


class PolicyValue:
    def __init__(self, path):
        self.checkpoint_id = digest(path)
        self.path = str(path)
        with np.load(path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive["metadata"]))
            if any(
                self.metadata.get(k) != v
                for k, v in {
                    "format": FORMAT,
                    "feature_version": 3,
                    "move_schema": 1,
                    "hidden": [128, 64],
                    "policy_hidden": 32,
                }.items()
            ):
                raise ValueError("Unsupported policy/value schema")
            shapes = {
                "e0w": (128, 853),
                "e0b": (128,),
                "e1w": (64, 128),
                "e1b": (64,),
                "vw": (1, 64),
                "vb": (1,),
                "pw": (32, 197),
                "pb": (32,),
                "ow": (1, 32),
                "ob": (1,),
            }
            self.arrays = {}
            for key, shape in shapes.items():
                a = archive[key].copy()
                if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                    raise ValueError("Invalid policy/value weights")
                self.arrays[key] = a

    def arrays_forward(self, x, moves):
        a = self.arrays
        h = np.maximum(x @ a["e0w"].T + a["e0b"], 0)
        h = np.maximum(h @ a["e1w"].T + a["e1b"], 0)
        v = np.tanh(h @ a["vw"].T + a["vb"]).squeeze(-1)
        base = h @ a["pw"][:, :64].T + a["pb"]
        hidden = np.maximum(base[..., None, :] + a["pw"][:, 64:].T[moves].sum(axis=-2), 0)
        return (hidden @ a["ow"].T + a["ob"]).squeeze(-1), v

    def predict(self, board):
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("Policy inference requires a nonterminal position")
        logits, value = self.arrays_forward(encode(board, 3), move_features(board, moves))
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return moves, probabilities, float(value)


def load(path, initializer):
    import torch

    evaluator = PolicyValue(path)
    model = build(initializer)
    with torch.no_grad():
        for name, layer in [
            ("e0", model.encoder[0]),
            ("e1", model.encoder[2]),
            ("v", model.value),
            ("p", model.policy),
            ("o", model.policy_out),
        ]:
            layer.weight.copy_(torch.from_numpy(evaluator.arrays[name + "w"]))
            layer.bias.copy_(torch.from_numpy(evaluator.arrays[name + "b"]))
    return model


def objective(logits, values, targets, outcomes, initial_values, anchor_weight):
    import torch

    if anchor_weight not in (0, 0.1):
        raise ValueError("Only the two preregistered anchor treatments are supported")
    # Masked logits contain -inf: select zero-target terms before multiplication.
    log_p = torch.log_softmax(logits, dim=-1)
    policy = -(targets * log_p.masked_fill(targets == 0, 0)).sum(dim=-1)
    value = (values - outcomes).square()
    anchor = (values - initial_values.detach()).square() if anchor_weight else torch.zeros_like(value)
    return (policy + value + anchor_weight * anchor).mean()
