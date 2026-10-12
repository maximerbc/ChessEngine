from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from .features import encode, FEATURE_VERSION, feature_size
from .common import digest

CHECKPOINT_VERSION = 1


class NumpyEvaluator:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            if (
                self.metadata["format_version"] != CHECKPOINT_VERSION
            ):
                raise ValueError("Unsupported checkpoint/feature schema")
            self.feature_version = self.metadata['feature_version']
            widths = [feature_size(self.feature_version), *self.metadata["hidden"], 1]
            self.layers = []
            for i, (left, right) in enumerate(zip(widths, widths[1:])):
                w, b = data[f"w{i}"].copy(), data[f"b{i}"].copy()
                if (
                    w.shape != (right, left)
                    or b.shape != (right,)
                    or not np.isfinite(w).all()
                    or not np.isfinite(b).all()
                ):
                    raise ValueError("Invalid model weights")
                self.layers.append((w, b))
        self.checkpoint_id = digest(path)

    def batch(self, x):
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            x = np.maximum(x, 0) if i < len(self.layers) - 1 else np.tanh(x)
        return x.squeeze(-1)

    def __call__(self, board):
        return float(self.batch(encode(board, self.feature_version)))


def build_model(hidden=(128, 64), dropout=0.0, feature_version=1):
    import torch.nn as nn

    if not 0 <= dropout < 1:
        raise ValueError("Dropout must be in [0, 1)")
    widths = [feature_size(feature_version), *hidden, 1]
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers += [nn.Linear(a, b), nn.ReLU() if i < len(widths) - 2 else nn.Tanh()]
        if dropout and i < len(widths) - 2:
            layers.append(nn.Dropout(dropout))
    model = nn.Sequential(*layers)
    model.feature_version = feature_version
    return model


def export(model, path, metadata):
    import torch.nn as nn

    arrays = {}
    for i, layer in enumerate(m for m in model if isinstance(m, nn.Linear)):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy()
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
    version = getattr(model, 'feature_version', FEATURE_VERSION)
    if metadata.get('feature_version', version) != version:
        raise ValueError('Model/metadata feature mismatch')
    metadata = {**metadata, "feature_version": version, "format_version": CHECKPOINT_VERSION}
    arrays["metadata"] = np.array(json.dumps(metadata))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_export(path):
    import torch

    evaluator = NumpyEvaluator(path)
    model = build_model(evaluator.metadata["hidden"], evaluator.metadata.get("config", {}).get("dropout", 0),
                        evaluator.feature_version)
    with torch.no_grad():
        for layer, (w, b) in zip((m for m in model if isinstance(m, torch.nn.Linear)), evaluator.layers):
            layer.weight.copy_(torch.from_numpy(w))
            layer.bias.copy_(torch.from_numpy(b))
    return model, evaluator.metadata
