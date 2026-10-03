import chess
import numpy as np
import torch
from chessengine.features import encode
from chessengine.model import build_model, export, NumpyEvaluator


def test_portable_model_matches_training_model(tmp_path):
    torch.manual_seed(42)
    model = build_model([16, 8])
    path = tmp_path / "model.npz"
    export(model, path, {"hidden": [16, 8]})
    board = chess.Board()
    x = encode(board)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).item()
    np.testing.assert_allclose(NumpyEvaluator(path)(board), expected, atol=1e-6)
