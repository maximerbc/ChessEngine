from pathlib import Path
import json
import chess
import numpy as np
from chessengine.common import digest
from chessengine.model import NumpyEvaluator
from chessengine.evaluate import paired_summary


def test_release_models_and_recorded_results_are_consistent():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "artifacts/models/manifest.json"
    if not manifest_path.exists():
        import pytest

        pytest.skip("Release artifacts are delivered in the showcase batch")
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["models"]:
        if entry.get("kind") == "move-guided":
            from chessengine.ranking import NumpyRanker, GuidedEvaluator
            rank = manifest_path.parent / entry["ranker_file"]
            assert digest(rank) == entry["ranker_sha256"]
            assert digest(root / "web/public/models" / entry["ranker_file"]) == entry["ranker_sha256"]
            value = NumpyEvaluator(manifest_path.parent / "supervised.npz")
            assert GuidedEvaluator(value, NumpyRanker(rank)).checkpoint_id == entry["sha256"]
            continue
        path = manifest_path.parent / entry["file"]
        assert digest(path) == entry["sha256"]
        assert digest(root / "web/public/models" / entry["file"]) == entry["sha256"]
        assert np.isfinite(NumpyEvaluator(path)(chess.Board()))
    report = json.loads((root / "web/public/research/results.json").read_text())
    for row in report["aggregate"]["runs"]:
        games = json.loads((root / f"web/public/research/games-seed-{row['seed']}.json").read_text())["games"]
        assert paired_summary(games, repetitions=100)["score"] == row["summary"]["score"]
        assert len(games) == 400
        assert [point["selfplay_games"] for point in row["curve"]] == [20, 60]
    assert (root / "web/public/research/report.pdf").read_bytes().startswith(b"%PDF")
