import json
from pathlib import Path

import chess
import numpy as np
import pytest

from chessengine.common import digest
from chessengine.model import build_model, export, NumpyEvaluator
from chessengine.evaluate import paired_summary
from chessengine import supervised_release as release


def test_development_release_gate_requires_complete_pairs_and_cap_sensitivity():
    summary = dict(games=80, pairs=40, score=0.5, capped_games=0, uncapped_score=0.5)
    assert release.release_eligible(summary)
    for change in [dict(games=78), dict(pairs=39), dict(score=0.49),
                   dict(capped_games=5), dict(uncapped_score=None), dict(uncapped_score=0.49)]:
        assert not release.release_eligible({**summary, **change})


def test_inference_export_preserves_predictions_and_omits_private_metadata(tmp_path, monkeypatch):
    source, target = tmp_path / "source.npz", tmp_path / "public.npz"
    export(build_model([128, 64]), source, dict(
        hidden=[128, 64], kind="supervised", evidence="supervised-200k-development",
        private_path="/Users/private/checkpoint", provenance={"private": "research"},
        config={"seed": 42}, data_sha256="d" * 64,
    ))
    monkeypatch.setattr(release, "CANDIDATE_SHA256", digest(source))
    assert release.export_inference(source, target) == digest(target)
    original, clean = NumpyEvaluator(source), NumpyEvaluator(target)
    for fen in [chess.STARTING_FEN, "7k/8/8/8/8/8/Q7/7K b - - 0 1"]:
        assert original(chess.Board(fen)) == clean(chess.Board(fen))
    with np.load(target, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
    assert meta["training_positions"] == 180302 and meta["training_seed"] == 42
    assert meta["feature_version"] == 1
    assert not {"private_path", "provenance", "config", "data_sha256"}.intersection(meta)
    monkeypatch.setattr(release, "CANDIDATE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="preselected"):
        release.export_inference(source, target)


def test_release_evidence_matches_candidate_and_replayed_games():
    root = Path(__file__).resolve().parents[1]
    evidence_path = root / 'docs/experiments/supervised-180k-release/evidence.json'
    if not evidence_path.exists():
        pytest.skip('The candidate has not passed its release screen')
    evidence = json.loads(evidence_path.read_text())
    games = json.loads(evidence_path.with_name('games.json').read_text())['games']
    assert release.release_eligible(paired_summary(games, seed=42)) == evidence['release_eligible']
    assert paired_summary(games, seed=42) == evidence['summary']
    for game in games:
        import io
        import chess.pgn
        pgn = chess.pgn.read_game(io.StringIO(game['pgn']))
        board = pgn.end().board()
        outcome = board.outcome(claim_draw=True)
        assert game['capped'] == (outcome is None)
        assert game['result'] == board.result(claim_draw=True)
        if outcome:
            color = game['candidate_color'] == 'white'
            assert game['score'] == (0.5 if outcome.winner is None else float(outcome.winner == color))
    model = NumpyEvaluator(root / 'artifacts/candidates/supervised-180k.npz')
    assert model.checkpoint_id == evidence['checkpoint_sha256']
    assert model.metadata['source_checkpoint_sha256'] == release.CANDIDATE_SHA256
    assert model.metadata['training_positions'] == evidence['training_positions'] == 180302
    if not evidence['release_eligible']:
        assert digest(root / 'artifacts/models/supervised.npz') == release.BASELINE_SHA256
        assert digest(root / 'web/public/models/supervised.npz') == release.BASELINE_SHA256
