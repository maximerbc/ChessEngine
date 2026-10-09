import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chessengine.common import read_json
from chessengine.diagnostics import (
    development_rows, grouped_bootstrap, metrics, nested_subsets, run_study, screen, strata, summarize_runs,
)
from chessengine.features import encode
from chessengine.model import build_model, export, load_export, NumpyEvaluator
from chessengine.train import fit, seed_all
from test_learning import dataset, config


def test_subsets_whole_games_nested_and_test_exclusion(tmp_path):
    rows = [{"game_id": str(i // 3), "split": "train"} for i in range(15)]
    rows += [{"game_id": "test", "split": "test"}, {"game_id": "val", "split": "validation"}]
    subsets = nested_subsets(rows, [4, 8, "all"], 2026)
    assert [len(v) for v in subsets.values()] == [6, 9, 15]
    assert set(subsets["4"]) < set(subsets["8"]) < set(subsets["all"])
    assert subsets == nested_subsets(rows, [4, 8, "all"], 2026)
    for ids in subsets.values():
        for game in {rows[i]["game_id"] for i in ids}:
            assert sum(rows[i]["game_id"] == game for i in ids) == 3
    source = tmp_path / "data.jsonl"
    original = dataset(source)
    # Even deliberately invalid test records must not enter the diagnostic path.
    with source.open("a") as stream:
        stream.write(json.dumps({"split": "test", "target": "not inspected"}) + "\n")
    assert development_rows(source) == original


def test_metrics_and_grouped_bootstrap():
    assert metrics([1, -1], [0, 0]) == {"mse": 1, "mae": 1, "signed_error": 0}
    targets = np.zeros(4)
    predictions = {"supervised": np.array([0., 0., 0., 2.]), "zero": np.zeros(4)}
    a = grouped_bootstrap(predictions, targets, ["a", "a", "a", "b"], 2000, 1)
    assert a == grouped_bootstrap(predictions, targets, ["a", "a", "a", "b"], 2000, 1)
    assert a["intervals"]["supervised"]["mse"] == [0, 4]
    assert a["paired_supervised_minus_baseline"]["zero"]["mse"] == [0, 4]
    predictions["zero"] = predictions["supervised"].copy()
    assert grouped_bootstrap(predictions, targets, ["a", "a", "a", "b"], 30)[
        "paired_supervised_minus_baseline"]["zero"]["mse"] == [0, 0]


@pytest.mark.parametrize("target,label", [(0, "[0,.25)"), (.25, "[.25,.5)"),
                                            (-.5, "[.5,.75)"), (.75, "[.75,1]"), (1, "[.75,1]")])
def test_strata_boundaries(target, label):
    board = chess.Board()
    assert strata(board, {"target": target, "phase": "opening"})["target_magnitude"] == label
    assert strata(board, {"target": target, "phase": "opening"})["nonpawn_material"] == "17+"
    board.remove_piece_at(chess.A2)
    assert strata(board, {"target": 0, "phase": "opening"})["material_imbalance"] == "<=1"
    for square in [chess.B2, chess.C2]:
        board.remove_piece_at(square)
    assert strata(board, {"target": 0, "phase": "opening"})["material_imbalance"] == "1-3"
    for units, expected in [(8, "0-8"), (9, "9-16"), (16, "9-16"), (17, "17+")]:
        b = chess.Board(None)
        for square in range(units):
            b.set_piece_at(square, chess.Piece(chess.KNIGHT, bool(square % 2)))
        assert strata(b, {"target": 0, "phase": "endgame"})["nonpawn_material"] == expected


def test_dropout_export_and_exact_resume(tmp_path):
    seed_all(42)
    model = build_model([16, 8], .2)
    x = torch.from_numpy(np.stack([encode(chess.Board())] * 16))
    assert not torch.equal(model(x), model(x))
    model.eval()
    assert torch.equal(model(x), model(x))
    checkpoint = tmp_path / "m.npz"
    export(model, checkpoint, {"hidden": [16, 8], "config": {"dropout": .2}})
    restored, _ = load_export(checkpoint)
    restored.eval()
    with torch.no_grad():
        np.testing.assert_allclose(NumpyEvaluator(checkpoint).batch(x.numpy()), model(x).squeeze().numpy(), atol=1e-6)
        assert torch.equal(restored(x), model(x))
    source = tmp_path / "data.jsonl"
    dataset(source)
    fit(source, tmp_path / "full", {**config(4), "dropout": .2})
    fit(source, tmp_path / "resume", {**config(2), "dropout": .2})
    fit(source, tmp_path / "resume", {**config(4), "dropout": .2}, resume=True)
    for (a, b), (c, d) in zip(NumpyEvaluator(tmp_path / "full/last.npz").layers,
                              NumpyEvaluator(tmp_path / "resume/last.npz").layers):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
    with pytest.raises(ValueError, match="mismatch"):
        fit(source, tmp_path / "resume", {**config(4), "dropout": .1}, resume=True)


def test_screen_and_promising_thresholds():
    cfg = {"gap_ratio": 1.1, "gap_seeds_required": 2,
           "promising_relative_improvement": .02, "promising_seeds_required": 2}
    baseline = [{"experiment": "baseline", "size": "all", "positions": 10, "seed": s,
                 "train": {"mse": 1}, "validation": {"mse": v}} for s, v in enumerate([1.1, 1.2, 1.3])]
    assert screen(baseline, cfg)
    assert not screen(baseline[:2], cfg)
    treatments = [{**r, "experiment": "dropout", "validation": {"mse": r["validation"]["mse"] * .97}}
                  for r in baseline]
    assert summarize_runs(baseline + treatments, cfg)[0]["promising"]


def test_tiny_full_study_resume_and_identity(tmp_path):
    source = tmp_path / "data.jsonl"
    dataset(source)
    checkpoint = tmp_path / "frozen.npz"
    seed_all(42)
    export(build_model([16, 8]), checkpoint, {"hidden": [16, 8]})
    cfg = read_json("configs/supervised-diagnostics.json")
    cfg.update(source=str(source), checkpoint=str(checkpoint), output=str(tmp_path / "study"),
               subset_sizes=[2, 4, "all"], bootstrap_resamples=20, training=config(2), gap_ratio=0)
    result = run_study(cfg)
    assert len(result["runs"]) == 21
    assert result["screening"]["triggered"]
    from chessengine.diagnostics_report import generate
    report = generate(cfg["output"], tmp_path / "report")
    assert report.exists()
    assert "not proof of overfitting" in report.read_text()
    raw = np.load(Path(cfg["output"]) / "predictions.npz")
    audit = read_json(Path(cfg["output"]) / "audit.json")
    assert audit["validation"]["supervised"]["mse"] == pytest.approx(
        np.mean((raw["supervised"] - raw["targets"]) ** 2))
    assert len({r["game_id"] for r in audit["worst_positions"]}) == len(audit["worst_positions"])
    histogram = read_json(tmp_path / "report/evidence/target-histograms.json")
    assert sum(histogram["train_counts"]) == 6
    assert sum(histogram["validation_counts"]) == 2
    resumed = run_study(cfg, resume=True)
    assert result["runs"] == resumed["runs"]
    for path in Path(cfg["output"]).glob("subset-*.jsonl"):
        assert all(r["split"] != "test" for r in map(json.loads, path.read_text().splitlines()))
    with pytest.raises(ValueError, match="mismatch"):
        run_study({**cfg, "sampling_seed": 4}, resume=True)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="mismatch"):
        run_study(cfg, resume=True)
