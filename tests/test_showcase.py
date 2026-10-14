import json
import numpy as np
import pytest
from chessengine.common import write_json, digest
from chessengine.model import build_model, export, NumpyEvaluator
from chessengine.ranking import build_ranker, export_ranker, NumpyRanker
from chessengine.showcase import build


def test_preview_strips_metadata_and_release_requires_gate(tmp_path):
    value = tmp_path / "value.npz"
    export(
        build_model([128, 64], feature_version=3),
        value,
        dict(hidden=[128, 64], kind="supervised", private_path="/private/example"),
    )
    rank = tmp_path / "rank.npz"
    export_ranker(build_ranker(value), rank, dict(private_path="/private/example"))
    write_json(
        tmp_path / "training-results.json",
        [dict(seed=42, checkpoint=str(rank), sha256=digest(rank), history=[])],
    )
    config = dict(output=str(tmp_path), baseline=str(value), rl_checkpoint=str(value))
    destination = tmp_path / "preview"
    manifest = build(config, destination)
    assert manifest["preview"] and len(manifest["models"]) == 3
    for name in ["supervised.npz", "rl.npz", "ranker.npz"]:
        with np.load(destination / "models" / name) as data:
            meta = json.loads(str(data["metadata"]))
            assert "private_path" not in meta
            assert "source_checkpoint_sha256" in meta
        assert digest(destination / "models" / name) == digest(destination / "public/models" / name)
    NumpyEvaluator(destination / "models/supervised.npz")
    NumpyRanker(destination / "models/ranker.npz")
    with pytest.raises(ValueError, match="release gate"):
        build(config, destination, publish=True)


def test_guided_bundle_and_api_identity(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from chessengine.common import write_json, digest
    from chessengine.ranking import GuidedEvaluator, load_guided
    from chessengine.model import NumpyEvaluator
    from chessengine.api import create_app

    value = tmp_path / "value.npz"
    export(build_model([128, 64], feature_version=3), value, {"hidden": [128, 64], "kind": "supervised"})
    rank = tmp_path / "rank.npz"
    export_ranker(build_ranker(value), rank, {})
    guided = GuidedEvaluator(NumpyEvaluator(value), NumpyRanker(rank))
    bundle = tmp_path / "bundle.json"
    write_json(
        bundle,
        {
            "value": value.name,
            "ranker": rank.name,
            "value_sha256": digest(value),
            "ranker_sha256": digest(rank),
        },
    )
    assert load_guided(bundle).checkpoint_id == guided.checkpoint_id
    manifest = tmp_path / "manifest.json"
    rows = [dict(id=i, name=i, file=value.name, sha256=digest(value)) for i in ["supervised", "rl"]]
    rows.append(
        dict(
            id="move-guided",
            name="guided",
            kind="move-guided",
            value_model_id="supervised",
            ranker_file=rank.name,
            ranker_sha256=digest(rank),
            sha256=guided.checkpoint_id,
        )
    )
    write_json(manifest, {"showcase_version": 2, "default": "supervised", "models": rows})
    monkeypatch.setenv("REQUIRE_MODELS", "1")
    with TestClient(create_app(manifest)) as client:
        result = client.post("/analyze", json={"model_id": "move-guided", "nodes": 100}).json()
        assert result["checkpoint_id"] == guided.checkpoint_id
        assert len(result["root_rankings"]) == 20
        assert result["components"]["value"] == digest(value)
        classical = client.post("/analyze", json={"model_id": "handcrafted", "nodes": 10}).json()
        terms = classical["explanation"]["handcrafted"]
        assert (
            abs(
                np.tanh((terms["material"] + terms["position"]) / terms["scale"])
                - classical["explanation"]["raw_value"]
            )
            < 1e-7
        )
    write_json(manifest, {"showcase_version": 2, "default": "supervised", "models": rows[:2]})
    with pytest.raises(ValueError, match="move-guided"):
        create_app(manifest)
    rank.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_guided(bundle)
