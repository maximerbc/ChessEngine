import json
import shutil

import pytest
from fastapi.testclient import TestClient

from chessengine.api import create_app


def test_production_requires_versioned_explicit_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("REQUIRE_MODELS", "1")
    manifest = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="manifest is missing"):
        create_app(manifest)
    manifest.write_text(json.dumps({"models": []}))
    with pytest.raises(ValueError, match="v3 manifest"):
        create_app(manifest)


def test_production_rejects_corrupt_checksum(monkeypatch, tmp_path):
    monkeypatch.setenv("REQUIRE_MODELS", "1")
    shutil.copytree("artifacts/models", tmp_path / "models")
    manifest = tmp_path / "models/manifest.json"
    data = json.loads(manifest.read_text())
    data["models"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum mismatch"):
        create_app(manifest)


def test_production_models_and_cors(monkeypatch):
    monkeypatch.setenv("REQUIRE_MODELS", "1")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://demo.example")
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["models"] == 1 + len(json.load(open("artifacts/models/manifest.json"))["models"])
        expected = json.load(open("artifacts/models/manifest.json"))
        actual = {row["id"]: row for row in client.get("/models").json()["models"]}
        for row in expected["models"]:
            assert actual[row["id"]]["checkpoint_id"] == row["sha256"]
        for origin, status in [("https://demo.example", 200), ("https://other.example", 400)]:
            response = client.options("/analyze", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            })
            assert response.status_code == status
            assert (response.headers.get("access-control-allow-origin") == origin) == (status == 200)
