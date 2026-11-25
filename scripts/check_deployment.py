"""Smoke-check running production containers; no credentials or mutations."""

import hashlib
import json
import re
import urllib.error
import urllib.request

import chess


def get(url):
    with urllib.request.urlopen(url, timeout=15) as response:
        return response.read(), response.headers


def main():
    api, web = "http://127.0.0.1:8000", "http://127.0.0.1:8080"
    assert json.loads(get(api + "/health")[0])["models"] >= 2
    models = {row["id"]: row for row in json.loads(get(api + "/models")[0])["models"]}
    with open("artifacts/models/manifest.json") as source:
        manifest = json.load(source)
    assert list(models) == ["handcrafted", "supervised-1m", "elite-policy"]
    try:
        get(web + "/models/rl.npz")
    except urllib.error.HTTPError as error:
        assert error.code == 404
    else:
        raise AssertionError("Retired weights remain publicly downloadable")
    for row in manifest["models"]:
        assert models[row["id"]]["checkpoint_id"] == row["sha256"]
        if row['id'] in {'supervised', 'supervised-1m'}:
            weights, headers = get(web + "/models/" + row.get("file", row.get("ranker_file")))
            assert hashlib.sha256(weights).hexdigest() == row.get("ranker_sha256", row["sha256"])
            assert "text/html" not in headers["Content-Type"]
        else:
            try:
                get(web + "/models/" + row['file'])
            except urllib.error.HTTPError as error:
                assert error.code == 404
            else:
                raise AssertionError("Private experimental weights are downloadable")
        request = urllib.request.Request(
            api + "/analyze",
            data=json.dumps(
                {
                    "model_id": row["id"],
                    "moves": ["e2e4"],
                    "nodes": 128,
                }
            ).encode(),
            headers={"Content-Type": "application/json", "Origin": web},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            assert response.headers["Access-Control-Allow-Origin"] == web
            result = json.load(response)
        board = chess.Board()
        board.push_uci("e2e4")
        assert chess.Move.from_uci(result["best_move"]) in board.legal_moves
    for model_id in models:
        request = urllib.request.Request(
            api + "/evaluate-comparison",
            data=json.dumps({"model_id": model_id, "moves": ["e2e4"]}).encode(),
            headers={"Content-Type": "application/json", "Origin": web},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            comparison = json.load(response)
        assert comparison["perspective"] == "white"
        assert comparison["moves"] == ["e2e4"]
        assert comparison["checkpoint_id"] == models[model_id]["checkpoint_id"]
        assert comparison["stockfish"]["status"] == "ok"
        assert comparison["stockfish"]["version"] == "Stockfish 19"
        if models[model_id].get('strategy') == 'direct-policy-v1':
            assert comparison['position_value'] is None and comparison['search'] is None
            assert comparison['policy']['root_rankings']
        else:
            assert comparison["search"]["nodes"] <= 10000
    for path, content_type in [
        ("/", "text/html"),
        ("/health", "text/plain"),
        ("/research/report.pdf", "application/pdf"),
        ("/research/rl.png", "image/png"),
        ("/research/games-seed-42.json", "application/json"),
    ]:
        body, headers = get(web + path)
        assert body and content_type in headers["Content-Type"]
        assert headers["X-Content-Type-Options"] == "nosniff"
    for route in ["classical", "supervised", "move-guided", "rl"]:
        body, headers = get(web + "/lab/" + route)
        assert body and "text/html" in headers["Content-Type"]
    if manifest.get("showcase_version") == 2:
        catalog = json.loads(get(web + "/research/model-catalog.json")[0])
        for key, row in catalog["models"].items():
            assert models[key]["checkpoint_id"] == row["checkpoint_id"]
            if row.get("report"):
                assert "text/html" in get(web + row["report"])[1]["Content-Type"]
    homepage = get(web + "/")[0].decode()
    for asset in re.findall(r'"(/assets/[^"]+)"', homepage):
        body, headers = get(web + asset)
        assert body and "max-age=31536000" in headers["Cache-Control"]
    denied = urllib.request.Request(
        api + "/analyze",
        method="OPTIONS",
        headers={
            "Origin": "https://other.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    try:
        urllib.request.urlopen(denied)
    except urllib.error.HTTPError as error:
        assert error.code == 400
        assert error.headers.get("Access-Control-Allow-Origin") is None
    else:
        raise AssertionError("Unexpected CORS acceptance")
    for path in [
        "/.env",
        "/AGENTS.md",
        "/src/chessengine/api.py",
        "/runs/test.pt",
        "/research/missing.pdf",
        "/models/missing.npz",
    ]:
        try:
            get(web + path)
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError(f"Unexpected public file: {path}")
    print("Production HTTP and artifact checks passed")


if __name__ == "__main__":
    main()
