from __future__ import annotations

import io
import json
import random
from pathlib import Path
import urllib.request
import ssl
import certifi
import chess
import chess.engine
import chess.pgn
from .common import digest, json_hash, write_json
from .features import position_key, record_board, restore


def split_game(game_id: str) -> str:
    bucket = int(json_hash(game_id)[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def records(path):
    with open(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def open_pgn(source):
    raw = (
        urllib.request.urlopen(source, timeout=60, context=ssl.create_default_context(cafile=certifi.where()))
        if source.startswith("https://")
        else open(source, "rb")
    )
    if source.endswith(".zst"):
        import zstandard

        raw = zstandard.ZstdDecompressor().stream_reader(raw)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def prepare(source, output, limit=10000, seed=42, reserved=None, max_games=100000):
    rng, seen, counts = random.Random(seed), set(), {"train": 0, "validation": 0, "test": 0}
    excluded = set()
    excluded_games = set()
    if reserved:
        for group in json.loads(Path(reserved).read_text()).values():
            for row in group:
                board = restore(row)
                if row.get("source_game_id"):
                    excluded_games.add(row["source_game_id"])
                excluded.add(position_key(board))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open_pgn(source) as stream, open(output, "w") as out:
        games, written = 0, 0
        while written < limit and games < max_games:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            games += 1
            if game.errors or game.headers.get("Variant", "Standard") != "Standard":
                continue
            board = game.board()
            if board.chess960 or not board.is_valid():
                continue
            moves = list(game.mainline_moves())
            game_id = json_hash({"initial": board.fen(), "moves": [m.uci() for m in moves]})
            if game_id in excluded_games:
                continue
            split = split_game(game_id)
            # Two samples per phase, bounded memory and bounded work per game.
            phases = [
                [i for i in range(len(moves)) if lo <= i < hi] for lo, hi in [(8, 30), (30, 70), (70, 1000)]
            ]
            selected = {i for phase in phases for i in rng.sample(phase, min(2, len(phase)))}
            for i, move in enumerate(moves):
                board.push(move)
                if i not in selected or board.is_game_over(claim_draw=True):
                    continue
                key = position_key(board)
                if key in seen or key in excluded:
                    continue
                seen.add(key)
                row = {
                    **record_board(board),
                    "game_id": game_id,
                    "source": source,
                    "split": split,
                    "phase": "opening" if i < 30 else "middlegame" if i < 70 else "endgame",
                }
                out.write(json.dumps(row) + "\n")
                counts[split] += 1
                written += 1
                if written >= limit:
                    break
    manifest = {
        "source": source,
        "seed": seed,
        "positions": written,
        "games_scanned": games,
        "counts": counts,
        "sha256": digest(output),
        "reserved": digest(reserved) if reserved else None,
    }
    write_json(str(output) + ".manifest.json", manifest)
    return manifest


def label(source, output, stockfish, nodes=10000):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with chess.engine.SimpleEngine.popen_uci(stockfish) as engine, open(output, "w") as out:
        engine.configure({"Threads": 1, "Hash": 32, "UCI_ShowWDL": True})
        engine_id = engine.id.copy()
        for row in records(source):
            board = restore(row)
            engine.configure({"Clear Hash": None})
            info = engine.analyse(board, chess.engine.Limit(nodes=nodes), game=object())
            wdl = info.get("wdl")
            if wdl is None:
                raise ValueError("Teacher must support UCI_ShowWDL; no silent target conversion")
            score = wdl.pov(board.turn)
            target = (score.wins - score.losses) / score.total()
            out.write(json.dumps({**row, "target": target}) + "\n")
    manifest = {
        "source_sha256": digest(source),
        "sha256": digest(output),
        "teacher": engine_id,
        "binary_sha256": digest(stockfish),
        "nodes": nodes,
        "threads": 1,
        "hash_mb": 32,
    }
    write_json(str(output) + ".manifest.json", manifest)
    return manifest


def validate_splits(rows):
    games, positions = {}, {}
    for row in rows:
        split = row["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        key = position_key(restore(row))
        for mapping, item in [(games, row["game_id"]), (positions, key)]:
            if item in mapping and mapping[item] != split:
                raise ValueError("Cross-split leakage detected")
            mapping[item] = split
        if not -1 <= row["target"] <= 1:
            raise ValueError("Invalid teacher target")
