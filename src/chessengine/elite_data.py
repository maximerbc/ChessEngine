"""Fresh PGN played-move pairs, immutable sources, disk-backed exclusion/dedup index."""
import argparse
from bisect import bisect_right
from collections import Counter
import io
import json
from pathlib import Path
import sqlite3
import time
import urllib.request
import zipfile
import chess
import chess.pgn
import numpy as np
from .common import digest, json_hash, read_json
from .data import split_game
from .features import record_board, position_key, restore
from .elite_policy import encode_board, MOVE_IDS, MOVES, SCHEMA


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def source_months(first):
    year, month = map(int, first.split("-"))
    while year >= 2020:
        yield f"{year:04}-{month:02}"
        month -= 1
        if not month:
            year, month = year - 1, 12


def download(month, directory):
    path = Path(directory) / f"lichess_elite_{month}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://database.nikonoel.fr/lichess_elite_{month}.zip"
    receipt = path.with_suffix(".source.json")
    if path.exists():
        if receipt.exists() and read_json(receipt)["sha256"] != digest(path):
            raise ValueError("Elite source changed")
    else:
        import ssl
        import certifi
        with urllib.request.urlopen(url, timeout=60, context=ssl.create_default_context(
                cafile=certifi.where())) as response, path.with_suffix(".part").open("wb") as out:
            size = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > 2 * 1024 ** 3:
                    raise ValueError("Elite source exceeds 2 GiB bound")
                out.write(block)
        path.with_suffix(".part").replace(path)
    # PGNs are streamed, never extracted to untrusted archive paths.
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or not any(n.lower().endswith(".pgn") for n in archive.namelist()):
            raise ValueError("Invalid elite archive")
    atomic_json(receipt, dict(url=url, sha256=digest(path), bytes=path.stat().st_size))
    return path


def source_games(path):
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.lower().endswith(".pgn"):
                with archive.open(name) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig",
                                                                  errors="replace") as stream:
                    while (game := chess.pgn.read_game(stream)) is not None:
                        yield game


def rows_from_file(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".csv":
        import csv
        with path.open() as stream:
            for row in csv.DictReader(stream):
                fen = row.get("FEN") or row.get("fen")
                if fen:
                    yield dict(fen=fen)
    else:
        def walk(value):
            if isinstance(value, dict):
                if "fen" in value or "initial_fen" in value:
                    yield value
                else:
                    for child in value.values():
                        yield from walk(child)
            elif isinstance(value, list):
                for child in value:
                    yield from walk(child)
        yield from walk(read_json(path))


def exclusion_index(config, directory, source_root):
    directory = Path(directory)
    files = [Path(source_root) / name for name in config["exclusions"]]
    if any(not p.is_file() for p in files):
        raise ValueError(f"Missing required exclusions: {[str(p) for p in files if not p.is_file()]}")
    identity = {str(p): digest(p) for p in files}
    receipt = directory / "exclusions.json"
    index = directory / "exclusions.sqlite"
    if receipt.exists():
        existing = read_json(receipt)
        if existing["inputs"] != identity or digest(index) != existing["sha256"]:
            raise ValueError("Frozen exclusion index/input mismatch")
        return index
    if index.exists():
        index.unlink()  # Incomplete deterministic construction only; no completed dataset is changed.
    db = sqlite3.connect(index)
    db.execute("CREATE TABLE positions (key TEXT PRIMARY KEY) WITHOUT ROWID")
    db.execute("CREATE TABLE games (key TEXT PRIMARY KEY) WITHOUT ROWID")
    db.execute("CREATE TABLE reserved (key TEXT PRIMARY KEY) WITHOUT ROWID")
    began = time.perf_counter()
    for path in files:
        count = 0
        for row in rows_from_file(path):
            fen = row.get("fen")
            key = " ".join(fen.split()[:4]) if fen else position_key(restore(row))
            db.execute("INSERT OR IGNORE INTO positions VALUES (?)", (key,))
            if path.suffix != ".jsonl":
                db.execute("INSERT OR IGNORE INTO reserved VALUES (?)", (key,))
            gid = row.get("game_id") or row.get("source_game_id")
            if gid:
                db.execute("INSERT OR IGNORE INTO games VALUES (?)", (gid,))
            count += 1
            if count % 50000 == 0:
                db.commit()
                print(f"Exclusions {path.name}: {count}", flush=True)
        db.commit()
    counts = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
              for table in ["positions", "games"]}
    db.close()
    atomic_json(receipt, dict(inputs=identity, sha256=digest(index), counts=counts,
                             seconds=time.perf_counter() - began))
    return index


def game_pairs(game):
    """Validate a whole game before accepting any examples from it."""
    if game.errors or game.headers.get("Variant", "Standard") not in ("Standard", "From Position"):
        return None
    board = game.board()
    if board.chess960 or not board.is_valid():
        return None
    moves = list(game.mainline_moves())
    gid = json_hash(dict(initial=board.fen(), moves=[m.uci() for m in moves]))
    pairs, repetitions = [], Counter()
    repeated = False
    for move in moves:
        fen = board.fen()
        key = " ".join(fen.split()[:4])
        repetitions[key] += 1
        repeated |= repetitions[key] >= 2
        legal = list(board.legal_moves)
        if move not in legal or not board.is_valid():
            return None
        # Intended-move repetition claims only become possible after a repeated key.
        terminal = (board.outcome(claim_draw=False) is not None
                    or ((repeated or board.halfmove_clock >= 99) and board.is_game_over(claim_draw=True)))
        if terminal:
            break
        pairs.append((key, {**record_board(board), "game_id": gid, "target_move": move.uci()},
                      encode_board(board, legal), [MOVE_IDS[m.uci()] for m in legal]))
        board.push(move)
    return gid, pairs


def write_shard(directory, split, number, rows):
    path = Path(directory) / "encoded" / split / f"{number:05}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    pointers = np.cumsum([0] + [len(row[3]) for row in rows], dtype=np.uint32)
    with path.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream,
            x=np.packbits(np.stack([r[2].reshape(-1) for r in rows]).astype(np.uint8), axis=1),
            y=np.array([MOVE_IDS[r[1]["target_move"]] for r in rows], dtype=np.uint16),
            games=np.array([r[1]["game_id"] for r in rows], dtype="U64"),
            legal=np.array([m for r in rows for m in r[3]], dtype=np.uint16), pointers=pointers)
    path.with_suffix(".tmp").replace(path)
    raw = path.with_suffix(".jsonl")
    with raw.open("w") as stream:
        for row in rows:
            stream.write(json.dumps({**row[1], "split": split}, separators=(",", ":")) + "\n")
    return dict(path=str(path.relative_to(directory)), split=split, count=len(rows),
                sha256=digest(path), records=str(raw.relative_to(directory)), records_sha256=digest(raw))


def prepare(config, directory, source_root, archives, max_games=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "dataset.json"
    if complete.exists():
        meta = read_json(complete)
        if meta["config"] != config:
            raise ValueError("Existing elite dataset has another configuration")
        preflight(directory)
        return meta
    index = exclusion_index(config, directory, source_root)
    db = sqlite3.connect(directory / "dedup.sqlite")
    db.execute("ATTACH DATABASE ? AS excluded", (str(index),))
    db.execute("CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY) WITHOUT ROWID")
    db.execute("CREATE TABLE IF NOT EXISTS processed (key TEXT PRIMARY KEY) WITHOUT ROWID")
    progress_path = directory / "preparation.json"
    if progress_path.exists():
        progress = read_json(progress_path)
        if progress["config"] != config:
            raise ValueError("Preparation configuration changed")
    else:
        progress = dict(config=config, counts={s: 0 for s in config["positions"]}, shards=[],
                        sources={}, games_scanned=0, rejected_games=0, seconds=0)
    # Shard writes and index updates are committed together at each source-game boundary.
    # The durable checkpoint owns the exact shard inventory. Rebuild seen/processed after a crash
    # from checkpointed raw records, rather than trusting a transaction left ahead of the receipt.
    db.execute("DELETE FROM seen")
    db.execute("DELETE FROM processed")
    for shard in progress["shards"]:
        if digest(directory / shard["path"]) != shard["sha256"]:
            raise ValueError("Prepared shard changed")
        for row in rows_from_file(directory / shard["records"]):
            db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (" ".join(row["fen"].split()[:4]),))
    db.commit()
    began = time.perf_counter()
    pending = {s: [] for s in config["positions"]}
    total_scanned = 0

    def checkpoint():
        for split, rows in pending.items():
            if rows:
                number = sum(s["split"] == split for s in progress["shards"])
                progress["shards"].append(write_shard(directory, split, number, rows))
                rows.clear()
        progress["seconds"] += time.perf_counter() - tick[0]
        tick[0] = time.perf_counter()
        db.commit()
        atomic_json(progress_path, progress)

    tick = [began]
    try:
        for month in source_months(config["first_month"]):
            source = download(month, archives)
            sha = digest(source)
            if month in progress["sources"] and progress["sources"][month] != sha:
                raise ValueError("Source archive changed during resume")
            progress["sources"][month] = sha
            for game in source_games(source):
                total_scanned += 1
                if max_games and total_scanned > max_games:
                    checkpoint()
                    return dict(progress, complete=False)
                prepared = game_pairs(game)
                if prepared is None:
                    progress["rejected_games"] += 1
                    continue
                gid, pairs = prepared
                split = split_game(gid)
                if progress["counts"][split] >= config["positions"][split]:
                    continue
                if db.execute("SELECT 1 FROM excluded.games WHERE key=?", (gid,)).fetchone():
                    continue
                if any(db.execute("SELECT 1 FROM excluded.reserved WHERE key=?", (r[0],)).fetchone()
                       for r in pairs):
                    continue
                if db.execute("SELECT 1 FROM processed WHERE key=?", (gid,)).fetchone():
                    continue
                db.execute("INSERT INTO processed VALUES (?)", (gid,))
                progress["games_scanned"] += 1
                # Quarantine complete games touching a reserved position. For prior training
                # corpus positions, skip individual duplicates; no new split shares a position.
                for row in pairs:
                    if progress["counts"][split] >= config["positions"][split]:
                        break
                    key = row[0]
                    if db.execute("SELECT 1 FROM excluded.positions WHERE key=?", (key,)).fetchone():
                        continue
                    if db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (key,)).rowcount == 0:
                        continue
                    pending[split].append(row)
                    progress["counts"][split] += 1
                if max(map(len, pending.values())) >= config["shard_size"]:
                    checkpoint()
                    print(json.dumps(dict(counts=progress["counts"], seconds=round(progress["seconds"], 1))), flush=True)
                if progress["counts"] == config["positions"]:
                    checkpoint()
                    meta = dict(config=config, counts=progress["counts"], shards=progress["shards"],
                                schema=SCHEMA, sources=progress["sources"], label_source="human-played-move",
                                exclusions_sha256=digest(directory / "exclusions.json"),
                                preparation_seconds=progress["seconds"], games_scanned=progress["games_scanned"])
                    atomic_json(complete, meta)
                    return meta
        raise ValueError("Elite archives exhausted before exact split quotas")
    finally:
        db.close()


def preflight(directory):
    directory = Path(directory)
    meta = read_json(directory / "dataset.json")
    if meta["schema"] != SCHEMA or meta["counts"] != meta["config"]["positions"]:
        raise ValueError("Elite dataset schema/count mismatch")
    counts = {s: 0 for s in meta["counts"]}
    for shard in meta["shards"]:
        for key, hash_key in [("path", "sha256"), ("records", "records_sha256")]:
            path = (directory / shard[key]).resolve()
            if not path.is_relative_to(directory.resolve()) or digest(path) != shard[hash_key]:
                raise ValueError("Elite shard checksum/path mismatch")
        counts[shard["split"]] += shard["count"]
    if counts != meta["counts"]:
        raise ValueError("Elite shard quota mismatch")
    return meta


class ShardDataset:
    def __init__(self, directory, split):
        self.directory = Path(directory)
        meta = read_json(self.directory / "dataset.json")
        self.parts = [s for s in meta["shards"] if s["split"] == split]
        self.ends = np.cumsum([s["count"] for s in self.parts]).tolist()
        self.cached, self.data = None, None

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        part = bisect_right(self.ends, index)
        row = index - (self.ends[part - 1] if part else 0)
        if self.cached != part:
            with np.load(self.directory / self.parts[part]["path"], allow_pickle=False) as data:
                self.data = {k: data[k] for k in data.files}
            self.cached = part
        d = self.data
        x = np.unpackbits(d["x"][row], count=832).reshape(13, 8, 8).astype(np.float32)
        legal = d["legal"][d["pointers"][row]:d["pointers"][row + 1]].astype(np.int64)
        y = int(d["y"][row])
        if y not in legal or np.any(legal >= len(MOVES)):
            raise ValueError("Invalid played move/legal vocabulary in shard")
        return x, y, str(d["games"][row]), legal


def collate(rows):
    import torch
    x, y, games, legal = zip(*rows)
    mask = torch.zeros((len(rows), len(MOVES)), dtype=torch.bool)
    for i, ids in enumerate(legal):
        mask[i, ids] = True
    return torch.from_numpy(np.stack(x)), torch.tensor(y, dtype=torch.long), games, mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--source-root", required=True)
    p.add_argument("--archives", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-games", type=int)
    a = p.parse_args()
    result = prepare(read_json(a.config), a.output, a.source_root, a.archives, a.max_games)
    print(json.dumps({k: v for k, v in result.items() if k not in ["config", "shards"]}, indent=2))


if __name__ == "__main__":
    main()
