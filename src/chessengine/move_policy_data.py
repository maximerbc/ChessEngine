"""Isolated, history-preserving preparation, teacher labels and bounded shards."""
import bisect
import csv
import json
import random
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlsplit
import chess
import chess.engine
import chess.pgn
import numpy as np
from .common import digest, json_hash, read_json, write_json, provenance
from .data import open_pgn, records, split_game
from .features import encode, restore, record_board, position_key
from .ranking import move_features

SPLITS = ("train", "validation", "test")


def canonical(board):
    return position_key(board if board.turn else board.mirror())


def position_rows(value):
    if isinstance(value, dict):
        if "fen" in value:
            yield value
        else:
            for nested in value.values():
                yield from position_rows(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from position_rows(nested)


def input_identity(config, input_root):
    root = Path(input_root)
    paths = [config["source"], *config["exclusions"]]
    if not config["exclusions"]:
        raise ValueError("Explicit exclusion sources are required")
    return {name: digest(root / name) for name in paths}  # Missing inputs fail closed.


def exclusion_rows(path):
    if path.suffix == ".jsonl":
        yield from records(path)
    elif path.suffix == ".csv":
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                board = chess.Board(row["FEN"])
                origin = {"source_site_id": site_id(row.get("GameUrl", ""))}
                yield {**record_board(board), **origin}
                for move in row["Moves"].split():
                    board.push_uci(move)
                    yield {**record_board(board), **origin}
    else:
        yield from position_rows(read_json(path))


def site_id(url):
    parsed = urlsplit(url)
    if parsed.hostname == "lichess.org" or (parsed.hostname or "").endswith(".lichess.org"):
        return parsed.path.strip("/").split("/")[0][:8] or None
    return None


def prepare(config, output, input_root="."):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "inputs": input_identity(config, input_root)}
    manifest_path = root / "positions.manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["identity"] != identity or digest(root / "positions.jsonl") != old["sha256"]:
            raise ValueError("Prepared policy data changed")
        return old
    excluded_games, excluded_positions, excluded_sites = set(), set(), set()
    began = time.perf_counter()
    indexed = 0
    for name in config["exclusions"]:
        path = Path(input_root) / name
        for row in exclusion_rows(path):
            excluded_games.update(row[k] for k in ("game_id", "source_game_id") if row.get(k))
            if row.get("source_site_id"):
                excluded_sites.add(row["source_site_id"])
            # Exclusion keys ignore clocks/history; only candidate examples replay history.
            excluded_positions.add(canonical(chess.Board(row["fen"])))
            indexed += 1
            if indexed % 50000 == 0:
                write_json(root / "progress.json", {"stage": "exclusions", "rows": indexed,
                           "source": name, "seconds": time.perf_counter() - began})
                print(f"Indexed {indexed:,} exclusion rows", flush=True)
    targets = config["positions"]
    if set(targets) != set(SPLITS) or any(type(v) is not int or v < 1 for v in targets.values()):
        raise ValueError("Positive train/validation/test sizes required")
    database = root / "preparation.sqlite"
    database.unlink(missing_ok=True)  # Incomplete preparation is rebuilt, never silently resumed.
    db = sqlite3.connect(database)
    db.executescript("CREATE TABLE games(id TEXT PRIMARY KEY, split TEXT, quarantine INTEGER);"
                     "CREATE TABLE samples(key TEXT, game TEXT, row TEXT, PRIMARY KEY(key,game));"
                     "CREATE INDEX positions ON samples(key);"
                     "CREATE INDEX game_samples ON samples(game);")
    rng = random.Random(config["sampling_seed"])
    scanned = 0
    try:
        with open_pgn(str(Path(input_root) / config["source"])) as stream:
            while scanned < config["max_games"]:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                scanned += 1
                if game.errors or game.headers.get("Variant", "Standard") != "Standard":
                    continue
                if site_id(game.headers.get("Site", "")) in excluded_sites:
                    continue
                board = game.board()
                if board.chess960 or not board.is_valid():
                    continue
                moves = list(game.mainline_moves())
                gid = json_hash({"initial": board.fen(), "moves": [m.uci() for m in moves]})
                if gid in excluded_games or db.execute("SELECT 1 FROM games WHERE id=?", (gid,)).fetchone():
                    continue
                split = split_game(gid)
                db.execute("INSERT INTO games VALUES (?,?,0)", (gid, split))
                phases = [[i for i in range(len(moves)) if lo <= i < hi]
                          for lo, hi in [(8, 30), (30, 70), (70, 1000)]]
                chosen = {i for phase in phases for i in rng.sample(phase, min(2, len(phase)))}
                for i, move in enumerate(moves):
                    board.push(move)
                    if i not in chosen or board.is_game_over(claim_draw=True):
                        continue
                    key = canonical(board)
                    if key in excluded_positions:
                        db.execute("UPDATE games SET quarantine=1 WHERE id=?", (gid,))
                        continue
                    conflicts = db.execute(
                        "SELECT games.id FROM samples JOIN games ON samples.game=games.id "
                        "WHERE key=? AND split!=?", (key, split)).fetchall()
                    if conflicts:
                        db.executemany("UPDATE games SET quarantine=1 WHERE id=?", [(gid,), *conflicts])
                    row = {**record_board(board), "game_id": gid, "split": split}
                    db.execute("INSERT OR IGNORE INTO samples VALUES (?,?,?)", (key, gid, json.dumps(row)))
                if scanned % 1000 == 0:
                    db.commit()
                    counts = dict(db.execute("SELECT split,count(*) FROM samples JOIN games "
                                             "ON samples.game=games.id WHERE quarantine=0 GROUP BY split"))
                    write_json(root / "progress.json", {"stage": "preparation", "games": scanned,
                               "eligible_samples": counts, "seconds": time.perf_counter() - began})
                    print(f"Scanned {scanned:,} games: {counts}", flush=True)
                    if all(counts.get(s, 0) >= targets[s] * 1.2 for s in SPLITS):
                        break
        db.commit()
        counts = dict.fromkeys(SPLITS, 0)
        seen = set()
        temp = root / "positions.jsonl.tmp"
        with temp.open("w") as out:
            for gid, split in db.execute("SELECT id,split FROM games WHERE quarantine=0 ORDER BY id"):
                if counts[split] >= targets[split]:
                    continue
                for key, row in db.execute("SELECT key,row FROM samples WHERE game=? ORDER BY key", (gid,)):
                    if key not in seen:
                        seen.add(key)
                        out.write(row + "\n")
                        counts[split] += 1
        if any(counts[s] < targets[s] for s in SPLITS):
            raise ValueError(f"Insufficient fresh positions: {counts}; require {targets}")
        temp.replace(root / "positions.jsonl")
        result = {"identity": identity, "sha256": digest(root / "positions.jsonl"), "counts": counts,
                  "games_scanned": scanned,
                  "quarantined_games": db.execute("SELECT count(*) FROM games WHERE quarantine=1").fetchone()[0],
                  "provenance": provenance()}
        write_json(manifest_path, result)
        return result
    finally:
        db.close()


def verify_positions(root, config):
    meta = read_json(root / "positions.manifest.json")
    if meta["identity"]["config"] != config or meta["sha256"] != digest(root / "positions.jsonl"):
        raise ValueError("Policy positions/config mismatch")
    return meta


def chunks(iterator, size):
    chunk = []
    for row in iterator:
        chunk.append(row)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def label(config, output, input_root="."):
    root = Path(output)
    positions = verify_positions(root, config)
    teacher = config["teacher"]
    if teacher.get("threads", 1) != 1 or teacher.get("hash_mb", 32) != 32 or teacher["nodes"] < 1:
        raise ValueError("Teacher protocol requires one thread, 32 MiB hash and positive nodes")
    if type(config["shard_size"]) is not int or config["shard_size"] < 1:
        raise ValueError("Positive shard size required")
    binary = Path(input_root) / teacher["binary"]
    if digest(binary) != teacher["sha256"]:
        raise ValueError("Pinned Stockfish executable mismatch")
    identity = json_hash({"positions": positions["sha256"], "teacher": teacher})
    networks = teacher.get("nnue", {})
    for network in networks.values():
        if digest(Path(input_root) / network["path"]) != network["sha256"]:
            raise ValueError("Pinned teacher NNUE mismatch")
    manifests = []
    with chess.engine.SimpleEngine.popen_uci(str(binary)) as engine:
        if networks:
            declared = {name for name in ("EvalFile", "EvalFileSmall") if name in engine.options}
            if declared != set(networks):
                raise ValueError("Teacher NNUE option mismatch")
            engine.configure({name: str((Path(input_root) / net["path"]).resolve())
                              for name, net in networks.items()})
        engine.configure({"Threads": 1, "Hash": 32, "MultiPV": 1, "Skill Level": 20,
                          "UCI_LimitStrength": False})
        for i, subset in enumerate(chunks(records(root / "positions.jsonl"), config["shard_size"])):
            path = root / "labels" / f"{i:05}.jsonl"
            meta_path = path.with_suffix(".manifest.json")
            source_hash = json_hash(subset)
            if meta_path.exists():
                meta = read_json(meta_path)
                if (meta["identity"] != identity or meta["source"] != source_hash
                        or meta["sha256"] != digest(path)):
                    raise ValueError("Policy label resume mismatch")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                start = time.perf_counter()
                temp = path.with_suffix(".tmp")
                with temp.open("w") as out:
                    for row in subset:
                        board = restore(row)
                        engine.configure({"Clear Hash": None})
                        info = engine.analyse(board, chess.engine.Limit(nodes=teacher["nodes"]), game=object())
                        if not info.get("pv") or info["pv"][0] not in board.legal_moves:
                            raise ValueError("Missing or illegal teacher move")
                        out.write(json.dumps({**row, "teacher_move": info["pv"][0].uci()}) + "\n")
                temp.replace(path)
                meta = {"identity": identity, "source": source_hash, "sha256": digest(path),
                        "positions": len(subset), "seconds": time.perf_counter() - start,
                        "teacher": engine.id}
                write_json(meta_path, meta)
            manifests.append({"path": str(path.relative_to(root)), **meta})
            write_json(root / "progress.json", {"stage": "labeling",
                       "positions": sum(m["positions"] for m in manifests),
                       "total": sum(positions["counts"].values()),
                       "seconds": sum(m["seconds"] for m in manifests)})
            print(f"Labeled {sum(m['positions'] for m in manifests):,} positions", flush=True)
    result = {"identity": identity, "positions": positions["sha256"], "chunks": manifests,
              "complete": True, "seconds": sum(m["seconds"] for m in manifests)}
    write_json(root / "labels.manifest.json", result)
    return result


def encode_shards(config, output):
    root = Path(output)
    positions = verify_positions(root, config)
    labels = read_json(root / "labels.manifest.json")
    expected = json_hash({"positions": positions["sha256"], "teacher": config["teacher"]})
    if not labels["complete"] or labels["identity"] != expected:
        raise ValueError("Incomplete or mismatched policy labels")
    games, owners = {}, {}
    parts = []
    counts = dict.fromkeys(SPLITS, 0)
    # Each encoded chunk is bounded by shard_size, including during encoding.
    for chunk in labels["chunks"]:
        source = root / chunk["path"]
        if digest(source) != chunk["sha256"]:
            raise ValueError("Corrupt teacher labels")
        rows = list(records(source))
        for row in rows:
            split, gid = row["split"], row["game_id"]
            if split not in SPLITS or (gid in games and games[gid] != split):
                raise ValueError("Source-game split leakage")
            games[gid] = split
            board = restore(row)
            key = canonical(board)
            if key in owners:
                raise ValueError("Duplicate or cross-split policy position")
            owners[key] = split
            if board.is_game_over(claim_draw=True) or row["teacher_move"] not in {m.uci() for m in board.legal_moves}:
                raise ValueError("Terminal position or illegal teacher target")
        for split in SPLITS:
            selected = [r for r in rows if r["split"] == split]
            if not selected:
                continue
            xs, moves, targets, offsets = [], [], [], [0]
            for row in selected:
                board = restore(row)
                legal = list(board.legal_moves)
                xs.append(encode(board, 3))
                moves.append(move_features(board, legal))
                targets.append([m.uci() for m in legal].index(row["teacher_move"]))
                offsets.append(offsets[-1] + len(legal))
            name = f"encoded/{split}-{len(parts):05}.npz"
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix(".tmp").open("wb") as stream:
                np.savez_compressed(stream, x=np.stack(xs), moves=np.concatenate(moves),
                                    target=np.array(targets, dtype=np.int64), offsets=np.array(offsets),
                                    games=np.array([r["game_id"] for r in selected]))
            path.with_suffix(".tmp").replace(path)
            parts.append({"path": name, "split": split, "count": len(selected), "sha256": digest(path)})
            counts[split] += len(selected)
    if counts != positions["counts"]:
        raise ValueError("Encoded counts disagree with frozen positions")
    result = {"format": "direct-move-data-v1", "feature_version": 3, "move_schema": 1,
              "config": config, "labels_sha256": digest(root / "labels.manifest.json"),
              "preparation": positions, "teacher_chunks": labels["chunks"],
              "counts": counts, "shards": parts}
    write_json(root / "dataset.json", result)
    return result


class ShardDataset:
    """Map-style torch-compatible dataset with one decoded shard per process."""
    def __init__(self, root, split, verify=True):
        self.root = Path(root)
        meta = read_json(self.root / "dataset.json")
        if (meta["format"] != "direct-move-data-v1" or meta["feature_version"] != 3
                or meta["move_schema"] != 1 or split not in SPLITS):
            raise ValueError("Unsupported policy dataset")
        self.parts = [p for p in meta["shards"] if p["split"] == split]
        self.ends = np.cumsum([p["count"] for p in self.parts]).tolist()
        if not self.ends or self.ends[-1] != meta["counts"][split]:
            raise ValueError("Missing or inconsistent policy split")
        if verify:
            for part in self.parts:
                if digest(self.root / part["path"]) != part["sha256"]:
                    raise ValueError("Corrupt policy shard")
        self.cached_index, self.cached = None, None

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        part = bisect.bisect_right(self.ends, index)
        if part != self.cached_index:
            with np.load(self.root / self.parts[part]["path"], allow_pickle=False) as data:
                self.cached = {k: data[k] for k in data.files}
            self.cached_index = part
        local = index - (self.ends[part - 1] if part else 0)
        a = self.cached
        lo, hi = a["offsets"][local:local + 2]
        return a["x"][local], a["moves"][lo:hi], int(a["target"][local]), str(a["games"][local])


def collate(rows):
    import torch
    length = max(len(r[1]) for r in rows)
    moves = torch.zeros((len(rows), length, 3), dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.bool)
    for i, row in enumerate(rows):
        moves[i, :len(row[1])] = torch.from_numpy(row[1])
        mask[i, :len(row[1])] = True
    return (torch.from_numpy(np.stack([r[0] for r in rows])), moves, mask,
            torch.tensor([r[2] for r in rows]), [r[3] for r in rows])
