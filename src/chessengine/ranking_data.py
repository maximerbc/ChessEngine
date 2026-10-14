"""Whole-game subsets, frozen fresh openings, and resumable move labels."""

import argparse
import time
from pathlib import Path
import chess.engine
from .common import digest, json_hash, read_json, write_json, provenance
from .data import records
from .features import restore
from .rl_improvement import canonical, prepare_openings


class NoDeadline:
    def check(self):
        pass

    def checkpoint(self):
        pass


def select_games(rows, target, seed):
    groups = {}
    for row in rows:
        groups.setdefault(row["game_id"], []).append(row)
    chosen = []
    for game_id in sorted(groups, key=lambda g: json_hash([seed, g])):
        chosen.extend(groups[game_id])
        if len(chosen) >= target:
            break
    if len(chosen) < target:
        raise ValueError("Insufficient positions for whole-game subset")
    return chosen


def prepare(config):
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "data-protocol.json"
    identity = dict(
        config=config,
        inputs={
            key: digest(config[key])
            for key in [
                "source",
                "baseline",
                "teacher_manifest",
                "binary",
                "old_openings",
                "previous_openings",
                "archive",
            ]
        },
    )
    teacher = read_json(config["teacher_manifest"])
    if teacher["binary_sha256"] != identity["inputs"]["binary"]:
        raise ValueError("Pinned teacher executable changed")
    if manifest.exists():
        existing = read_json(manifest)
        if existing["identity"] != identity:
            raise ValueError("Move-label data/config mismatch")
        for name, sha in existing["outputs"].items():
            if digest(root / name) != sha:
                raise ValueError("Prepared move data changed")
        return root
    old = read_json(config["old_openings"])
    previous = read_json(config["previous_openings"])
    reserved = {
        canonical(restore(r)) for groups in [old, previous] for values in groups.values() for r in values
    }
    # Training openings are not excluded from the existing scalar corpus here: the original
    # data pipeline already excluded reserved positions; all fresh finals exclude the full corpus.
    split_games = {}
    split_positions = {}
    split_rows = {"train": [], "validation": []}
    terminal = 0
    for row in records(config["source"]):
        split = row["split"]
        game = row["game_id"]
        board = restore(row)
        key = canonical(board)
        for mapping, item in [(split_games, game), (split_positions, key)]:
            if item in mapping and mapping[item] != split:
                raise ValueError("Canonical or source-game split leakage")
            mapping[item] = split
        if split not in split_rows:
            continue
        if board.is_game_over(claim_draw=True):
            terminal += 1
            continue
        if key in reserved:
            continue
        split_rows[split].append(row)
    selected = []
    counts = {}
    for split, target in [
        ("train", config["train_positions"]),
        ("validation", config["validation_positions"]),
    ]:
        rows = select_games(split_rows[split], target, config["sampling_seed"])
        selected.extend(rows)
        counts[split] = len(rows)
    write_json(root / "positions.json", selected)
    all_old = {**old, "previous_final": previous["test"]}
    write_json(root / "reserved-all.json", all_old)
    opening_config = dict(
        old_openings=str(root / "reserved-all.json"),
        corpus=config["source"],
        archive=config["archive"],
        final_pairs=config["final_pairs"],
    )
    prepare_openings(opening_config, root, NoDeadline())
    write_json(
        manifest,
        dict(
            identity=identity,
            counts=counts,
            excluded_terminal=terminal,
            teacher=teacher,
            provenance=provenance(),
            outputs={
                name: digest(root / name)
                for name in ["positions.json", "reserved-all.json", "openings.json", "openings-manifest.json"]
            },
        ),
    )
    return root


def label(config, pilot=False):
    root = prepare(config)
    rows = read_json(root / "positions.json")
    size = config["label_chunk_size"]
    limit = min(len(rows), size) if pilot else len(rows)
    with chess.engine.SimpleEngine.popen_uci(config["binary"]) as teacher:
        teacher.configure({"Threads": 1, "Hash": 32})
        if teacher.id != read_json(config["teacher_manifest"])["teacher"]:
            raise ValueError("Teacher identity mismatch")
        for start in range(0, limit, size):
            path = root / "labels" / f"{start // size:04}.json"
            manifest = path.with_suffix(".manifest.json")
            subset = rows[start : min(start + size, len(rows))]
            source_hash = json_hash(subset)
            if path.exists() and manifest.exists():
                meta = read_json(manifest)
                if meta["source_sha256"] != source_hash or meta["sha256"] != digest(path):
                    raise ValueError("Move label chunk mismatch")
                continue
            result = []
            begin = time.perf_counter()
            for row in subset:
                board = restore(row)
                teacher.configure({"Clear Hash": None})
                info = teacher.analyse(board, chess.engine.Limit(nodes=config["label_nodes"]), game=object())
                if not info.get("pv") or info["pv"][0] not in board.legal_moves:
                    raise ValueError("Missing or illegal teacher move")
                result.append({**row, "teacher_move": info["pv"][0].uci()})
            write_json(path, result)
            seconds = time.perf_counter() - begin
            write_json(
                manifest,
                dict(source_sha256=source_hash, sha256=digest(path), seconds=seconds, positions=len(result)),
            )
            print(
                f"Move labels {min(start + size, len(rows))}/{len(rows)} ({seconds:.1f}s chunk)", flush=True
            )
    chunks = sorted((root / "labels").glob("*.manifest.json"))
    meta = [read_json(p) for p in chunks]
    write_json(
        root / "label-progress.json",
        dict(
            positions=sum(m["positions"] for m in meta),
            total=len(rows),
            seconds=sum(m["seconds"] for m in meta),
            complete=sum(m["positions"] for m in meta) == len(rows),
        ),
    )
    return read_json(root / "label-progress.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    print(label(read_json(a.config), a.pilot))
