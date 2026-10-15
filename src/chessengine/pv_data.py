"""Fresh source-game-separated policy/value starts; no teacher labels consumed."""

from pathlib import Path
import chess
import chess.pgn
from .common import read_json, write_json, digest, json_hash, provenance
from .data import open_pgn, records
from .features import record_board, restore
from .rl_improvement import canonical


def validate_groups(groups):
    seen_games, seen_positions, seen_ids = set(), set(), set()
    for rows in groups.values():
        for row in rows:
            board = restore(row)
            game, key, identity = row["source_game_id"], canonical(board), row["id"]
            if game in seen_games or key in seen_positions or identity in seen_ids:
                raise ValueError("Opening source-game/position split leakage")
            if board.is_game_over(claim_draw=True):
                raise ValueError("Terminal opening")
            seen_games.add(game)
            seen_positions.add(key)
            seen_ids.add(identity)


def prepare(config):
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    inputs = {p: digest(p) for p in [config["archive"], config["corpus"], *config["reserved"]]}
    sizes = dict(
        pilot=config["pilot_games"],
        train=config["train_openings"],
        development=config["development_pairs"],
        final=config["final_pairs"],
        control=config["control_pairs"],
        reference=config["reference_pairs"],
    )
    signature = json_hash(dict(inputs=inputs, sizes=sizes, sampling_seed=config["sampling_seed"]))
    path, manifest = root / "openings.json", root / "openings-manifest.json"
    if path.exists():
        old = read_json(manifest)
        if old["signature"] != signature or old["sha256"] != digest(path):
            raise ValueError("Frozen opening inputs changed")
        groups = read_json(path)
        validate_groups(groups)
        return groups
    games, positions = set(), set()
    for row in records(config["corpus"]):
        games.add(row["game_id"])
        positions.add(canonical(chess.Board(row["fen"])))
    for reserved in config["reserved"]:
        for group in read_json(reserved).values():
            for row in group:
                games.add(row["source_game_id"])
                positions.add(canonical(restore(row)))
    chosen, scanned = [], 0
    with open_pgn(config["archive"]) as stream:
        while len(chosen) < sum(sizes.values()):
            game = chess.pgn.read_game(stream)
            if game is None:
                raise ValueError("Insufficient fresh openings; do not reuse old finals")
            scanned += 1
            moves = list(game.mainline_moves())
            board = game.board()
            if (
                game.errors
                or len(moves) < 16
                or game.headers.get("Variant", "Standard") != "Standard"
                or board.chess960
                or not board.is_valid()
            ):
                continue
            game_id = json_hash({"initial": board.fen(), "moves": [m.uci() for m in moves]})
            if game_id in games:
                continue
            for move in moves[:12]:
                board.push(move)
            key = canonical(board)
            if key in positions or board.is_game_over(claim_draw=True):
                continue
            games.add(game_id)
            positions.add(key)
            chosen.append({**record_board(board), "source_game_id": game_id, "id": game_id[:16]})
    # Independent deterministic split allocation, not chronological group assignment.
    chosen.sort(key=lambda r: json_hash([config["sampling_seed"], r["source_game_id"]]))
    groups, offset = {}, 0
    for name, size in sizes.items():
        groups[name] = chosen[offset : offset + size]
        offset += size
    validate_groups(groups)
    write_json(path, groups)
    write_json(
        manifest,
        dict(
            signature=signature,
            inputs=inputs,
            sizes=sizes,
            sha256=digest(path),
            scanned=scanned,
            canonical_overlaps=0,
            source_game_overlaps=0,
            provenance=provenance(),
        ),
    )
    return groups
