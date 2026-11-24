"""Independent trajectory and completed-study integrity checks."""

from pathlib import Path
import chess
from .common import read_json, digest, json_hash
from .features import restore
from .pv_data import validate_groups


def trajectory(game, simulations):
    board = restore(game["opening"])
    if len(game["played"]) != len(game["visits"]) or len(game["played"]) != len(game["metrics"]):
        raise ValueError("Trajectory target length mismatch")
    for uci, visits, work in zip(game["played"], game["visits"], game["metrics"]):
        if board.is_game_over(claim_draw=True):
            raise ValueError("Move after terminal state")
        legal = list(board.legal_moves)
        if (
            len(visits) != len(legal)
            or any(type(n) is not int or n < 0 for n in visits)
            or sum(visits) != simulations
        ):
            raise ValueError("Invalid simulation visit counts")
        if work["simulations"] != simulations or work["legal"] != len(legal):
            raise ValueError("Work/target inconsistency")
        move = chess.Move.from_uci(uci)
        if move not in legal:
            raise ValueError("Illegal trajectory move")
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None or outcome.winner is None else ("white" if outcome.winner else "black")
    if game["truncated"] != (outcome is None) or winner != game["winner"]:
        raise ValueError("Trajectory outcome mismatch")
    return not game["truncated"]


def completed(config, groups, root):
    validate_groups(groups)
    result = read_json(root / "results.json")
    selections = read_json(root / "selection-lock.json")
    if result["selections"] != selections or len(selections) != 2 * len(config["seeds"]):
        raise ValueError("Selection-lock mismatch")
    artifacts = {}
    for treatment in config["anchor_weights"]:
        for seed in config["seeds"]:
            directory = root / f"{treatment}-{seed}"
            training = read_json(directory / "training.json")
            if len(training["history"]) != config["batches"]:
                raise ValueError("Incomplete training history")
            for row in training["history"]:
                batch = row["batch"]
                actor_sha = digest(directory / f"actor-{batch:03}.npz")
                sources = {}
                complete = 0
                for index in range(config["games_per_batch"]):
                    path = directory / f"games-{batch:03}" / f"{index:04}.json"
                    game = read_json(path)
                    if game["actor_sha256"] != actor_sha or game["opening"] not in groups["train"]:
                        raise ValueError("Frozen actor or training opening mismatch")
                    complete += trajectory(game, config["simulations"])
                    sources[str(path)] = digest(path)
                if (
                    json_hash(sources) != row["data_sha256"]
                    or row["samples"] != complete * config["positions_per_game"]
                ):
                    raise ValueError("Uniform game-weighting/data identity mismatch")
                if digest(row["checkpoint"]) != row["sha256"]:
                    raise ValueError("Training checkpoint changed")
            artifacts[str(directory / "training.json")] = digest(directory / "training.json")
            if digest(directory / "selection.json") != selections[f"{treatment}-{seed}"]["selection_sha256"]:
                raise ValueError("Selection record changed")
    for row in result["seeds"]:
        directory = root / f"final-{row['seed']}"
        expected = [("primary", config["final_pairs"], row["primary"])]
        for treatment in config["anchor_weights"]:
            expected.extend(
                [
                    (f"{treatment}-initial", config["control_pairs"], row["secondary"][treatment]["initial"]),
                    (
                        f"{treatment}-released",
                        config["reference_pairs"],
                        row["secondary"][treatment]["released"],
                    ),
                ]
            )
        for name, pairs, summary in expected:
            path = directory / name / "matches.json"
            games = read_json(path)
            if len(games["games"]) != 2 * pairs or digest(path) != summary["matches_sha256"]:
                raise ValueError("Incomplete or modified final matches")
            artifacts[str(path)] = digest(path)
    manifest = read_json("artifacts/models/manifest.json")
    for row in manifest["models"]:
        path = Path("artifacts/models") / row["file"]
        if digest(path) != row["sha256"]:
            raise ValueError("Released model mismatch")
        if row.get('kind') == 'human-move-policy':
            from .elite_policy import NumpyElitePolicy, validate_release
            validate_release(row, NumpyElitePolicy(path), path.parent)
        elif digest(Path("web/public/models") / row["file"]) != row["sha256"]:
            raise ValueError("Released model mismatch")
    return dict(
        complete=True,
        artifacts=artifacts,
        selection_lock_sha256=digest(root / "selection-lock.json"),
        results_sha256=digest(root / "results.json"),
        released_manifest_sha256=digest("artifacts/models/manifest.json"),
        released_models_unchanged=True,
    )
