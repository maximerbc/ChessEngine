from __future__ import annotations

import csv
import time
from pathlib import Path
import chess
import chess.engine
import chess.pgn
import numpy as np
from .common import digest, read_json, write_json, provenance
from .features import restore
from .engine import Search, Handcrafted
from .model import NumpyEvaluator


def evaluator_for(spec):
    return Handcrafted() if spec == "handcrafted" else NumpyEvaluator(spec)


def identity(spec):
    if str(spec).startswith("stockfish:"):
        return "stockfish:" + digest(str(spec).removeprefix("stockfish:"))
    return "handcrafted-v1" if spec == "handcrafted" else digest(spec)


def paired_summary(games, seed=42, repetitions=10000):
    pairs = {}
    for game in games:
        pairs.setdefault(game["opening_id"], []).append(game)
    if any(
        len(pair) != 2 or {g["candidate_color"] for g in pair} != {"white", "black"}
        for pair in pairs.values()
    ):
        raise ValueError("Each opening must have exactly one game per color")
    if not pairs:
        raise ValueError("No complete opening pairs")
    values = np.array([np.mean([g["score"] for g in pair]) for pair in pairs.values()])
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(repetitions)])
    ci = np.quantile(samples, [0.025, 0.975]).tolist()
    uncapped = [np.mean([g["score"] for g in p]) for p in pairs.values() if not any(g["capped"] for g in p)]
    return {
        "games": len(games),
        "pairs": len(values),
        "wins": sum(g["score"] == 1 for g in games),
        "draws": sum(g["score"] == 0.5 for g in games),
        "losses": sum(g["score"] == 0 for g in games),
        "score": float(values.mean()),
        "ci95": ci,
        "capped_games": sum(g["capped"] for g in games),
        "uncapped_pairs": len(uncapped),
        "uncapped_score": float(np.mean(uncapped)) if uncapped else None,
        "conclusion": "improvement" if ci[0] > 0.5 else "regression" if ci[1] < 0.5 else "inconclusive",
    }


def freeze_protocol(candidate, baseline, openings, config, output, split="test"):
    if split not in ("development", "test"):
        raise ValueError("Matches require development or test openings")
    groups = read_json(openings)
    selected = groups[split][: config["pairs"]]
    if len(selected) < config["pairs"]:
        raise ValueError("Not enough reserved openings")
    keys = [o["id"] for o in selected]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate opening IDs")
    protocol = {
        "candidate": candidate,
        "baseline": baseline,
        "candidate_sha256": identity(candidate),
        "baseline_sha256": identity(baseline),
        "openings_sha256": digest(openings),
        "openings": selected,
        "split": split,
        "config": config,
        "provenance": provenance(),
    }
    path = Path(output)
    if path.exists():
        raise ValueError("Frozen protocol already exists; do not overwrite")
    write_json(path, protocol)
    return protocol


def run_matches(protocol_path, output, budget=None):
    """Always close the optional reference engine, including on interruption."""
    protocol = read_json(protocol_path)
    reference = None
    if protocol["baseline"].startswith("stockfish:"):
        binary = protocol["baseline"].removeprefix("stockfish:")
        if identity(protocol["baseline"]) != protocol["baseline_sha256"]:
            raise ValueError("Frozen reference executable has changed")
        reference = chess.engine.SimpleEngine.popen_uci(binary)
        reference.configure({"Threads": 1, "Hash": 32})
    try:
        return _run_matches(protocol_path, output, reference, budget)
    finally:
        if reference:
            reference.quit()


def _run_matches(protocol_path, output, reference=None, budget=None):
    protocol = read_json(protocol_path)
    for field in ("candidate", "baseline"):
        if identity(protocol[field]) != protocol[field + "_sha256"]:
            raise ValueError("Frozen checkpoint has changed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "matches.json"
    protocol_hash = digest(protocol_path)
    games = []
    if result_path.exists():
        previous = read_json(result_path)
        if previous["protocol_sha256"] != protocol_hash:
            raise ValueError("Output belongs to a different protocol")
        games = previous["games"]
    done = {(g["opening_id"], g["candidate_color"]) for g in games}
    candidate = evaluator_for(protocol["candidate"])
    baseline = None if reference else evaluator_for(protocol["baseline"])
    config = protocol["config"]
    for opening in protocol["openings"]:
        for color in (chess.WHITE, chess.BLACK):
            color_name = "white" if color else "black"
            if (opening["id"], color_name) in done:
                continue
            board = restore(opening)
            if reference:
                reference.configure({"Clear Hash": None})
            started = time.perf_counter()
            node_count = 0
            depth_sum, searched_moves, depth_one_moves = 0, 0, 0
            for _ in range(config["max_plies"]):
                if budget:
                    budget.check()
                if board.is_game_over(claim_draw=True):
                    break
                if reference and board.turn != color:
                    move = reference.play(
                        board, chess.engine.Limit(nodes=config.get("reference_nodes", 10000))
                    ).move
                    board.push(move)
                    continue
                evaluator = candidate if board.turn == color else baseline
                search = Search(
                    evaluator,
                    nodes=config["nodes"],
                    time_ms=config.get("time_ms"),
                    qdepth=config.get("qdepth", 2),
                )
                result = search.run(board)
                node_count += result.nodes
                depth_sum += result.depth
                searched_moves += 1
                depth_one_moves += result.depth >= 1
                board.push_uci(result.move)
            outcome = board.outcome(claim_draw=True)
            capped = outcome is None
            score = 0.5 if capped or outcome.winner is None else float(outcome.winner == color)
            game = chess.pgn.Game.from_board(board)
            game.headers["Event"] = f"ChessEngine {protocol['split']} evaluation"
            game.headers["White"] = "Candidate" if color else "Baseline"
            game.headers["Black"] = "Baseline" if color else "Candidate"
            game.headers["Result"] = "1/2-1/2" if capped else board.result(claim_draw=True)
            game.headers["Termination"] = "move-cap adjudication" if capped else outcome.termination.name
            games.append(
                {
                    "opening_id": opening["id"],
                    "candidate_color": color_name,
                    "score": score,
                    "capped": capped,
                    "pgn": str(game),
                    "nodes": node_count,
                    "searched_moves": searched_moves, "depth_one_moves": depth_one_moves, "depth_sum": depth_sum,
                    "seconds": time.perf_counter() - started,
                    "result": game.headers["Result"],
                }
            )
            write_json(result_path, {"protocol_sha256": protocol_hash, "protocol": protocol, "games": games})
            if budget:
                budget.checkpoint()
    summary = paired_summary(games, config["seed"])
    write_json(
        output / "summary.json",
        {
            **summary,
            "protocol_sha256": protocol_hash,
            "split": protocol["split"],
            "evidence": config.get("evidence", "pilot"),
        },
    )
    return summary


def select_checkpoint(candidates, output):
    """Select exclusively from development match directories; never inspect test scores."""
    scored = []
    reference_protocol = None
    for directory in candidates:
        matches = read_json(Path(directory) / "matches.json")
        protocol = matches["protocol"]
        if protocol["split"] != "development":
            raise ValueError("Checkpoint selection cannot use final-test matches")
        comparison = {
            key: protocol[key] for key in ("baseline_sha256", "openings_sha256", "openings", "config")
        }
        if reference_protocol is not None and comparison != reference_protocol:
            raise ValueError("Development candidates must use identical opponents, openings and budgets")
        reference_protocol = comparison
        summary = paired_summary(matches["games"])
        scored.append((summary["score"], protocol["candidate"], protocol["candidate_sha256"]))
    if not scored:
        raise ValueError("No candidate development results")
    best = max(scored, key=lambda item: item[0])
    write_json(
        output,
        {
            "checkpoint": best[1],
            "sha256": best[2],
            "development_score": best[0],
            "selection": "highest paired development score; input-order tie break",
        },
    )
    return best


def measure(checkpoint, data_path, output):
    from .data import records

    evaluator = NumpyEvaluator(checkpoint)
    rows = [r for r in records(data_path) if r["split"] == "test"]
    if not rows:
        raise ValueError("No held-out positions")
    by_phase = {}
    predictions, targets = [], []
    start = time.perf_counter()
    for row in rows:
        predicted = evaluator(restore(row))
        predictions.append(predicted)
        targets.append(row["target"])
        by_phase.setdefault(row["phase"], []).append((predicted - row["target"]) ** 2)
    elapsed = time.perf_counter() - start
    result = {
        "positions": len(rows),
        "teacher_mse": float(np.mean((np.array(predictions) - targets) ** 2)),
        "phase_mse": {k: float(np.mean(v)) for k, v in by_phase.items()},
        "ms_per_position_including_features": 1000 * elapsed / len(rows),
        "checkpoint_sha256": digest(checkpoint),
        "data_sha256": digest(data_path),
        "note": "Teacher agreement is not playing strength or outcome calibration.",
    }
    write_json(output, result)
    return result


def puzzles(checkpoint, csv_path, output, nodes=2000, limit=100):
    evaluator = evaluator_for(checkpoint)
    results = []
    with open(csv_path) as stream:
        for row in csv.DictReader(stream):
            board = chess.Board(row["FEN"])
            solution = row["Moves"].split()
            board.push_uci(solution[0])  # Lichess FEN precedes the opponent's setup move.
            result = Search(evaluator, nodes=nodes).run(board)
            proposed = chess.Move.from_uci(result.move)
            board.push(proposed)
            correct = result.move == solution[1] or board.is_checkmate()
            results.append(
                {"id": row["PuzzleId"], "correct": correct, "move": result.move, "expected": solution[1]}
            )
            if len(results) == limit:
                break
    write_json(
        output,
        {
            "results": results,
            "accuracy": sum(r["correct"] for r in results) / len(results),
            "checkpoint": identity(checkpoint),
            "data_sha256": digest(csv_path),
        },
    )
