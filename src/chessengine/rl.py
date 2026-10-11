from __future__ import annotations

import random
import time
from pathlib import Path
import chess
import chess.pgn
import numpy as np
import torch
from .common import digest, json_hash, provenance, write_json, read_json
from .engine import Search
from .features import encode, restore
from .model import NumpyEvaluator, export, load_export
from .train import seed_all, rng_state, restore_rng, save_state, trusted_load, device_for


def outcome_value(winner, turn):
    return 0.0 if winner is None else 1.0 if winner == turn else -1.0


def play_game(evaluator, opening, config, rng):
    if getattr(evaluator, 'feature_version', 1) != 1:
        raise ValueError('RL currently supports feature version 1 only')
    board = restore(opening)
    root = board.copy(stack=True)
    samples, played = [], []
    started = time.perf_counter()
    for ply in range(config["max_plies"]):
        if board.is_game_over(claim_draw=True):
            break
        samples.append((encode(board), board.turn))
        if ply < config["exploration_plies"] and rng.random() < config["epsilon"]:
            move = rng.choice(list(board.legal_moves))
        else:
            result = Search(evaluator, nodes=config["nodes"], qdepth=config.get("qdepth", 2)).run(board)
            move = chess.Move.from_uci(result.move)
        board.push(move)
        played.append(move)
    outcome = board.outcome(claim_draw=True)
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "ChessEngine self-play"
    game.headers["Result"] = board.result(claim_draw=True)
    game.headers["Termination"] = "truncated" if outcome is None else outcome.termination.name
    result = {
        "pgn": str(game),
        "plies": len(played),
        "truncated": outcome is None,
        "result": game.headers["Result"],
        "seconds": time.perf_counter() - started,
        "opening_id": opening.get("id", json_hash(root.fen())),
        "termination": game.headers["Termination"],
    }
    if outcome is None or not samples:
        return result, None
    return result, (
        np.stack([x for x, _ in samples]),
        np.array([outcome_value(outcome.winner, turn) for _, turn in samples], dtype=np.float32),
    )


def selfplay(checkpoint, openings_path, output, config, resume=False):
    if NumpyEvaluator(checkpoint).feature_version != 1:
        raise ValueError('RL currently supports feature version 1 only')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config["seed"])
    rng = random.Random(config["seed"])
    groups = read_json(openings_path)
    openings = groups["train"]
    if not openings:
        raise ValueError("Training openings are required")
    # Shared position between training and development/test is a hard error.
    from .features import position_key

    train_keys = {position_key(restore(o)) for o in openings}
    reserved = {position_key(restore(o)) for key in ("development", "test") for o in groups[key]}
    if train_keys & reserved:
        raise ValueError("Opening split leakage")
    device = device_for(config.get("device", "cpu"))
    model, metadata = load_export(checkpoint)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0)
    identity = {k: v for k, v in config.items() if k != "batches"}
    signature = json_hash(
        {"initial": digest(checkpoint), "openings": digest(openings_path), "config": identity}
    )
    history, start_batch, games_total = [], 0, 0
    state_path = output / "resume.pt"
    if resume:
        state = trusted_load(state_path)
        if state["signature"] != signature:
            raise ValueError("Resume data/config mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        rng.setstate(state["game_rng"])
        history, start_batch, games_total = state["history"], state["batch"], state["games_total"]
    elif state_path.exists():
        raise ValueError("Output exists; use --resume")
    for batch in range(start_batch, config["batches"]):
        actor_path = output / "actor.npz"
        export(model, actor_path, {**metadata, "kind": "rl-actor"})
        actor = NumpyEvaluator(actor_path)
        game_records, game_samples = [], []
        for _ in range(config["games_per_batch"]):
            result, sample = play_game(actor, rng.choice(openings), config, rng)
            result["actor_sha256"] = actor.checkpoint_id
            game_records.append(result)
            if sample is not None:
                game_samples.append(sample)
        # Same number of samples from each completed game => equal game weighting.
        started = time.perf_counter()
        losses = []
        if game_samples:
            xs, ys = [], []
            for x, y in game_samples:
                ids = [rng.randrange(len(x)) for _ in range(config["positions_per_game"])]
                xs.append(x[ids])
                ys.append(y[ids])
            x = torch.from_numpy(np.concatenate(xs))
            y = torch.from_numpy(np.concatenate(ys))
            model.train()
            for ids in torch.randperm(len(x)).split(config["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x[ids].to(device)).squeeze(-1)
                loss = ((prediction - y[ids].to(device)) ** 2).mean()
                loss.backward()
                optimizer.step()
                losses.append((loss.item(), len(ids)))
        update_seconds = time.perf_counter() - started
        games_total += len(game_records)
        batch_metrics = {
            "batch": batch + 1,
            "games_total": games_total,
            "completed": len(game_samples),
            "truncated": sum(g["truncated"] for g in game_records),
            "loss": sum(a * n for a, n in losses) / sum(n for _, n in losses) if losses else None,
            "generation_seconds": sum(g["seconds"] for g in game_records),
            "update_seconds": update_seconds,
            "actor_sha256": actor.checkpoint_id,
        }
        history.append(batch_metrics)
        metadata_out = {
            **metadata,
            "kind": "rl",
            "initial_sha256": digest(checkpoint),
            "config": config,
            "provenance": provenance(),
            "batch": batch + 1,
            "evidence": config.get("evidence", "pilot"),
        }
        export(model, output / f"batch-{batch + 1:03}.npz", metadata_out)
        export(model, output / "last.npz", metadata_out)
        write_json(output / f"games-{batch + 1:03}.json", game_records)
        save_state(
            state_path,
            {
                "signature": signature,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(),
                "game_rng": rng.getstate(),
                "batch": batch + 1,
                "history": history,
                "games_total": games_total,
            },
        )
        write_json(
            output / "metrics.json",
            {
                "config": config,
                "device": device,
                "history": history,
                "initial_sha256": digest(checkpoint),
                "openings_sha256": digest(openings_path),
                "provenance": provenance(),
            },
        )
    return {"batches": len(history), "games": games_total, "checkpoint": str(output / "last.npz")}
