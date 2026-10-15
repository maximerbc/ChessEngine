"""Frozen batch actors, uniform game-weighted updates and atomic CPU resume."""

from pathlib import Path
import time
import numpy as np
import torch
from .common import digest, json_hash, read_json, write_json
from .features import encode, restore
from .ranking import move_features
from .model import NumpyEvaluator
from .policy_value import build, export, PolicyValue, objective
from .pv_games import generate
from .train import save_state, trusted_load, rng_state, restore_rng


def seed_for(seed, stream, batch, item=0):
    return int(np.random.SeedSequence([seed, stream, batch, item]).generate_state(1)[0])


def dataset(games, config, seed, batch):
    xs, ms, ps, zs = [], [], [], []
    completed = 0
    for index, game in enumerate(games):
        if game["truncated"]:
            continue
        if not game["played"]:
            raise ValueError("Completed game has no post-opening training states")
        completed += 1
        rng = np.random.default_rng(seed_for(seed, 2, batch, index))
        choices = rng.integers(0, len(game["played"]), config["positions_per_game"])
        counts = np.bincount(choices, minlength=len(game["played"]))
        board = restore(game["opening"])
        for i, uci in enumerate(game["played"]):
            legal = list(board.legal_moves)
            visits = np.array(game["visits"][i], dtype=np.float32)
            if (
                len(visits) != len(legal)
                or np.any(visits < 0)
                or not np.isfinite(visits).all()
                or visits.sum() != config["simulations"]
            ):
                raise ValueError("Invalid root visit targets")
            for _ in range(counts[i]):
                xs.append(encode(board, 3))
                ms.append(move_features(board, legal))
                ps.append(visits / visits.sum())
                winner = game["winner"]
                zs.append(0.0 if winner is None else (1.0 if (winner == "white") == board.turn else -1.0))
            board.push_uci(uci)
        outcome = board.outcome(claim_draw=True)
        winner = (
            None if outcome is None or outcome.winner is None else ("white" if outcome.winner else "black")
        )
        if outcome is None or winner != game["winner"]:
            raise ValueError("Outcome/history target mismatch")
    if completed < config["minimum_completed_per_batch"]:
        raise ValueError("Insufficient completed games in batch; quality gate failed")
    width = max(map(len, ms))
    moves = np.zeros((len(ms), width, 3), dtype=np.int64)
    targets = np.zeros((len(ms), width), dtype=np.float32)
    mask = np.zeros((len(ms), width), dtype=bool)
    for i, (m, p) in enumerate(zip(ms, ps)):
        moves[i, : len(m)], targets[i, : len(m)], mask[i, : len(m)] = m, p, True
    return tuple(
        torch.from_numpy(x) for x in (np.stack(xs), moves, mask, targets, np.array(zs, dtype=np.float32))
    )


def train(config, groups, treatment, seed, signature, stop_after_steps=None):
    if config["target_temperature"] != 1.0 or config["anchor_weights"] != {"baseline": 0.0, "anchored": 0.1}:
        raise ValueError("Unsupported treatment/target configuration")
    if treatment not in config["anchor_weights"]:
        raise ValueError("Unknown treatment")
    root = Path(config["output"]) / f"{treatment}-{seed}"
    root.mkdir(parents=True, exist_ok=True)
    initial = config["initializer"]
    model = build(initial, config["policy_seed"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    signature = json_hash(
        dict(study=signature, config=config, treatment=treatment, seed=seed, initial_sha256=digest(initial))
    )
    state_path = root / "resume.pt"
    state = dict(signature=signature, batch=0, phase="generate", cursor=0, history=[], optimizer_steps=0)
    if state_path.exists():
        state = trusted_load(state_path)
        if state["signature"] != signature:
            raise ValueError("Training scientific identity changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
    frozen = NumpyEvaluator(initial)

    def save():
        save_state(
            state_path,
            {**state, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state()},
        )

    steps_this_call = 0
    while state["batch"] < config["batches"]:
        batch = state["batch"] + 1
        metadata = dict(
            initial_sha256=digest(initial),
            policy_seed=config["policy_seed"],
            treatment=treatment,
            run_seed=seed,
            signature=signature,
        )
        actor_path = root / f"actor-{batch:03}.npz"
        if state["phase"] == "generate":
            export(model, actor_path, {**metadata, "batch": batch - 1, "role": "actor"})
        actor = PolicyValue(actor_path)
        game_paths, games = [], []
        for index in range(config["games_per_batch"]):
            path = root / f"games-{batch:03}" / f"{index:04}.json"
            game_seed = seed_for(seed, 1, batch, index)
            opening_rng = np.random.default_rng(seed_for(seed, 0, batch, index))
            opening = groups["train"][int(opening_rng.integers(len(groups["train"])))]
            if path.exists():
                game = read_json(path)
                if (
                    game["actor_sha256"] != actor.checkpoint_id
                    or game["seed"] != game_seed
                    or game["opening"] != opening
                ):
                    raise ValueError("Frozen actor/game identity changed")
            else:
                if state["phase"] != "generate":
                    raise ValueError("Missing committed training game")
                game = generate(actor, opening, config, game_seed)
                write_json(path, game)
                print(
                    f"{treatment}/{seed} batch {batch}: game {index + 1}/{config['games_per_batch']}",
                    flush=True,
                )
            game_paths.append(path)
            games.append(game)
        data_signature = json_hash({str(p): digest(p) for p in game_paths})
        x, moves, mask, targets, outcomes = dataset(games, config, seed, batch)
        init_values = torch.from_numpy(frozen.batch(x.numpy()).astype(np.float32))
        rng = np.random.default_rng(seed_for(seed, 3, batch))
        order = np.concatenate([rng.permutation(len(x)) for _ in range(config["update_epochs"])])
        if state["phase"] == "generate":
            state.update(
                phase="update", cursor=0, data_signature=data_signature, loss_sum=0.0, update_seconds=0.0
            )
            save()
        elif state["data_signature"] != data_signature:
            raise ValueError("Training game data changed")
        model.train()
        while state["cursor"] < len(order):
            ids = order[state["cursor"] : state["cursor"] + config["batch_size"]]
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits, values = model(x[ids], moves[ids], mask[ids])
            loss = objective(
                logits,
                values,
                targets[ids],
                outcomes[ids],
                init_values[ids],
                config["anchor_weights"][treatment],
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            state["loss_sum"] += float(loss.detach()) * len(ids)
            state["cursor"] += len(ids)
            state["optimizer_steps"] += 1
            state["update_seconds"] += time.perf_counter() - started
            save()
            steps_this_call += 1
            if stop_after_steps and steps_this_call >= stop_after_steps:
                return dict(status="interrupted", batch=batch)
        with torch.no_grad():
            _, values = model(x, moves, mask)
            drift = float((values - init_values).square().mean())
            outcome_mse = float((values - outcomes).square().mean())
        checkpoint = root / f"batch-{batch:03}.npz"
        export(model, checkpoint, {**metadata, "batch": batch, "role": "candidate"})
        state["history"].append(
            dict(
                batch=batch,
                checkpoint=str(checkpoint),
                sha256=digest(checkpoint),
                completed=sum(not g["truncated"] for g in games),
                attempted=len(games),
                samples=len(x),
                actor_sha256=actor.checkpoint_id,
                data_sha256=data_signature,
                online_loss=state["loss_sum"] / len(order),
                frozen_checkpoint_outcome_mse=outcome_mse,
                frozen_checkpoint_anchor_mse=drift,
                update_seconds=state["update_seconds"],
            )
        )
        state.update(batch=batch, phase="generate", cursor=0)
        save()
        write_json(
            root / "training.json",
            dict(
                signature=signature,
                treatment=treatment,
                seed=seed,
                history=state["history"],
                optimizer_steps=state["optimizer_steps"],
            ),
        )
    return read_json(root / "training.json")
