"""Fixed-trajectory update ablation: development evidence, not on-policy RL."""

import argparse
import io
import time
from pathlib import Path
import chess.pgn
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .common import read_json, write_json, digest, provenance
from .data import records
from .features import encode, restore
from .model import load_export, export, NumpyEvaluator
from .train import seed_all, trusted_load, save_state, rng_state, restore_rng
from .rl import outcome_value
from .rl_updates import value_loss, frozen_predictions
from .fixed_data import predict
from .diagnostics import metrics
from .evaluate import freeze_protocol, run_matches


def sample_game(record, count, rng):
    """Recover original-history states before generated moves; omit opening prefix."""
    if record["truncated"]:
        return None
    game = chess.pgn.read_game(io.StringIO(record["pgn"]))
    if game is None or game.errors:
        raise ValueError("Invalid saved PGN")
    moves = list(game.mainline_moves())
    start = len(moves) - record["plies"]
    if start < 0:
        raise ValueError("Invalid generated-ply count")
    outcome = game.end().board().outcome(claim_draw=True)
    if outcome is None or outcome.result() != record["result"]:
        raise ValueError("Saved outcome mismatch")
    ids = rng.integers(0, record["plies"], size=count)
    board = game.board()
    selected = {}
    for i, move in enumerate(moves):
        offset = i - start
        if offset in ids:
            selected[offset] = (encode(board, 3), outcome_value(outcome.winner, board.turn))
        board.push(move)
    return (
        np.stack([selected[i][0] for i in ids]),
        np.array([selected[i][1] for i in ids], dtype=np.float32),
    )


def prepare(config, root):
    source = Path(config["source"])
    files = [
        source / f"seed-{seed}/training/games-{b:03}.json"
        for seed in config["seeds"]
        for b in range(1, config["batches"] + 1)
    ]
    identity = dict(
        config=config,
        baseline=digest(config["baseline"]),
        data=digest(config["development_data"]),
        openings=digest(source / "openings.json"),
        games={str(p): digest(p) for p in files},
        code={
            name: digest(Path(__file__).with_name(name + ".py"))
            for name in [
                "rl_update_diagnosis",
                "rl_updates",
                "features",
                "model",
                "train",
                "evaluate",
                "engine",
            ]
        },
    )
    path = root / "protocol.json"
    if path.exists() and read_json(path)["identity"] != identity:
        raise ValueError("Diagnostic config/data/code mismatch")
    if not path.exists():
        write_json(
            path,
            dict(
                identity=identity,
                provenance=provenance(),
                evidence="Off-policy fixed-trajectory development ablation; no new final results",
            ),
        )
    baseline = NumpyEvaluator(config["baseline"])
    for seed in config["seeds"]:
        for batch in range(1, config["batches"] + 1):
            cache = root / f"cache-{seed}-{batch}.npz"
            checksum = cache.with_suffix(".sha256")
            if cache.exists() and checksum.exists():
                if digest(cache) != checksum.read_text().strip():
                    raise ValueError("Cached trajectory data changed")
                continue
            rng = np.random.default_rng(config["sampling_seed"] + seed * 100 + batch)
            samples = [
                sample_game(g, config["positions_per_game"], rng)
                for g in read_json(source / f"seed-{seed}/training/games-{batch:03}.json")
            ]
            samples = [s for s in samples if s is not None]
            x = np.concatenate([s[0] for s in samples])
            y = np.concatenate([s[1] for s in samples])
            p = predict(baseline, x)
            temporary = cache.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, x=x, y=y, initial=p.astype(np.float32))
            temporary.replace(cache)
            checksum.write_text(digest(cache) + "\n")
    return digest(path)


def fit(config, root, name, seed, signature, stop_after=None):
    directory = root / f"{name}-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    treatment = config["treatments"][name]
    seed_all(seed)
    model, metadata = load_export(config["baseline"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=treatment["learning_rate"], weight_decay=0)
    state_path = directory / "resume.pt"
    history = []
    start = 0
    if state_path.exists():
        state = trusted_load(state_path)
        if state["signature"] != signature or state["treatment"] != treatment:
            raise ValueError("Update resume mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        history = state["history"]
        start = state["batch"]
    for batch in range(start + 1, config["batches"] + 1):
        data = np.load(root / f"cache-{seed}-{batch}.npz")
        x = torch.from_numpy(data["x"])
        y = torch.from_numpy(data["y"])
        initial = torch.from_numpy(data["initial"])
        before = frozen_predictions(model, x)
        begin = time.perf_counter()
        model.train()
        for ids in torch.randperm(len(x)).split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = value_loss(model(x[ids]).squeeze(-1), y[ids], initial[ids], treatment["anchor_weight"])
            loss.backward()
            optimizer.step()
        seconds = time.perf_counter() - begin
        after = frozen_predictions(model, x)
        history.append(
            dict(
                batch=batch,
                positions=len(x),
                before_outcome_mse=float(((before - y) ** 2).mean()),
                after_outcome_mse=float(((after - y) ** 2).mean()),
                initial_outcome_mse=float(((initial - y) ** 2).mean()),
                drift_mse=float(((after - initial) ** 2).mean()),
                seconds=seconds,
            )
        )
        export(
            model,
            directory / f"batch-{batch:03}.npz",
            {
                **metadata,
                "kind": "fixed-trajectory-diagnostic",
                "treatment": treatment,
                "seed": seed,
                "source_protocol_sha256": signature,
                "batch": batch,
            },
        )
        save_state(
            state_path,
            dict(
                signature=signature,
                treatment=treatment,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(),
                batch=batch,
                history=history,
            ),
        )
        if stop_after is not None and batch >= stop_after:
            break
    write_json(directory / "history.json", history)
    return directory


def run(config, matches=True):
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        signature = prepare(config, root)
        val = [r for r in records(config["development_data"]) if r["split"] == "validation"]
        xv = np.stack([encode(restore(r), 3) for r in val])
        yv = np.array([r["target"] for r in val])
        baseline_prediction = predict(NumpyEvaluator(config["baseline"]), xv)
        results = []
        for name in config["treatments"]:
            for seed in config["seeds"]:
                print(f"Fit {name} seed {seed}", flush=True)
                directory = fit(config, root, name, seed, signature)
                path = directory / f"batch-{config['batches']:03}.npz"
                prediction = predict(NumpyEvaluator(path), xv)
                row = dict(
                    treatment=name,
                    seed=seed,
                    checkpoint=str(path),
                    sha256=digest(path),
                    validation=metrics(prediction, yv),
                    validation_drift_mse=float(np.mean((prediction - baseline_prediction) ** 2)),
                    prediction_std=float(prediction.std()),
                    history=read_json(directory / "history.json"),
                )
                if matches:
                    protocol = directory / "development.protocol.json"
                    if not protocol.exists():
                        freeze_protocol(
                            str(path),
                            config["baseline"],
                            str(Path(config["source"]) / "openings.json"),
                            config["evaluation"],
                            protocol,
                            "development",
                        )
                    print(f"Development matches: {name} seed {seed}", flush=True)
                    row["development"] = run_matches(protocol, directory / "development")
                results.append(row)
                write_json(root / "progress.json", results)
        write_json(
            root / ("results.json" if matches else "fits.json"),
            dict(
                protocol=read_json(root / "protocol.json"),
                baseline_validation=metrics(baseline_prediction, yv),
                baseline_prediction_std=float(baseline_prediction.std()),
                runs=results,
                paid_cost_cad=0,
                evidence="Development-only fixed-trajectory ablation; no on-policy or final-test claim",
            ),
        )
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    run(read_json(args.config), matches=not args.fit_only)
