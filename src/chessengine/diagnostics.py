"""Development-only supervised diagnosis, frozen protocols and sequential experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time

import chess
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .common import digest, json_hash, provenance, read_json, write_json
from .data import records, validate_splits
from .engine import Handcrafted, VALUES
from .features import encode, restore, FEATURE_VERSION
from .model import NumpyEvaluator
from .train import fit


def development_rows(source):
    # Test targets are never passed to training, predictions, diagnostics, or selection.
    rows = [r for r in records(source) if r["split"] in ("train", "validation")]
    validate_splits(rows)
    if not all(any(r["split"] == s for r in rows) for s in ("train", "validation")):
        raise ValueError("Training and validation records are required")
    return rows


def nested_subsets(rows, sizes, seed):
    games = defaultdict(list)
    for i, row in enumerate(rows):
        if row["split"] == "train":
            games[row["game_id"]].append(i)
    ordered = sorted(games, key=lambda g: (json_hash([seed, g]), g))
    result = {}
    for size in sizes:
        chosen = []
        for game in ordered:
            chosen.extend(games[game])
            if size != "all" and len(chosen) >= int(size):
                break
        result[str(size)] = chosen
    return result


def metrics(predictions, targets):
    error = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    if not len(error):
        raise ValueError("Empty metric group")
    return {"mse": float(np.mean(error ** 2)), "mae": float(np.mean(abs(error))),
            "signed_error": float(np.mean(error))}


def grouped_bootstrap(predictions, targets, games, resamples=2000, seed=2026):
    """Resample games; position-weighted metric within each sampled game collection."""
    names = list(predictions)
    _, group = np.unique(games, return_inverse=True)
    counts = np.bincount(group)
    sums = []
    for name in names:
        error = np.asarray(predictions[name]) - targets
        sums.append(np.stack([np.bincount(group, weights=e, minlength=len(counts))
                              for e in (error ** 2, abs(error), error)], axis=1))
    sums = np.stack(sums)
    rng = np.random.default_rng(seed)
    estimates = np.empty((resamples, len(names), 3))
    for i in range(resamples):
        ids = rng.integers(len(counts), size=len(counts))
        estimates[i] = sums[:, ids, :].sum(axis=1) / counts[ids].sum()
    def ci(x):
        return np.quantile(x, [.025, .975]).tolist()
    intervals = {
        name: {metric: ci(estimates[:, j, k]) for k, metric in enumerate(("mse", "mae", "signed_error"))}
        for j, name in enumerate(names)
    }
    neural = names.index("supervised")
    differences = {}
    for j, name in enumerate(names):
        if name == "supervised":
            continue
        differences[name] = {
            metric: ci(estimates[:, neural, k] - estimates[:, j, k])
            for k, metric in enumerate(("mse", "mae", "signed_error"))
        }
    return {"intervals": intervals, "paired_supervised_minus_baseline": differences,
            "resamples": resamples, "seed": seed, "unit": "source game",
            "estimand": "position-weighted metrics; games sampled with replacement"}


def strata(board, row):
    magnitude = abs(row["target"])
    target_bin = ("[0,.25)" if magnitude < .25 else "[.25,.5)" if magnitude < .5
                  else "[.5,.75)" if magnitude < .75 else "[.75,1]")
    units = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
    material = sum(units.get(p.piece_type, 0) for p in board.piece_map().values())
    balance = abs(sum(VALUES[p.piece_type] * (1 if p.color else -1)
                      for p in board.piece_map().values()))
    return {"side": "white" if board.turn else "black", "move_count_phase": row["phase"],
            "target_magnitude": target_bin,
            "nonpawn_material": "0-8" if material <= 8 else "9-16" if material <= 16 else "17+",
            "material_imbalance": "<=1" if balance <= 1 + 1e-8 else "1-3" if balance <= 3 + 1e-8 else ">3"}


def screen(full_runs, config):
    return sum(r["validation"]["mse"] > config["gap_ratio"] * r["train"]["mse"]
               for r in full_runs) >= config["gap_seeds_required"]


def summarize_runs(runs, config):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["experiment"], run["size"])].append(run)
    summaries = []
    baseline = {r["seed"]: r for r in runs if r["experiment"] == "baseline" and r["size"] == "all"}
    baseline_mean = np.mean([r["validation"]["mse"] for r in baseline.values()])
    for (name, size), group in groups.items():
        errors = [r["validation"]["mse"] for r in group]
        mean = float(np.mean(errors))
        relative = float(1 - mean / baseline_mean) if baseline_mean else None
        improved = sum(r["validation"]["mse"] < baseline[r["seed"]]["validation"]["mse"] for r in group)
        summaries.append({"experiment": name, "size": size, "positions": group[0]["positions"],
                          "mean_mse": mean, "std_mse": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0,
                          "relative_improvement_vs_full_baseline": relative,
                          "improved_seeds": improved,
                          "promising": name != "baseline" and relative is not None
                          and relative >= config["promising_relative_improvement"]
                          and improved >= config["promising_seeds_required"]})
    return sorted(summaries, key=lambda r: r["mean_mse"])


def run_study(config, resume=False):
    started = time.perf_counter()
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    code_hashes = {name: digest(Path(__file__).with_name(name + ".py"))
                   for name in ("diagnostics", "train", "model", "features", "data", "engine")}
    identity = {"version": 1, "config": config, "source_sha256": digest(config["source"]),
                "checkpoint_sha256": digest(config["checkpoint"]), "feature_version": FEATURE_VERSION,
                "code_sha256": code_hashes}
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if not resume:
            raise ValueError("Study exists; use --resume")
        if read_json(protocol_path)["identity"] != identity:
            raise ValueError("Study data/config/code mismatch")
    else:
        write_json(protocol_path, {"identity": identity, "provenance": provenance()})
    rows = development_rows(config["source"])
    print(f"Encoding {len(rows)} development positions", flush=True)
    boards_features, validation_rows, validation_strata, handcrafted = [], [], [], []
    for row in rows:
        board = restore(row)
        boards_features.append(encode(board))
        if row["split"] == "validation":
            validation_rows.append(row)
            validation_strata.append(strata(board, row))
            handcrafted.append(Handcrafted()(board))
    x = np.stack(boards_features)
    del boards_features
    y = np.array([r["target"] for r in rows], dtype=np.float32)
    train_ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
    validation_ids = [i for i, r in enumerate(rows) if r["split"] == "validation"]
    xv, yv = x[validation_ids], y[validation_ids]
    mean_target = float(y[train_ids].mean(dtype=np.float64))
    subsets = nested_subsets(rows, config["subset_sizes"], config["sampling_seed"])
    with threadpool_limits(limits=1):
        frozen = NumpyEvaluator(config["checkpoint"])
        predictions = {"supervised": frozen.batch(xv).astype(float), "zero": np.zeros(len(yv)),
                       "training_mean": np.full(len(yv), mean_target), "handcrafted": np.array(handcrafted)}
        audit = {"train": metrics(frozen.batch(x[train_ids]), y[train_ids]),
                 "validation": {name: metrics(pred, yv) for name, pred in predictions.items()},
                 "training_mean": mean_target, "training_positions": len(train_ids),
                 "validation_positions": len(yv), "validation_games": len({r["game_id"] for r in validation_rows})}
    reference_mse = audit["validation"]["training_mean"]["mse"]
    for result in audit["validation"].values():
        result["relative_mse_reduction_vs_training_mean"] = 1 - result["mse"] / reference_mse if reference_mse else None
    audit["bootstrap"] = grouped_bootstrap(predictions, yv.astype(float),
                                            [r["game_id"] for r in validation_rows],
                                            config["bootstrap_resamples"], config["bootstrap_seed"])
    breakdown = {}
    for dimension in validation_strata[0]:
        breakdown[dimension] = {}
        for label in sorted({s[dimension] for s in validation_strata}):
            ids = [i for i, s in enumerate(validation_strata) if s[dimension] == label]
            breakdown[dimension][label] = {
                "positions": len(ids), "games": len({validation_rows[i]["game_id"] for i in ids}),
                "sparse": len(ids) < config["sparse_threshold"],
                "metrics": {name: metrics(pred[ids], yv[ids]) for name, pred in predictions.items()}}
    audit["breakdown"] = breakdown
    worst, seen = [], set()
    for i in np.argsort(-(predictions["supervised"] - yv) ** 2, kind="stable"):
        row = validation_rows[i]
        if row["game_id"] in seen:
            continue
        seen.add(row["game_id"])
        worst.append({"game_id": row["game_id"], "fen": row["fen"], "target": float(yv[i]),
                      "prediction": float(predictions["supervised"][i]), "strata": validation_strata[i]})
        if len(worst) == 20:
            break
    audit["worst_positions"] = worst
    np.savez_compressed(output / "predictions.npz", targets=yv,
                        game_ids=np.array([r["game_id"] for r in validation_rows]), **predictions)
    write_json(output / "audit.json", audit)
    write_json(output / "subsets.json", {size: {"positions": len(ids),
               "games": len({rows[i]["game_id"] for i in ids}),
               "membership_sha256": json_hash([rows[i] for i in ids])} for size, ids in subsets.items()})
    completed = []

    def experiment(name, size, settings):
        ids = subsets[size]
        dataset_path = output / f"subset-{size}.jsonl"
        # Dataset ordering is identical across training seeds and regularizers.
        import json
        if not dataset_path.exists():
            with open(dataset_path.with_suffix(".tmp"), "w") as stream:
                for i in [*ids, *validation_ids]:
                    stream.write(json.dumps(rows[i]) + "\n")
            dataset_path.with_suffix(".tmp").replace(dataset_path)
        expected_hash = json_hash([rows[i] for i in [*ids, *validation_ids]])
        if json_hash(list(records(dataset_path))) != expected_hash:
            raise ValueError("Derived subset data mismatch")
        subset_sha = digest(dataset_path)
        prepared = {"source_sha256": subset_sha, "subsets": {
            "train": (torch.from_numpy(x[ids]), torch.from_numpy(y[ids])),
            "validation": (torch.from_numpy(xv), torch.from_numpy(yv))}}
        for seed in config["seeds"]:
            destination = output / f"{name}-{size}-seed-{seed}"
            result_path = destination / "result.json"
            training = {**config["training"], **settings, "seed": seed}
            if result_path.exists():
                result = read_json(result_path)
                if result["config"] != training or result["checkpoint_sha256"] != digest(destination / "best.npz"):
                    raise ValueError("Completed run configuration/checkpoint mismatch")
                completed.append(result)
                continue
            print(f"Training {name}, size={len(ids)}, seed={seed}", flush=True)
            run_start = time.perf_counter()
            fit(dataset_path, destination, training, resume=(destination / "resume.pt").exists(), prepared=prepared)
            with threadpool_limits(limits=1):
                evaluator = NumpyEvaluator(destination / "best.npz")
                tr = metrics(evaluator.batch(x[ids]), y[ids])
                va = metrics(evaluator.batch(xv), yv)
            history = read_json(destination / "metrics.json")["history"]
            result = {"experiment": name, "size": size, "positions": len(ids), "seed": seed,
                      "config": training, "train": tr, "validation": va,
                      "selected_epoch": evaluator.metadata["epoch"], "epochs": len(history),
                      "optimizer_steps": history[-1]["optimizer_steps"],
                      "training_seconds": history[-1]["update_seconds"],
                      "run_wall_seconds": time.perf_counter() - run_start,
                      "wall_time_note": "completed invocation; interrupted attempts excluded",
                      "data_sha256": subset_sha, "checkpoint_sha256": evaluator.checkpoint_id,
                      "provenance": provenance()}
            write_json(result_path, result)
            completed.append(result)
            print(f"  train={tr['mse']:.5f} validation={va['mse']:.5f}", flush=True)

    for size in subsets:
        experiment("baseline", size, {})
    full = [r for r in completed if r["size"] == "all"]
    triggered = screen(full, config)
    write_json(output / "screening.json", {"triggered": triggered, "ratio": config["gap_ratio"],
               "required_seeds": config["gap_seeds_required"],
               "per_seed": [{"seed": r["seed"], "train_mse": r["train"]["mse"],
                             "validation_mse": r["validation"]["mse"],
                             "exceeds": r["validation"]["mse"] > config["gap_ratio"] * r["train"]["mse"]}
                            for r in full]})
    print(f"Regularization screening triggered: {triggered}", flush=True)
    if triggered:
        for setting in config["regularization"]:
            experiment(setting["name"], "all", {k: v for k, v in setting.items() if k != "name"})
    result = {"version": 1, "protocol": read_json(protocol_path), "runs": completed,
              "summary": summarize_runs(completed, config), "screening": read_json(output / "screening.json"),
              "invocation_wall_seconds": time.perf_counter() - started,
              "training_seconds": sum(r["training_seconds"] for r in completed),
              "paid_cost_cad": 0, "provenance": provenance()}
    write_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_study(read_json(args.config), args.resume)


if __name__ == "__main__":
    main()
