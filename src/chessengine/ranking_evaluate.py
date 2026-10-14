"""Frozen development, fixed-depth and fresh final move-ranking comparisons."""

import argparse
import os
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .common import read_json, write_json, digest, provenance
from .evaluate import freeze_protocol, run_matches, evaluator_for
from .features import restore
from .engine import Search
from .ranking import NumpyRanker
from .ranking_train import dataset


def metrics(ranker, data, batch_size=256):
    # Ragged inference avoids allocating a whole-corpus legal-move tensor.
    ce = top1 = top3 = rank_sum = 0.0
    n = len(data["x"])
    for i in range(n):
        start, end = int(data["offsets"][i]), int(data["offsets"][i + 1])
        y = int(data["target"][i])
        scores = ranker.logits(data["x"][i].numpy(), data["moves"][start:end].numpy()).astype(np.float64)
        shifted = scores - scores.max()
        ce += float(np.log(np.exp(shifted).sum()) - shifted[y])
        rank = int(np.where(np.argsort(-scores, kind="stable") == y)[0][0]) + 1
        top1 += rank == 1
        top3 += rank <= 3
        rank_sum += rank
    return dict(cross_entropy=ce / n, top1=top1 / n, top3=top3 / n, mean_rank=rank_sum / n, positions=n)


def run(config, stage="all"):
    root = Path(config["output"])
    baseline = config["baseline"]
    training = read_json(root / "training-results.json")
    if {x["seed"] for x in training} != set(config["seeds"]):
        raise ValueError("Complete all ranker fits before evaluation")
    for row in training:
        if digest(row["checkpoint"]) != row["sha256"]:
            raise ValueError("Selected ranker changed")
        directory = root / f"seed-{row['seed']}"
        bundle = dict(
            value=os.path.relpath(baseline, directory),
            value_sha256=digest(baseline),
            ranker="best.npz",
            ranker_sha256=row["sha256"],
        )
        path = directory / "guided.json"
        if path.exists() and read_json(path) != bundle:
            raise ValueError("Guided bundle changed")
        if not path.exists():
            write_json(path, bundle)
    manifest = read_json("artifacts/models/manifest.json")
    old = next(x for x in manifest["models"] if x["id"] == manifest["default"])
    released = str(Path("artifacts/models") / old["file"])
    if digest(released) != old["sha256"]:
        raise ValueError("Released reference checksum mismatch")
    identity = dict(
        config=config,
        baseline=digest(baseline),
        released=digest(released),
        openings=digest(root / "openings.json"),
        rankers={str(r["seed"]): r["sha256"] for r in training},
        code={
            n: digest(Path(__file__).with_name(n + ".py"))
            for n in ["ranking", "ranking_evaluate", "engine", "evaluate", "features"]
        },
    )
    snapshot = root / "evaluation-protocol.json"
    if snapshot.exists() and read_json(snapshot)["identity"] != identity:
        raise ValueError("Ranking evaluation changed")
    if not snapshot.exists():
        write_json(snapshot, dict(identity=identity, provenance=provenance()))

    def match(name, candidate, opponent, split, pairs, nodes, time_ms):
        protocol = root / "evaluation" / f"{name}.protocol.json"
        settings = dict(
            seed=2026,
            pairs=pairs,
            nodes=nodes,
            time_ms=time_ms,
            qdepth=2,
            max_plies=config["max_plies"],
            evidence="move-ranking-" + stage,
        )
        if not protocol.exists():
            freeze_protocol(candidate, opponent, str(root / "openings.json"), settings, protocol, split)
        print("Matches:", name, flush=True)
        return run_matches(protocol, root / "evaluation" / name)

    def guided(seed):
        return "guided:" + str(root / f"seed-{seed}/guided.json")

    with threadpool_limits(limits=1):
        if stage == "benchmark":
            result = {}
            for label, nodes, ms in [
                ("nodes", config["evaluation_nodes"], None),
                ("time", 1000000000, config["evaluation_time_ms"]),
            ]:
                result[label] = match("benchmark-" + label, guided(42), baseline, "development", 5, nodes, ms)
            write_json(root / "evaluation-benchmark.json", result)
            return result
        data, _ = dataset(config)
        results = []
        for seed in config["seeds"]:
            ranker = NumpyRanker(root / f"seed-{seed}/best.npz")
            row = dict(
                seed=seed,
                checkpoint_sha256=ranker.checkpoint_id,
                classification=metrics(ranker, data["validation"]),
            )
            for label, nodes, ms in [
                ("nodes", config["evaluation_nodes"], None),
                ("time", 1000000000, config["evaluation_time_ms"]),
            ]:
                row[label] = match(
                    f"development-{seed}-{label}",
                    guided(seed),
                    baseline,
                    "development",
                    config["development_pairs"],
                    nodes,
                    ms,
                )
            results.append(row)
        release = match(
            "release-check",
            baseline,
            released,
            "development",
            config["development_pairs"],
            1000000000,
            config["evaluation_time_ms"],
        )
        # Representative seed fixed before fitting. Final outcome cannot change selection.
        lock = dict(seed=42, ranker_sha256=digest(root / "seed-42/best.npz"), value_sha256=digest(baseline))
        lock_path = root / "final-selection.json"
        if lock_path.exists() and read_json(lock_path) != lock:
            raise ValueError("Final ranker selection changed")
        write_json(lock_path, lock)
        final = match(
            "final-42-time",
            guided(42),
            baseline,
            "test",
            config["final_pairs"],
            1000000000,
            config["evaluation_time_ms"],
        )
        depth_path = root / "depth-comparison.json"
        depth_rows = read_json(depth_path) if depth_path.exists() else []
        done = {(r["opening_id"], r["depth"], r["engine"]) for r in depth_rows}
        for opening in read_json(root / "openings.json")["development"][: config["development_pairs"]]:
            for depth in [2, 3]:
                for name, spec in [("ordinary", baseline), ("guided", guided(42))]:
                    if (opening["id"], depth, name) in done:
                        continue
                    result = Search(
                        evaluator_for(spec), nodes=100000, time_ms=3000, max_depth=depth, qdepth=2
                    ).run(restore(opening))
                    depth_rows.append(
                        dict(
                            opening_id=opening["id"],
                            depth=depth,
                            engine=name,
                            completed=result.depth >= depth,
                            nodes=result.nodes,
                            elapsed_ms=result.elapsed_ms,
                        )
                    )
                    write_json(depth_path, depth_rows)
        result = dict(
            development=results,
            release_check=release,
            release_eligible=release["score"] >= 0.5,
            final=final,
            selection=lock,
            depth_comparison=depth_rows,
            protocol=read_json(snapshot),
            paid_cost_cad=0,
        )
        write_json(root / "results.json", result)
        return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--stage", choices=["benchmark", "all"], default="all")
    args = p.parse_args()
    run(read_json(args.config), args.stage)
