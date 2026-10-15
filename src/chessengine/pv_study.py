"""Preregistered, resumable local policy/value study. No production model promotion."""

import argparse
import html
import resource
import sys
import time
from pathlib import Path
import chess
import numpy as np
from threadpoolctl import threadpool_limits
from .common import digest, json_hash, read_json, write_json, provenance
from .features import record_board
from .policy_value import build, export, PolicyValue
from .pv_games import generate, matches
from .pv_data import prepare
from .pv_train import train
from .pv_audit import completed
from .rl_improvement import aggregate_finals


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if sys.platform == "darwin" else 1024
    )


def seal(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen scientific identity changed: {path.name}")
    else:
        write_json(path, value)


def initialize(config, root):
    path = root / "initial.npz"
    meta = dict(
        initial_sha256=digest(config["initializer"]),
        policy_seed=config["policy_seed"],
        role="common-initialization",
    )
    if not path.exists():
        export(build(config["initializer"], config["policy_seed"]), path, meta)
    evaluator = PolicyValue(path)
    if any(evaluator.metadata.get(k) != v for k, v in meta.items()):
        raise ValueError("Frozen initializer identity changed")
    return path


def benchmark(config):
    root = Path(config["output"]) / "benchmark"
    root.mkdir(parents=True, exist_ok=True)
    initial = initialize(config, root)
    cfg = {**config, "simulations": config["pilot_simulations"][0], "max_plies": 32}
    game = generate(
        PolicyValue(initial),
        {**record_board(chess.Board()), "id": "disposable-benchmark"},
        cfg,
        config["pilot_seed"],
    )
    seconds_per_ply = game["seconds"] / max(1, len(game["played"]))
    # Upper-envelope plies, linear simulation scaling, and 50% margin; not a main-run forecast.
    forecast = (
        1.5
        * config["pilot_games"]
        * config["max_plies"]
        * seconds_per_ply
        * sum(n / cfg["simulations"] for n in config["pilot_simulations"])
    )
    result = dict(
        measured_plies=len(game["played"]),
        measured_seconds=game["seconds"],
        seconds_per_ply=seconds_per_ply,
        pilot_upper_envelope_seconds=forecast,
        peak_ram_mb=peak_mb(),
        paid_cost_cad=0,
    )
    write_json(root / "benchmark.json", result)
    return result


def pilot(config, groups, initial, root):
    if (root / "pilot-seal.json").exists():
        saved = read_json(root / "pilot-seal.json")
        for name, sha in saved.items():
            if digest(root / name) != sha:
                raise ValueError("Frozen pilot evidence changed")
        for row in read_json(root / "pilot-summary.json"):
            for i, sha in enumerate(row["game_hashes"]):
                if digest(root / f"pilot-{row['simulations']}" / f"{i:04}.json") != sha:
                    raise ValueError("Frozen pilot game changed")
        return read_json(root / "pilot-decision.json")
    actor = PolicyValue(initial)
    summaries = []
    for simulations in config["pilot_simulations"]:
        games = []
        cfg = {**config, "simulations": simulations}
        for i in range(config["pilot_games"]):
            path = root / f"pilot-{simulations}" / f"{i:04}.json"
            seed = config["pilot_seed"] + i
            if path.exists():
                game = read_json(path)
                if (
                    game["actor_sha256"] != actor.checkpoint_id
                    or game["opening"] != groups["pilot"][i]
                    or game["seed"] != seed
                ):
                    raise ValueError("Pilot game identity mismatch")
            else:
                game = generate(actor, groups["pilot"][i], cfg, seed)
                write_json(path, game)
            games.append(game)
            print(
                f"Pilot {simulations}: {i + 1}/{config['pilot_games']}; complete={not game['truncated']}; {game['seconds']:.1f}s",
                flush=True,
            )
        moves = [m for g in games for m in g["metrics"] if m["legal"] >= 4]
        row = dict(
            simulations=simulations,
            attempted=len(games),
            completed=sum(not g["truncated"] for g in games),
            mean_game_seconds=float(np.mean([g["seconds"] for g in games])),
            p90_plies=float(np.quantile([len(g["played"]) for g in games], 0.9)),
            root_coverage=float(np.mean([m["root_coverage"] for m in moves])) if moves else 0.0,
            dominant_fraction=float(np.mean([m["root_max_share"] >= 0.99 for m in moves])) if moves else 1.0,
            mean_depth=float(np.mean([m["max_depth"] for m in moves])) if moves else 0.0,
            peak_ram_mb=peak_mb(),
            game_hashes=[digest(root / f"pilot-{simulations}" / f"{i:04}.json") for i in range(len(games))],
        )
        row["qualifies"] = (
            row["completed"] / row["attempted"] >= config["pilot_min_completion"]
            and row["root_coverage"] >= config["pilot_min_root_coverage"]
            and row["dominant_fraction"] < config["pilot_max_dominant_fraction"]
            and row["peak_ram_mb"] <= config["pilot_max_ram_mb"]
        )
        summaries.append(row)
        write_json(root / "pilot-summary.json", summaries)
    eligible = [r for r in summaries if r["qualifies"]]
    decision = dict(
        approved=bool(eligible),
        simulations=max(r["simulations"] for r in eligible) if eligible else None,
        reason="Highest quality-gated simulation budget"
        if eligible
        else "No pilot satisfies the frozen completion, coverage, degeneracy and memory criteria",
    )
    seal(root / "pilot-decision.json", decision)
    if eligible:
        row = next(r for r in eligible if r["simulations"] == decision["simulations"])
        generation = (
            2
            * len(config["seeds"])
            * config["batches"]
            * config["games_per_batch"]
            * row["mean_game_seconds"]
        )
        development_games = (
            2 * len(config["seeds"]) * len(config["development_batches"]) * 2 * config["development_pairs"]
        )
        final_games = len(config["seeds"]) * (
            2 * config["final_pairs"] + 4 * config["control_pairs"] + 4 * config["reference_pairs"]
        )
        estimated_reference = (
            4
            * len(config["seeds"])
            * config["reference_pairs"]
            * row["p90_plies"]
            * config["equal_time_ms"]
            / 1000
        )
        forecast = 1.5 * (
            generation
            + (development_games + final_games - 4 * len(config["seeds"]) * config["reference_pairs"])
            * row["mean_game_seconds"]
            + estimated_reference
            + 2 * len(config["seeds"]) * config["batches"] * 20
        )
        write_json(
            root / "forecast.json",
            dict(
                seconds=forecast,
                generation_seconds=generation,
                development_games=development_games,
                final_games=final_games,
                margin=1.5,
                note="Pilot-throughput estimate; deterministic match lengths can differ substantially",
                paid_cost_cad=0,
            ),
        )
    seal(
        root / "pilot-seal.json",
        {
            name: digest(root / name)
            for name in ["pilot-summary.json", "pilot-decision.json"]
            + (["forecast.json"] if eligible else [])
        },
    )
    return decision


def evaluate(config, groups, root, initial, signature):
    cfg = {**config, "time_ms": None}
    selections = {}
    for treatment in config["anchor_weights"]:
        for seed in config["seeds"]:
            directory = root / f"{treatment}-{seed}"
            candidates = []
            for batch in config["development_batches"]:
                checkpoint = directory / f"batch-{batch:03}.npz"
                summary = matches(
                    "puct:" + str(checkpoint),
                    "puct:" + str(initial),
                    groups["development"],
                    cfg,
                    directory / f"development-{batch}",
                    "development",
                )
                candidates.append(
                    dict(
                        batch=batch,
                        checkpoint=str(checkpoint),
                        sha256=digest(checkpoint),
                        score=summary["score"],
                    )
                )
            selected = max(candidates, key=lambda c: (c["score"], -c["batch"]))
            seal(
                directory / "selection.json",
                dict(signature=signature, candidates=candidates, selected=selected),
            )
            selections[f"{treatment}-{seed}"] = dict(
                selected=selected, selection_sha256=digest(directory / "selection.json")
            )
    seal(root / "selection-lock.json", selections)
    reference = read_json("artifacts/models/manifest.json")
    reference_row = next(r for r in reference["models"] if r["id"] == reference["default"])
    reference_path = Path("artifacts/models") / reference_row["file"]
    if digest(reference_path) != reference_row["sha256"]:
        raise ValueError("Released reference checksum mismatch")
    primary_sets, rows = [], []
    for seed in config["seeds"]:
        selected = {t: selections[f"{t}-{seed}"]["selected"]["checkpoint"] for t in config["anchor_weights"]}
        directory = root / f"final-{seed}"
        primary = matches(
            "puct:" + selected["anchored"],
            "puct:" + selected["baseline"],
            groups["final"],
            cfg,
            directory / "primary",
            "final",
        )
        primary_sets.append(read_json(directory / "primary/matches.json")["games"])
        controls = {}
        for treatment, candidate in selected.items():
            controls[treatment] = dict(
                initial=matches(
                    "puct:" + candidate,
                    "puct:" + str(initial),
                    groups["control"],
                    cfg,
                    directory / f"{treatment}-initial",
                    "final",
                ),
                released=matches(
                    "puct:" + candidate,
                    str(reference_path),
                    groups["reference"],
                    {**cfg, "time_ms": config["equal_time_ms"], "simulations": 1000000000},
                    directory / f"{treatment}-released",
                    "final",
                ),
            )
        rows.append(dict(seed=seed, primary=primary, secondary=controls))
    aggregate = aggregate_finals(primary_sets, config["bootstrap_resamples"], config["bootstrap_seed"])
    if aggregate["conclusion"] == "improvement" and sum(x > 0.5 for x in aggregate["seed_scores"]) < 2:
        aggregate["conclusion"] = "inconclusive"
    result = dict(
        status="complete",
        signature=signature,
        selections=selections,
        seeds=rows,
        aggregate=aggregate,
        released_reference=reference_row["sha256"],
        paid_cost_cad=0,
    )
    write_json(root / "results.json", result)
    return result


def report(config, destination="docs/experiments/policy-value"):
    root, out = Path(config["output"]), Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    evidence = dict(
        status=read_json(root / "status.json")
        if (root / "status.json").exists()
        else {"status": "not_started"},
        paid_cost_cad=0,
        model_promotion=False,
    )
    for name in ["pilot-summary", "pilot-decision", "forecast", "compute", "results", "final-audit"]:
        if (root / f"{name}.json").exists():
            evidence[name] = read_json(root / f"{name}.json")

    def clean(value):
        if isinstance(value, dict):
            return {
                (
                    "[private research artifact]" if str(k).startswith(("/Users/", "runs/", "data/")) else k
                ): clean(v)
                for k, v in value.items()
                if k not in ("initializer", "archive", "corpus", "reserved", "output")
            }
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str) and value.startswith(("/Users/", "runs/", "data/")):
            return "[private research artifact]"
        return value

    evidence = clean(evidence)
    write_json(out / "evidence.json", evidence)
    status = evidence["status"]["status"]
    text = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Policy/value self-play study</title><style>body{font:16px/1.6 system-ui;max-width:1050px;margin:36px auto;padding:0 20px;color:#23382f}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ccd}pre{white-space:pre-wrap;overflow-wrap:anywhere}.table-wrap{overflow:auto}</style><h1>Policy/value self-play and prediction anchoring</h1>'
    text += f"<p>Status: <strong>{html.escape(status)}</strong>. Local CPU only; C$0 paid resources. Released models remain unchanged.</p>"
    text += "<p>This adapted AlphaZero-inspired study compares visit-policy and outcome learning with one 0.1 prediction-anchored variant. Both start from the same supervised value network and fresh policy head. Each actor is frozen per batch. Pilot results are feasibility evidence, not final playing-strength evidence.</p>"
    if "pilot-summary" in evidence:
        text += '<h2>Frozen feasibility pilot</h2><div class="table-wrap"><table><tr><th>Simulations</th><th>Completed</th><th>Root coverage</th><th>Dominant positions</th><th>Seconds/game</th><th>Qualified</th></tr>'
        for r in evidence["pilot-summary"]:
            text += f"<tr><td>{r['simulations']}</td><td>{r['completed']}/{r['attempted']}</td><td>{r['root_coverage']:.1%}</td><td>{r['dominant_fraction']:.1%}</td><td>{r['mean_game_seconds']:.2f}</td><td>{r['qualifies']}</td></tr>"
        text += "</table></div><p>Coverage is measured on positions with at least four legal moves. All three frozen criteria plus the memory limit must pass. Rejected pilots never enter training.</p>"
    if "results" in evidence:
        a = evidence["results"]["aggregate"]
        text += f"<h2>Primary final result</h2><p>Anchored match score: {a['score']:.2%}; paired 95% interval {a['ci95'][0]:.2%}–{a['ci95'][1]:.2%}. {html.escape(a['conclusion'])}. Seed SD: {a['seed_sd']:.4f}.</p><pre>{html.escape(str(evidence['results']['seeds']))}</pre>"
    else:
        text += "<h2>Final evidence</h2><p>No completed final comparison is claimed. There are no invented curves, Elo estimates or model-promotion decisions.</p>"
    text += '<h2>Protocol and limitations</h2><p>The protocol fixes seeds 42/43/44, two treatments, uniform game weighting, completed-game targets only, source-game-separated starts, development-only checkpoint selection and fresh final pairs. Anchor coefficient 0.1 is distinct from the earlier coefficient-1 off-policy diagnostic. Values are side-to-move advantage estimates, not calibrated win probabilities. Simulations and alpha-beta nodes are different work units. Cap sensitivity and seed variation must accompany final scores.</p><p><a href="evidence.json">Machine-readable evidence</a> · <a href="protocol.json">Frozen protocol</a></p></html>'
    (out / "index.html").write_text(text)
    if (root / "protocol.json").exists():
        p = read_json(root / "protocol.json")
        write_json(out / "protocol.json", clean(p))
    else:
        write_json(out / "protocol.json", dict(config=clean(config), status="not frozen"))
    return evidence


def run(config, stage="pilot"):
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    old_seconds = (
        read_json(root / "compute.json")["active_seconds"] if (root / "compute.json").exists() else 0.0
    )
    try:
        if stage == "all" and not (root / "forecast.json").exists():
            raise ValueError("Run the pilot stage and report its measured forecast before the full launch")
        inputs = {
            p: digest(p)
            for p in [
                config["initializer"],
                config["corpus"],
                config["archive"],
                *config["reserved"],
                "artifacts/models/manifest.json",
            ]
        }
        code = {
            name: digest(Path(__file__).with_name(name + ".py"))
            for name in [
                "pv_study",
                "pv_audit",
                "pv_data",
                "pv_train",
                "pv_games",
                "puct",
                "policy_value",
                "features",
                "ranking",
                "model",
                "engine",
                "evaluate",
                "rl_improvement",
                "train",
                "common",
                "data",
            ]
        }
        identity = dict(
            config=config,
            inputs=inputs,
            code=code,
            protocol_sha256=digest("docs/policy-value-preregistration.md"),
        )
        signature = json_hash(identity)
        protocol_path = root / "protocol.json"
        if protocol_path.exists():
            if read_json(protocol_path)["identity"] != identity:
                raise ValueError("Frozen scientific protocol changed")
        else:
            write_json(protocol_path, dict(identity=identity, provenance=provenance()))
        with threadpool_limits(limits=1):
            import torch

            torch.set_num_threads(1)
            groups = prepare(config)
            seal(
                root / "opening-seal.json",
                dict(
                    sha256=digest(root / "openings.json"),
                    manifest_sha256=digest(root / "openings-manifest.json"),
                ),
            )
            initial = initialize(config, root)
            seal(root / "initial-seal.json", dict(sha256=digest(initial)))
            if stage == "prepare":
                write_json(root / "status.json", dict(status="prepared", signature=signature))
                return dict(status="prepared")
            decision = pilot(config, groups, initial, root)
            if not decision["approved"]:
                write_json(
                    root / "status.json",
                    dict(status="pilot_rejected", signature=signature, reason=decision["reason"]),
                )
                return decision
            write_json(root / "status.json", dict(status="pilot_complete", signature=signature))
            if stage == "pilot":
                return {**decision, "forecast": read_json(root / "forecast.json")}
            cfg = {**config, "simulations": decision["simulations"]}
            seal(
                root / "main-protocol.json",
                dict(signature=signature, config=cfg, decision_sha256=digest(root / "pilot-decision.json")),
            )
            # Full launch is a separate CLI stage so a forecast can be communicated first.
            for treatment in cfg["anchor_weights"]:
                for seed in cfg["seeds"]:
                    write_json(
                        root / "status.json",
                        dict(status="training", treatment=treatment, seed=seed, signature=signature),
                    )
                    train(cfg, groups, treatment, seed, signature)
            write_json(root / "status.json", dict(status="evaluation", signature=signature))
            result = evaluate(cfg, groups, root, initial, signature)
            for path, sha in inputs.items():
                if digest(path) != sha:
                    raise ValueError("Study input changed during execution")
            audit = completed(cfg, groups, root)
            write_json(root / "final-audit.json", audit)
            if not audit["complete"]:
                raise ValueError("Incomplete training audit")
            write_json(root / "status.json", dict(status="complete", signature=signature))
            return result
    except Exception as error:
        failures_path = root / "failures.json"
        failures = read_json(failures_path) if failures_path.exists() else []
        failures.append(dict(reason=str(error), provenance=provenance()))
        write_json(failures_path, failures)
        if "identity changed" not in str(error).lower() and "protocol changed" not in str(error).lower():
            write_json(root / "status.json", dict(status="failed", reason=str(error)))
        raise
    finally:
        write_json(
            root / "compute.json",
            dict(
                active_seconds=old_seconds + time.perf_counter() - started,
                peak_ram_mb=peak_mb(),
                paid_cost_cad=0,
            ),
        )
        report(config)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--stage", choices=["benchmark", "prepare", "pilot", "all", "report"], default="pilot")
    args = p.parse_args()
    config = read_json(args.config)
    if args.stage == "benchmark":
        with threadpool_limits(limits=1):
            print(benchmark(config))
    elif args.stage == "report":
        report(config)
    else:
        print(run(config, args.stage))
        from .research_catalog import publish

        publish()


if __name__ == "__main__":
    main()
