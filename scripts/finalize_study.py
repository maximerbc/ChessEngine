"""Aggregate all seeds, run secondary checks, and generate an auditable final report."""

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from chessengine.common import read_json, write_json
from chessengine.evaluate import freeze_protocol, run_matches, measure
from chessengine.report import generate

p = argparse.ArgumentParser()
p.add_argument("--root", default="runs/local-study")
p.add_argument("--baseline", default="runs/main-supervised/best.npz")
p.add_argument("--dataset", default="data/main-labeled.jsonl")
p.add_argument("--stockfish")
a = p.parse_args()
root = Path(a.root)
config = read_json("configs/local-study.json")
runs = []
for seed in config["seeds"]:
    result = read_json(root / f"seed-{seed}" / "result.json")
    runs.append(result)
representative = max(runs, key=lambda r: r["selection"]["development_score"])
seed = representative["seed"]
checkpoint = representative["selection"]["checkpoint"]
secondary = {}
with threadpool_limits(limits=1):
    comparisons = [
        ("handcrafted", "handcrafted", {**read_json(config["development"]), "reference_nodes": 1000}),
        ("equal-time", a.baseline, {**read_json(config["development"]), "nodes": 100000000, "time_ms": 30}),
    ]
    if a.stockfish:
        comparisons.append(
            (
                "stockfish",
                "stockfish:" + a.stockfish,
                {**read_json(config["development"]), "reference_nodes": 1000},
            )
        )
    for name, opponent, cfg in comparisons:
        protocol = root / f"secondary-{name}.protocol.json"
        if not protocol.exists():
            freeze_protocol(checkpoint, opponent, config["openings"], cfg, protocol, "test")
        secondary[name] = run_matches(protocol, root / f"secondary-{name}")
    measure(a.baseline, a.dataset, root / "measurement.json")
    subprocess.run(
        [
            sys.executable,
            "scripts/diagnostics.py",
            checkpoint,
            str(root / "calibration.json"),
            "--matches",
            str(root / f"seed-{seed}" / "final/matches.json"),
        ],
        check=True,
    )
    scores = [r["summary"]["score"] for r in runs]
    aggregate = {
        "runs": runs,
        "representative": seed,
        "mean_score": float(np.mean(scores)),
        "seed_score_std": float(np.std(scores, ddof=1)),
        "secondary": secondary,
    }
    write_json(root / "aggregate.json", aggregate)
    manifest = {
        "status": "Bounded local study - three seeds, shallow search",
        "description": "100,000 teacher-labeled positions; 60 self-play games per seed; 400 final games per seed. The search budget is 48 nodes, so conclusions apply only to this constrained setup.",
        "supervised_metrics": str(Path(a.baseline).parent / "metrics.json"),
        "rl_metrics": str(root / f"seed-{seed}" / "training/metrics.json"),
        "matches": str(root / f"seed-{seed}" / "final/matches.json"),
        "summary": str(root / f"seed-{seed}" / "final/summary.json"),
        "measurement": str(root / "measurement.json"),
        "calibration": str(root / "calibration.json"),
        "aggregate": str(root / "aggregate.json"),
        "cost_cad": 0,
        "diagnostics": str(root / "diagnostics.json") if (root / "diagnostics.json").exists() else None,
        "limitations": [
            "Only 60 self-play games per seed were used, versus the original 1,000-game target. Search was limited to 48 nodes per move.",
            "The 400-game result highlighted on the site belongs to the representative checkpoint selected on development matches. All three final seed results are reported separately.",
            "The historical source corpus and shallow search restrict generalization; results do not establish improvement at the demo search budget or against strong opponents.",
            "Draw-heavy experience and exclusion of truncated games can bias outcome learning. Move-cap sensitivity is reported.",
            "Supervised fitting preceded the first Git commit and records that fact. The RL and match protocols identify the committed implementation and immutable artifact hashes.",
            "No Elo rating, AlphaZero-scale capability, or general reinforcement-learning improvement is claimed.",
        ],
    }
    write_json("configs/report-local.json", manifest)
    generate("configs/report-local.json", "output/pdf/local-study")
    subprocess.run(
        [
            sys.executable,
            "scripts/publish_artifacts.py",
            a.baseline,
            checkpoint,
            "output/pdf/local-study",
            "--selection",
            str(root / f"seed-{seed}" / "selection.json"),
        ],
        check=True,
    )
    print("Finalized local study, representative seed:", seed)
