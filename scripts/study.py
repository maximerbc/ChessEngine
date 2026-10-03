"""Resumable multi-seed study. Run after labeling; never purchases compute.

The config is snapshotted before the first run. Changing it requires a new output directory.
"""

import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from chessengine.common import read_json, write_json, digest, provenance
from chessengine.train import fit
from chessengine.rl import selfplay
from chessengine.evaluate import freeze_protocol, run_matches, select_checkpoint, measure

p = argparse.ArgumentParser()
p.add_argument("dataset")
p.add_argument("output")
p.add_argument("--config", default="configs/study.json")
p.add_argument("--stockfish")
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
cfg = read_json(a.config)
# Identity includes nested config contents, so changing a referenced file cannot silently change a study.
resolved = {k: read_json(cfg[k]) for k in ["supervised", "large", "rl", "development", "evaluation", "timed"]}
identity = {
    "config": cfg,
    "resolved": resolved,
    "data_sha256": digest(a.dataset),
    "openings_sha256": digest(cfg["openings"]),
}
if (root / "study.json").exists():
    if read_json(root / "study.json")["identity"] != identity:
        raise ValueError("Study settings or input data changed")
else:
    write_json(root / "study.json", {"identity": identity, "provenance": provenance(), "cost_cad": 0})


def matches(candidate, baseline, name, config, split):
    protocol = root / f"{name}.protocol.json"
    directory = root / name
    if not protocol.exists():
        freeze_protocol(str(candidate), str(baseline), cfg["openings"], config, protocol, split)
    return run_matches(protocol, directory)


with threadpool_limits(limits=1):
    candidates = []
    for key in ["supervised", "large"]:
        directory = root / key
        if not (directory / "complete.json").exists():
            result = fit(a.dataset, directory, resolved[key], resume=(directory / "resume.pt").exists())
            write_json(directory / "complete.json", result)
        result = read_json(directory / "metrics.json")
        candidates.append((result["best_validation_mse"], directory / "best.npz"))
    # Prefer smaller model when its validation MSE is within 2% of the larger model.
    selected = candidates[0] if candidates[0][0] <= candidates[1][0] * 1.02 else candidates[1]
    baseline = selected[1]
    write_json(
        root / "supervised-selection.json",
        {
            "checkpoint": str(baseline),
            "sha256": digest(baseline),
            "validation_mse": selected[0],
            "rule": "Smaller model if validation MSE is within 2 percent; otherwise lower MSE.",
        },
    )
    all_runs = []
    for seed in cfg["seeds"]:
        directory = root / f"rl-{seed}"
        rl_config = {**resolved["rl"], "seed": seed}
        selfplay(baseline, cfg["openings"], directory, rl_config, resume=(directory / "resume.pt").exists())
        dev_dirs = []
        curve = []
        for batch in cfg["development_batches"]:
            checkpoint = directory / f"batch-{batch:03}.npz"
            name = f"development-{seed}-{batch}"
            summary = matches(checkpoint, baseline, name, resolved["development"], "development")
            dev_dirs.append(root / name)
            metrics = read_json(directory / "metrics.json")["history"][:batch]
            curve.append(
                {
                    "seed": seed,
                    "batch": batch,
                    "games": metrics[-1]["games_total"],
                    "compute_seconds": sum(x["generation_seconds"] + x["update_seconds"] for x in metrics),
                    **summary,
                }
            )
        selection_path = directory / "selection.json"
        select_checkpoint(dev_dirs, selection_path)
        selected_rl = Path(read_json(selection_path)["checkpoint"])
        summary = matches(selected_rl, baseline, f"final-{seed}", resolved["evaluation"], "test")
        all_runs.append(
            {
                "seed": seed,
                "checkpoint": str(selected_rl),
                "sha256": digest(selected_rl),
                "summary": summary,
                "curve": curve,
            }
        )
        write_json(
            root / "aggregate.json",
            {
                "runs": all_runs,
                "note": "Report seed variability separately from within-run pair-bootstrap intervals.",
            },
        )
    # Representative checkpoint selected by development performance only, never final score.
    representative = max(
        all_runs, key=lambda r: read_json(root / f"rl-{r['seed']}" / "selection.json")["development_score"]
    )
    candidate = representative["checkpoint"]
    summary = matches(candidate, "handcrafted", "secondary-handcrafted", resolved["development"], "test")
    timed = matches(candidate, baseline, "secondary-timed", resolved["timed"], "test")
    if a.stockfish:
        matches(candidate, "stockfish:" + a.stockfish, "secondary-stockfish", resolved["development"], "test")
    measure(baseline, a.dataset, root / "supervised-test.json")
    scores = [r["summary"]["score"] for r in all_runs]
    write_json(
        root / "aggregate.json",
        {
            "runs": all_runs,
            "mean_score": float(np.mean(scores)),
            "seed_score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
            "representative": representative["seed"],
            "secondary_handcrafted": summary,
            "secondary_timed": timed,
            "note": "Representative checkpoint chosen by development matches; no pooled confidence claim across seeds.",
        },
    )
    print("Completed study:", root)
