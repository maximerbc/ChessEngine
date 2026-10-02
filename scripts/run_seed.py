"""Run one frozen RL seed and its development/final matches; safe to run seeds in parallel."""

import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from chessengine.common import read_json, write_json, digest, provenance
from chessengine.rl import selfplay
from chessengine.evaluate import freeze_protocol, run_matches, select_checkpoint


def run(baseline, root, config, seed):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = read_json(config)
    rl_cfg = {**read_json(cfg["rl"]), "seed": seed}
    dev_cfg = read_json(cfg["development"])
    eval_cfg = read_json(cfg["evaluation"])
    frozen = {
        "baseline_sha256": digest(baseline),
        "openings_sha256": digest(cfg["openings"]),
        "rl": rl_cfg,
        "development": dev_cfg,
        "evaluation": eval_cfg,
        "batches": cfg["development_batches"],
    }
    snapshot = root / "seed-protocol.json"
    if snapshot.exists():
        if read_json(snapshot)["settings"] != frozen:
            raise ValueError("Seed protocol changed")
    else:
        write_json(snapshot, {"settings": frozen, "provenance": provenance()})
    with threadpool_limits(limits=1):
        directory = root / "training"
        selfplay(baseline, cfg["openings"], directory, rl_cfg, resume=(directory / "resume.pt").exists())
        dev_dirs = []
        curve = []
        for batch in cfg["development_batches"]:
            checkpoint = directory / f"batch-{batch:03}.npz"
            protocol = root / f"development-{batch}.protocol.json"
            out = root / f"development-{batch}"
            if not protocol.exists():
                freeze_protocol(str(checkpoint), baseline, cfg["openings"], dev_cfg, protocol, "development")
            summary = run_matches(protocol, out)
            dev_dirs.append(out)
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
        select_checkpoint(dev_dirs, root / "selection.json")
        selected = read_json(root / "selection.json")
        protocol = root / "final.protocol.json"
        if not protocol.exists():
            freeze_protocol(selected["checkpoint"], baseline, cfg["openings"], eval_cfg, protocol, "test")
        summary = run_matches(protocol, root / "final")
        write_json(
            root / "result.json", {"seed": seed, "selection": selected, "summary": summary, "curve": curve}
        )
        print({"seed": seed, "summary": summary}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("output")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    run(a.baseline, a.output, a.config, a.seed)
