"""Reproduce/resume the fixed seed-42 supervised development release screen."""
from pathlib import Path
import shutil
import time

from chessengine.common import digest, read_json, write_json, provenance
from chessengine.evolution_games import tasks_for, run_tasks, audit_ledger
from chessengine.evaluate import paired_summary
from chessengine.supervised_release import BASELINE_SHA256, CANDIDATE_SHA256, release_eligible

ROOT = Path(__file__).resolve().parents[1]
OUT = (ROOT / "runs/supervised-180k-release").resolve()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {
        "candidate": ROOT / "runs/supervised-200k/baseline-all-seed-42/best.npz",
        "baseline": ROOT / "artifacts/models/supervised.npz",
        "openings": ROOT / "runs/evolutionary/openings.json",
    }
    if digest(inputs["candidate"]) != CANDIDATE_SHA256:
        raise ValueError("The preselected candidate changed")
    settings = dict(nodes=1000000000, time_ms=100, qdepth=2, max_plies=512)
    identity = dict(
        candidate_sha256=CANDIDATE_SHA256, baseline_sha256=BASELINE_SHA256,
        openings_sha256=digest(inputs["openings"]),
        source_sha256={n: digest(ROOT / "src/chessengine" / (n + ".py"))
                       for n in ["engine", "model", "features", "evaluate", "evolution_games"]},
        seed=42, split="development", pairs=40, workers=2, settings=settings,
        gate=dict(minimum_score=0.5, maximum_capped_fraction=0.05, minimum_uncapped_score=0.5),
        note="Same 40-pair, 100 ms, qdepth-two development screen as the prior supervised release check. "
             "Seed 42 fixed before matches; no final-test selection or strength improvement claim. "
             "Candidate is the unmodified feature-v1 180302-position checkpoint, not the later v3 candidate.",
    )
    protocol = OUT / "protocol.json"
    if protocol.exists():
        if read_json(protocol)["identity"] != identity:
            raise ValueError("Frozen release protocol changed")
    else:
        if digest(inputs["baseline"]) != BASELINE_SHA256:
            raise ValueError("Baseline changed before the release screen")
        for name in ["candidate", "baseline"]:
            shutil.copyfile(inputs[name], OUT / (name + ".npz"))
        write_json(protocol, dict(identity=identity, provenance=provenance()))
    for name in ["candidate", "baseline"]:
        if digest(OUT / (name + ".npz")) != identity[name + "_sha256"]:
            raise ValueError("Saved release component changed")
    starts = read_json(inputs["openings"])["development"]
    if len(starts) != 40:
        raise ValueError("Expected the complete frozen development pool")
    started = time.monotonic()
    ledger = run_tasks(tasks_for(OUT / "candidate.npz", OUT / "baseline.npz", starts, settings),
                       OUT / "matches.json", workers=2)
    audit_ledger(ledger)
    summary = paired_summary(ledger["games"], seed=42)
    write_json(OUT / "summary.json", dict(
        **summary, passed=release_eligible(summary), protocol_sha256=digest(protocol),
        candidate_sha256=CANDIDATE_SHA256, baseline_sha256=BASELINE_SHA256,
        elapsed_seconds=time.monotonic()-started, paid_cost_cad=0,
        limitation="Development release screen, not fresh final evidence or a proven strength improvement.",
    ))
    print(read_json(OUT / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
