"""Promote the fixed 180k candidate only after its frozen development screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .common import digest, read_json, write_json
from .evaluate import paired_summary
from .evolution_games import audit_ledger
from .model import NumpyEvaluator

CANDIDATE_SHA256 = "05d37bdb0ecd6717c719608e66c21fd30b19e6dd6ce27d72763d6d982d848ba5"
BASELINE_SHA256 = "ff3436d07d18eb4953e24e6c83b55d25bbb46b44094ed347f99320148ae04c4b"


def release_eligible(summary):
    """All pairs must finish; caps must not conceal a worse uncapped score."""
    return (
        summary["games"] == 80
        and summary["pairs"] == 40
        and summary["score"] >= 0.5
        and summary["capped_games"] / summary["games"] <= 0.05
        and summary["uncapped_score"] is not None
        and summary["uncapped_score"] >= 0.5
    )


def export_inference(source, destination):
    """Retain exact arrays and explicit schemas, without private training metadata."""
    if digest(source) != CANDIDATE_SHA256:
        raise ValueError("Only the preselected seed-42 180k checkpoint is eligible")
    value = NumpyEvaluator(source)
    if value.feature_version != 1 or value.metadata["hidden"] != [128, 64]:
        raise ValueError("Unexpected 180k candidate schema")
    with np.load(source, allow_pickle=False) as saved:
        metadata = {k: value.metadata[k] for k in (
            "hidden", "kind", "evidence", "feature_version", "format_version"
        )}
        metadata.update(source_checkpoint_sha256=CANDIDATE_SHA256,
                        training_positions=180302, training_seed=42)
        arrays = {k: saved[k].copy() for k in saved.files if k != "metadata"}
    np.savez_compressed(destination, metadata=np.array(json.dumps(metadata)), **arrays)
    exported = NumpyEvaluator(destination)
    for original, clean in zip(value.layers, exported.layers):
        for a, b in zip(original, clean):
            np.testing.assert_array_equal(a, b)
    return exported.checkpoint_id


def prepare(candidate, gate_directory, activate=False):
    candidate, gate_directory = Path(candidate), Path(gate_directory)
    protocol_path = gate_directory / "protocol.json"
    protocol = read_json(protocol_path)
    identity = protocol["identity"]
    if (identity["candidate_sha256"] != CANDIDATE_SHA256
            or identity["baseline_sha256"] != BASELINE_SHA256
            or digest(candidate) != CANDIDATE_SHA256):
        raise ValueError("Release checkpoint identities changed")
    if (identity["split"] != "development" or identity["pairs"] != 40
            or identity["seed"] != 42
            or identity["settings"] != dict(nodes=1000000000, time_ms=100, qdepth=2, max_plies=512)):
        raise ValueError("Release screen differs from its frozen protocol")
    for name, sha in identity["source_sha256"].items():
        if digest(Path(__file__).with_name(name + ".py")) != sha:
            raise ValueError("Search or inference source changed after the release screen")
    ledger = read_json(gate_directory / "matches.json")
    audit_ledger(ledger)
    for task in ledger["tasks"]:
        if (digest(task["candidate"]) != CANDIDATE_SHA256
                or digest(task["opponent"]) != BASELINE_SHA256
                or any(task[k] != v for k, v in identity["settings"].items())):
            raise ValueError("Screened tasks differ from the release protocol")
    summary = paired_summary(ledger["games"], seed=42)
    eligible = release_eligible(summary)
    if activate and not eligible:
        raise ValueError("180k candidate did not pass the development release gate")
    manifest = read_json("artifacts/models/manifest.json")
    old = next(m for m in manifest["models"] if m["id"] == "supervised")
    if old["sha256"] != BASELINE_SHA256 or digest("artifacts/models/supervised.npz") != BASELINE_SHA256:
        raise ValueError("Active baseline changed; do not overwrite another release")
    staging = gate_directory / "inference.npz"
    release_sha = export_inference(candidate, staging)
    evidence = dict(
        version=1, status="passed" if eligible else "blocked", release_eligible=eligible,
        seed=42, training_positions=180302,
        validation_positions=9789, source_checkpoint_sha256=CANDIDATE_SHA256,
        checkpoint_sha256=release_sha, previous_release_sha256=BASELINE_SHA256,
        training_dataset_sha256=NumpyEvaluator(candidate).metadata["data_sha256"],
        selected_validation_mse=0.27411275530180584,
        expansion_mean_validation_mse=0.2734971541161662,
        expansion_relative_mse_reduction=0.09233588092114986,
        protocol_sha256=digest(protocol_path), ledger_sha256=digest(gate_directory / "matches.json"),
        settings=identity["settings"], split="development", summary=summary,
        gate=identity["gate"], elapsed_seconds=ledger['active_seconds'], paid_cost_cad=0,
        limitations=[
            "Development screen at 100 ms and qdepth two; production remains 10000 nodes/1500 ms/qdepth four.",
            "Point-score eligibility is not a statistically established playing-strength improvement.",
            "The expanded-data comparison did not match optimizer update counts.",
            "Original local-study results describe the archived 80k release, not this checkpoint.",
        ],
    )
    evidence_dir = Path("docs/experiments/supervised-180k-release")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    write_json(evidence_dir / "evidence.json", evidence)
    write_json(evidence_dir / "games.json", dict(games=[
        {**{k: g[k] for k in ("opening_id", "candidate_color", "score", "capped", "result", "nodes", "seconds")},
         "pgn": g["pgn"].replace('[Event "Evolutionary tournament"]',
                                  '[Event "Supervised 180k release screen"]', 1)}
        for g in ledger["games"]
    ]))
    write_json(evidence_dir / "protocol.json", dict(
        **{k: identity[k] for k in ("candidate_sha256", "baseline_sha256", "seed", "split", "pairs", "workers", "settings", "gate", "openings_sha256", "source_sha256", "note")},
        private_protocol_sha256=digest(protocol_path),
        openings=[t["opening"] for t in ledger["tasks"] if t["color"] == "white"],
    ))
    candidate_dir = Path('artifacts/candidates')
    candidate_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(staging, candidate_dir / 'supervised-180k.npz')
    write_json(candidate_dir / 'supervised-180k.json', dict(
        id='supervised-180k-candidate', file='supervised-180k.npz', sha256=release_sha,
        source_sha256=CANDIDATE_SHA256, training_positions=180302, seed=42,
        release_eligible=eligible, evidence='docs/experiments/supervised-180k-release/evidence.json',
    ))
    if activate:
        old.update(sha256=release_sha, source_sha256=CANDIDATE_SHA256,
                   training_positions=180302, training_seed=42,
                   documentation_path="/research?study=supervised-180k-release")
        for directory in (Path("artifacts/models"), Path("web/public/models")):
            shutil.copyfile(staging, directory / "supervised.npz")
            write_json(directory / "manifest.json", manifest)
    return evidence


def promote(candidate, gate_directory):
    return prepare(candidate, gate_directory, activate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("gate_directory")
    parser.add_argument("--promote", action="store_true", help="Activate only after a passing release gate")
    args = parser.parse_args()
    print(json.dumps(prepare(args.candidate, args.gate_directory, args.promote), indent=2))
