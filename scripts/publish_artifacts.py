"""Explicit allowlist export; no raw datasets, local checkpoints or logs are copied."""

import argparse
import shutil
from pathlib import Path
from chessengine.common import read_json, write_json, digest
from chessengine.model import NumpyEvaluator

p = argparse.ArgumentParser()
p.add_argument("supervised")
p.add_argument("rl")
p.add_argument("report")
p.add_argument("--selection")
a = p.parse_args()
root = Path("artifacts/models")
root.mkdir(parents=True, exist_ok=True)
public = Path("web/public/models")
public.mkdir(parents=True, exist_ok=True)
rows = []
for key, path, name in [("supervised", a.supervised, "Supervised model"), ("rl", a.rl, "Self-play model")]:
    model = NumpyEvaluator(path)
    # Strip local paths and training source details from public exports; weights and schema remain.
    from chessengine.model import load_export, export

    network, meta = load_export(path)
    release_meta = {k: meta[k] for k in ["hidden", "kind", "evidence"] if k in meta}
    release_meta["source_checkpoint_sha256"] = digest(path)
    target = root / f"{key}.npz"
    export(network, target, release_meta)
    shutil.copyfile(target, public / target.name)
    rows.append(
        {
            "id": key,
            "file": target.name,
            "name": name,
            "sha256": digest(target),
            "source_sha256": digest(path),
        }
    )
default = "supervised"
if a.selection:
    selected = read_json(a.selection)
    if selected["sha256"] not in {digest(a.rl), digest(a.supervised)}:
        raise ValueError("Selection does not identify either supplied checkpoint")
    if selected["sha256"] == digest(a.rl) and selected.get("development_score", 0) > 0.5:
        default = "rl"
write_json(
    root / "manifest.json",
    {
        "default": default,
        "models": rows,
        "selection": read_json(a.selection)
        if a.selection
        else {"note": "Supervised default; no development selection supplied"},
    },
)
report = Path(a.report)
target = Path("web/public/research")
target.mkdir(parents=True, exist_ok=True)
for name in [
    "report.html",
    "report.pdf",
    "results.json",
    "supervised.png",
    "rl.png",
    "calibration.png",
    "strength.png",
]:
    if (report / name).exists():
        shutil.copyfile(report / name, target / name)
write_json(
    public / "manifest.json",
    {"models": rows, "notice": "Research checkpoints; no guaranteed playing strength."},
)
print("Published allowlisted models and report assets")
