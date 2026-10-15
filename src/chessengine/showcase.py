"""Build an identity-checked, allowlisted four-model preview or gated release."""

import argparse
import json
import numpy as np
import shutil
import re
from pathlib import Path
from .common import read_json, write_json, digest
from .model import NumpyEvaluator
from .ranking import NumpyRanker, GuidedEvaluator


def build(config, output, publish=False):
    root = Path(config["output"])
    fits = read_json(root / "training-results.json")
    fit = next(r for r in fits if r["seed"] == 42)
    if digest(fit["checkpoint"]) != fit["sha256"]:
        raise ValueError("Ranker differs from selected training artifact")
    results_path = root / "results.json"
    results = read_json(results_path) if results_path.exists() else None
    if publish and (
        not results or not results["release_eligible"] or results["release_check"]["score"] < 0.5
    ):
        raise ValueError("Completed passing development release gate required")
    if results:
        if (
            digest(config["baseline"]) != results["selection"]["value_sha256"]
            or fit["sha256"] != results["selection"]["ranker_sha256"]
        ):
            raise ValueError("Release components differ from evaluated checkpoints")
    output = Path(output)
    if publish and output.resolve() != Path("runs/showcase-release").resolve():
        raise ValueError("Use the designated release staging directory")
    models = output / "models"
    public = output / "public"
    models.mkdir(parents=True, exist_ok=True)
    (public / "models").mkdir(parents=True, exist_ok=True)
    (public / "research").mkdir(parents=True, exist_ok=True)
    for name, source in [
        ("supervised.npz", config["baseline"]),
        ("rl.npz", config["rl_checkpoint"]),
        ("ranker.npz", fit["checkpoint"]),
    ]:
        with np.load(source, allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files if k != "metadata"}
            meta = json.loads(str(saved["metadata"]))
        allowed = {
            "format",
            "format_version",
            "feature_version",
            "move_schema",
            "board_hidden",
            "move_hidden",
            "hidden",
            "kind",
            "evidence",
        }
        clean = {k: v for k, v in meta.items() if k in allowed}
        clean["source_checkpoint_sha256"] = digest(source)
        np.savez_compressed(models / name, metadata=np.array(json.dumps(clean)), **arrays)
        shutil.copyfile(models / name, public / "models" / name)
    value = NumpyEvaluator(models / "supervised.npz")
    rl = NumpyEvaluator(models / "rl.npz")
    ranker = NumpyRanker(models / "ranker.npz")
    guided = GuidedEvaluator(value, ranker)
    manifest = dict(
        showcase_version=2,
        default="supervised",
        models=[
            dict(
                id="supervised",
                name="Supervised model",
                file="supervised.npz",
                sha256=value.checkpoint_id,
                documentation_path="/lab/supervised",
                experimental=False,
            ),
            dict(
                id="rl",
                name="Self-play model",
                file="rl.npz",
                sha256=rl.checkpoint_id,
                documentation_path="/lab/rl",
                experimental=True,
            ),
            dict(
                id="move-guided",
                name="Move-guided engine",
                kind="move-guided",
                value_model_id="supervised",
                ranker_file="ranker.npz",
                ranker_sha256=ranker.checkpoint_id,
                sha256=guided.checkpoint_id,
                documentation_path="/lab/move-guided",
                experimental=True,
            ),
        ],
        release_gate=results["release_check"] if results else None,
        preview=not publish,
    )
    write_json(models / "manifest.json", manifest)
    write_json(public / "models/manifest.json", manifest)
    catalog = dict(
        models={
            "handcrafted": dict(
                name="Classical baseline",
                checkpoint_id="handcrafted-v1",
                experimental=False,
                facts=[
                    "No training data or neural network.",
                    "Material and positional bonuses; shared alpha–beta search.",
                ],
            ),
            "supervised": dict(
                name="Supervised evaluator",
                checkpoint_id=value.checkpoint_id,
                experimental=False,
                facts=[
                    "Feature version 3; 853 player-relative inputs.",
                    "128/64 MLP, seed 42; scalar side-to-move target.",
                    "Live analysis uses bounded node/time search; study matches use their frozen settings.",
                ],
            ),
            "rl": dict(
                name="Self-play experiment",
                checkpoint_id=rl.checkpoint_id,
                experimental=True,
                facts=[
                    "Live self-play seed 42, batch 10.",
                    "Completed three-seed study regressed: aggregate score 44.125%.",
                    "This is not the later anchored offline replay model.",
                ],
            ),
            "move-guided": dict(
                name="Move-guided engine",
                checkpoint_id=guided.checkpoint_id,
                experimental=True,
                facts=[
                    "Separate legal-move scorer; representative seed 42.",
                    "The supervised value component is identical to the supervised page.",
                    "Cross-entropy imitates a teacher move, not win probability.",
                ],
                training=dict(
                    label="Validation cross-entropy by epoch",
                    points=[r["validation"]["cross_entropy"] for r in fit["history"]],
                ),
            ),
        }
    )
    sl_metrics = Path(config["baseline"]).parent / "metrics.json"
    rl_metrics = Path(config["rl_checkpoint"]).parent / "metrics.json"
    if sl_metrics.exists():
        catalog["models"]["supervised"]["training"] = dict(
            label="Validation MSE by epoch",
            points=[r["validation_loss"] for r in read_json(sl_metrics)["history"]],
        )
    if rl_metrics.exists():
        catalog["models"]["rl"]["training"] = dict(
            label="Outcome-fitting loss by batch (changing self-play data)",
            points=[r["loss"] for r in read_json(rl_metrics)["history"]],
        )
    for key, directory, names in [
        ("supervised", "fixed-data-improvements", ["index.html", "comparison.png", "curves.png"]),
        ("rl", "rl-improvement", ["index.html", "learning.png", "results.json"]),
    ]:
        destination = public / "research" / directory
        destination.mkdir(exist_ok=True)
        for name in names:
            source = Path("docs/experiments") / directory / name
            if source.exists():
                if name.endswith(".html"):

                    def public_link(match):
                        target, label = match.group(1), match.group(2)
                        if target in names or target.startswith(("https://", "#")):
                            return match.group(0)
                        return f"<span>{label} (private research archive)</span>"

                    text = re.sub(
                        r'<a href="([^"]+)"[^>]*>(.*?)</a>', public_link, source.read_text(), flags=re.DOTALL
                    )
                    (destination / name).write_text(text)
                else:
                    shutil.copyfile(source, destination / name)
        catalog["models"][key]["report"] = f"/research/{directory}/index.html"
    if results:
        from .ranking_report import build as report

        report(config, public / "research/move-ranking")
        catalog["models"]["move-guided"]["report"] = "/research/move-ranking/index.html"
    write_json(public / "research/model-catalog.json", catalog)
    # Explicit files only: never recursively copy runs, data, configs or source.
    if publish:
        for name in ["supervised.npz", "rl.npz", "ranker.npz", "manifest.json"]:
            shutil.copyfile(models / name, Path("artifacts/models") / name)
            shutil.copyfile(models / name, Path("web/public/models") / name)
        shutil.copyfile(public / "research/model-catalog.json", "web/public/research/model-catalog.json")
        target = Path("web/public/research/move-ranking")
        target.mkdir(exist_ok=True)
        for name in ["index.html", "evidence.json", "depth-summary.json"]:
            shutil.copyfile(public / "research/move-ranking" / name, target / name)
        for directory, names in [
            ("fixed-data-improvements", ["index.html", "comparison.png", "curves.png"]),
            ("rl-improvement", ["index.html", "learning.png", "results.json"]),
        ]:
            destination = Path("web/public/research") / directory
            destination.mkdir(exist_ok=True)
            for name in names:
                source = public / "research" / directory / name
                if source.exists():
                    shutil.copyfile(source, destination / name)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    build(read_json(args.config), args.output, args.publish)
