"""Publish an explicit research allowlist independently of neural-model release gates."""

import argparse
import html
import re
import shutil
from pathlib import Path
from .common import read_json, write_json, digest

# Only these named assets can leave the private research archive.
STUDIES = [
    (
        "supervised-diagnostics",
        "Supervised generalization",
        "Development diagnostic",
        "diagnostic",
        "Training coverage and regularization",
        "The fixed regularization search did not establish a better evaluator.",
        ["index.html", "data-size.png", "training-curves.png", "predictions.png"],
    ),
    (
        "supervised-200k",
        "More training positions",
        "Development study",
        "diagnostic",
        "Does broader training coverage help?",
        "An expanded training corpus; original held-out splits were preserved.",
        ["index.html", "data-size.png", "training-curves.png"],
    ),
    (
        "fixed-data-improvements",
        "Representation and optimization",
        "Development study",
        "diagnostic",
        "Improve fitting without adding examples",
        "Player-relative representation and optimization were evaluated on fixed data; teacher error is not playing strength.",
        ["index.html", "comparison.png", "curves.png"],
    ),
    (
        "supervised-averaging",
        "Model averaging",
        "Development study",
        "inconclusive",
        "Can averaging improve the initializer?",
        "The 1.65% MSE reduction missed the predefined 2% adoption threshold.",
        ["index.html", "comparison.png", "curves.png"],
    ),
    (
        "rl-improvement",
        "Live self-play follow-up",
        "Completed live self-play study",
        "regression",
        "Does outcome learning improve the v3 evaluator?",
        "Mean final score was 44.125% against its initializer. This is a different checkpoint from the earlier released self-play model.",
        ["index.html", "learning.png"],
    ),
    (
        "rl-update-diagnosis",
        "Understanding value drift",
        "Off-policy development diagnostic",
        "diagnostic",
        "Why did outcome fitting hurt?",
        "Anchoring passed the development screen on fixed historical trajectories; this does not establish live self-play success.",
        ["index.html"],
    ),
    (
        "move-ranking",
        "Learned move ordering",
        "Completed final evaluation",
        "inconclusive",
        "Does learned root ordering improve search?",
        "48.75% final score (95% paired interval 45.75–51.625%). The supervised release gate failed; this remains a local preview.",
        ["index.html"],
    ),
]


def sanitize(document, allowed):
    def link(match):
        tag, target = match.group(1), html.unescape(match.group(2))
        if target.startswith(("https://", "#")) or target in allowed:
            return match.group(0)
        if tag == "href":
            return 'title="Private research archive"'  # Keep prose, remove dead/private links.
        raise ValueError(f"Report references a non-allowlisted image: {target}")

    document = re.sub(r'(href|src)="([^"]+)"([^>]*)', link, document)
    document = re.sub(
        r'(?:/Users/[^\s<>"\']+|(?:runs|data)/[\w./-]+)', "[private research artifact]", document
    )
    return re.sub(r"(<h1\b)", '<nav><a href="/research">← All research</a></nav>\\1', document, count=1)


def publish(destination="web/public/research"):
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    released = read_json("artifacts/models/manifest.json")
    models = {m["id"]: m["sha256"] for m in released["models"]}
    rows = [
        dict(
            id="original-study",
            title="Original local study",
            level="Completed local study",
            result="inconclusive",
            availability="released",
            question="From supervision to self-play",
            takeaway="Historical three-seed evidence for the original released collection; later studies use different checkpoints.",
            report="/research/report.html",
            checkpoint_ids=models,
        )
    ]
    assets = {}
    for slug, title, level, result, question, takeaway, names in STUDIES:
        source = Path("docs/experiments") / slug
        target = out / slug
        target.mkdir(exist_ok=True)
        for name in names:
            path = source / name
            if not path.is_file():
                raise ValueError(f"Missing allowlisted report asset: {path}")
            if name.endswith(".html"):
                (target / name).write_text(sanitize(path.read_text(), set(names)))
            else:
                shutil.copyfile(path, target / name)
            assets[f"{slug}/{name}"] = digest(target / name)
        checkpoint_ids = {}
        identity_source = None
        if slug == "move-ranking":
            identity_source = source / "evidence.json"
            selected = read_json(identity_source)["results"]["selection"]
            checkpoint_ids = {k: v for k, v in selected.items() if k.endswith("sha256")}
        elif slug == "rl-improvement":
            identity_source = source / "results.json"
            checkpoint_ids = {
                str(r["seed"]): r["selection"]["sha256"] for r in read_json(identity_source)["seeds"]
            }
        rows.append(
            dict(
                checkpoint_ids=checkpoint_ids,
                source_evidence_sha256=digest(identity_source)
                if identity_source
                else digest(source / "index.html"),
                id=slug,
                title=title,
                level=level,
                result=result,
                question=question,
                takeaway=takeaway,
                availability="local preview" if slug == "move-ranking" else "research only",
                report=f"/research/{slug}/index.html",
            )
        )
    from .pv_reporting import build as policy_value_report

    policy_value_report()
    pv = Path("docs/experiments/policy-value")
    state = read_json(pv / "evidence.json") if (pv / "evidence.json").exists() else None
    status = state["status"]["status"] if state else "planned"
    if state:
        target = out / "policy-value"
        target.mkdir(exist_ok=True)
        names = ["index.html", "evidence.json", "protocol.json"]
        for name in names:
            source = pv / name
            if not source.exists():
                raise ValueError("Policy/value report is missing its frozen protocol or evidence")
            if name.endswith(".html"):
                (target / name).write_text(sanitize(source.read_text(), set(names)))
            else:
                shutil.copyfile(source, target / name)
            assets[f"policy-value/{name}"] = digest(target / name)
    rows.append(
        dict(
            id="policy-value",
            title="Policy/value self-play",
            level="Completed final study" if status == "complete" else "Preregistered feasibility pilot",
            result=state["results"]["aggregate"]["conclusion"]
            if state and "results" in state
            else "pilot"
            if state
            else "planned",
            availability="research only",
            question="Does prediction anchoring improve policy/value self-play?",
            takeaway=f"Execution status: {status.replace('_', ' ')}. Two treatments and three paired seeds; no automatic model promotion.",
            report="/research/policy-value/index.html" if state else None,
        )
    )
    write_json(
        out / "studies.json",
        dict(
            version=1, default_model=released["default"], released_models=models, studies=rows, assets=assets
        ),
    )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="web/public/research")
    print(f"Published {len(publish(p.parse_args().output))} research entries; no model files changed")
