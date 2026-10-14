"""Render compact move-ranking evidence without publishing raw training records."""

import argparse
import html
from pathlib import Path
import numpy as np
from .common import read_json, write_json, digest


def build(config, output):
    root = Path(config["output"])
    result = read_json(root / "results.json")
    fits = read_json(root / "training-results.json")
    if {r["seed"] for r in fits} != set(config["seeds"]):
        raise ValueError("Incomplete training evidence")
    for r in fits:
        if digest(r["checkpoint"]) != r["sha256"]:
            raise ValueError("Checkpoint changed after study")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    compact = dict(
        results=result,
        training=[dict(seed=r["seed"], sha256=r["sha256"], history=r["history"]) for r in fits],
        labeling=read_json(root / "label-progress.json"),
        exclusion_audit=read_json(root / "exclusion-audit.json"),
    )
    write_json(output / "evidence.json", compact)
    rows = []
    for r in result["development"]:
        c = r["classification"]
        rows.append(
            f"<tr><td>{r['seed']}</td><td>{c['cross_entropy']:.4f}</td><td>{c['top1']:.1%}</td><td>{c['top3']:.1%}</td><td>{c['mean_rank']:.2f}</td><td>{r['nodes']['score']:.1%}</td><td>{r['time']['score']:.1%}</td></tr>"
        )
    curves = []
    for fit in fits:
        values = [r["validation"]["cross_entropy"] for r in fit["history"]]
        low, high = min(values), max(values)
        points = " ".join(
            f"{20 + i * 560 / max(1, len(values) - 1):.1f},{180 - (v - low) * 150 / max(high - low, 1e-9):.1f}"
            for i, v in enumerate(values)
        )
        curves.append(
            f'<figure><svg viewBox="0 0 600 210" role="img" aria-label="Validation cross entropy by epoch, seed {fit["seed"]}"><polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3"/></svg><figcaption>Seed {fit["seed"]}: epochs 1–{len(values)}, CE range {low:.3f}–{high:.3f}. Lower is better.</figcaption></figure>'
        )
    depth = result["depth_comparison"]
    depth_summary = []
    for d in (2, 3):
        for engine in ("ordinary", "guided"):
            selected = [r for r in depth if r["depth"] == d and r["engine"] == engine]
            finished = [r for r in selected if r["completed"]]
            mean_nodes = float(np.mean([r["nodes"] for r in finished])) if finished else None
            depth_summary.append(
                dict(
                    depth=d,
                    engine=engine,
                    completed=len(finished),
                    total=len(selected),
                    mean_nodes_completed=mean_nodes,
                )
            )
    write_json(output / "depth-summary.json", depth_summary)
    f = result["final"]
    ci = f["ci95"]
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Move-ranking study</title><style>body{{font:17px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#172338}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ccd}}svg{{max-width:600px;width:100%}}pre{{white-space:pre-wrap}}</style><h1>Learning which move to search first</h1><p>A separate legal-move scorer changes root ordering only. The frozen supervised value evaluator and all legal moves remain. Ranking inference counts against the time budget.</p><h2>Development evidence</h2><table><tr><th>Seed</th><th>CE</th><th>Top 1</th><th>Top 3</th><th>Mean rank</th><th>Node score</th><th>Time score</th></tr>{"".join(rows)}</table><h2>Training curves</h2>{"".join(curves)}<h2>Fresh final comparison</h2><p>Preselected seed 42: {f["wins"]} wins, {f["draws"]} draws, {f["losses"]} losses. Score {f["score"]:.2%}; paired 95% interval {ci[0]:.2%}–{ci[1]:.2%}. Conclusion: {html.escape(f["conclusion"])}. Capped games: {f["capped_games"]}; uncapped-pair score: {f["uncapped_score"]}.</p><h2>Release gate</h2><p>New supervised initializer versus the previously released default: {result["release_check"]["score"]:.2%}. Release eligible: {result["release_eligible"]}. Final ranking results do not select another seed or override this gate.</p><h2>Fixed-depth efficiency</h2><pre>{html.escape(str(depth_summary))}</pre><p>Completed-only node means can be biased by differing completion rates. Safety limits are 100,000 nodes and 3 seconds; incomplete searches are included in the counts.</p><h2>Interpretation and limitations</h2><p>The loss is −log softmax of the teacher move among legal candidates. Lower cross-entropy means better imitation, not necessarily stronger play. The teacher supplies one preferred move even when alternatives are similar. Ranking probabilities are not win probabilities. All validation selection is development evidence; final results describe only the frozen seed and search protocol. Equal-time results depend on this CPU and its load.</p><p>The original corpus audit found two train/test position overlaps; four affected source games were excluded from this new study. Raw data and checkpoint hashes are retained locally; compact evidence is linked below. Paid compute cost: C$0.</p><p><a href="evidence.json">Full compact evidence</a> · <a href="depth-summary.json">Depth measurements</a></p></html>"""
    (output / "index.html").write_text(page)
    return compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("output")
    args = parser.parse_args()
    build(read_json(args.config), args.output)
