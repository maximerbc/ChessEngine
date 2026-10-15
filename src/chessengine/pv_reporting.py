"""Render policy/value evidence without changing the frozen scientific runner."""

import html
from pathlib import Path
import numpy as np
from .common import read_json, write_json, digest, json_hash


def public(value):
    if isinstance(value, dict):
        return {
            (
                "artifact-" + json_hash(str(k))[:16]
                if str(k).startswith(("/Users/", "runs/", "data/"))
                else k
            ): public(v)
            for k, v in value.items()
            if k not in ("initializer", "archive", "corpus", "reserved", "output")
        }
    if isinstance(value, list):
        return [public(v) for v in value]
    if isinstance(value, str) and value.startswith(("/Users/", "runs/", "data/")):
        return "artifact-" + json_hash(value)[:16]
    return value


def curve(values, label):
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    points = " ".join(
        f"{15 + i * 570 / (len(values) - 1):.1f},{165 - (v - low) * 145 / (high - low or 1):.1f}"
        for i, v in enumerate(values)
    )
    return f'<figure><svg viewBox="0 0 600 180" role="img" aria-label="{html.escape(label)}"><polyline fill="none" stroke="#24705b" stroke-width="3" points="{points}"/></svg><figcaption>{html.escape(label)}: first {values[0]:.5f}, last {values[-1]:.5f}; range {low:.5f}–{high:.5f}</figcaption></figure>'


def build(root="runs/policy-value", destination="docs/experiments/policy-value"):
    root, out = Path(root), Path(destination)
    if not (root / "protocol.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    protocol = read_json(root / "protocol.json")
    config = protocol["identity"]["config"]
    status = (
        read_json(root / "status.json")
        if (root / "status.json").exists()
        else dict(status="pilot_in_progress")
    )
    evidence = dict(
        status=status, paid_cost_cad=0, model_promotion=False, protocol_sha256=digest(root / "protocol.json")
    )
    for name in [
        "pilot-summary",
        "pilot-decision",
        "forecast",
        "compute",
        "results",
        "final-audit",
        "failures",
        "openings-manifest",
        "initial-seal",
        "environment",
        "selection-lock",
    ]:
        if (root / f"{name}.json").exists():
            evidence[name] = read_json(root / f"{name}.json")
    evidence["pilot_progress"] = {
        str(n): len(list((root / f"pilot-{n}").glob("*.json"))) for n in config["pilot_simulations"]
    }
    identity = protocol["identity"]
    evidence["initialization"] = dict(
        supervised_value_sha256=identity["inputs"][config["initializer"]],
        policy_seed=config["policy_seed"],
        run_seeds=config["seeds"],
    )
    evidence["training"] = {}
    for treatment in config["anchor_weights"]:
        for seed in config["seeds"]:
            path = root / f"{treatment}-{seed}/training.json"
            if path.exists():
                evidence["training"][f"{treatment}-{seed}"] = read_json(path)
    write_json(out / "evidence.json", public(evidence))
    write_json(out / "protocol.json", public(read_json(root / "protocol.json")))
    sections = [
        f"<h1>Policy/value self-play and prediction anchoring</h1><p>Status: <strong>{html.escape(status['status'])}</strong>. Local CPU; C$0 paid resources. No model promotion.</p>",
        "<p>Two treatments share a supervised-initialized value network and fresh policy head: visit-policy/outcome learning, and the same objective plus a 0.1 penalty toward frozen initial value predictions. This is an adapted AlphaZero-inspired study, with batch-frozen actors and a compact MLP.</p>",
        "<h2>Execution and provenance</h2><p>Pilot games completed by simulation budget: "
        + html.escape(str(evidence["pilot_progress"]))
        + ". Pilot evidence tests feasibility; it is not final playing-strength evidence.</p>",
    ]
    if "openings-manifest" in evidence:
        m = evidence["openings-manifest"]
        sections.append(
            f"<p>Fresh-opening source-game overlaps: {m['source_game_overlaps']}; canonical start overlaps: {m['canonical_overlaps']}. Full opening history is preserved. Existing corpus and prior reserved openings were excluded.</p>"
        )
    if "forecast" in evidence:
        sections.append(
            f"<p>Measured forecast before main launch: {evidence['forecast']['seconds'] / 3600:.1f} hours, including a 50% throughput margin. This is an estimate, not measured completion time.</p>"
        )
    if "compute" in evidence:
        c = evidence["compute"]
        sections.append(
            f"<p>Recorded active elapsed time: {c['active_seconds'] / 3600:.2f} hours. Peak process RAM: {c['peak_ram_mb']:.1f} MiB.</p>"
        )
    if "pilot-summary" in evidence:
        sections.append(
            '<h2>Frozen feasibility pilot</h2><div class="table-wrap"><table><tr><th>Simulations</th><th>Completed</th><th>Coverage</th><th>Dominant fraction</th><th>Seconds/game</th><th>Qualified</th></tr>'
        )
        for r in evidence["pilot-summary"]:
            sections.append(
                f"<tr><td>{r['simulations']}</td><td>{r['completed']}/{r['attempted']}</td><td>{r['root_coverage']:.1%}</td><td>{r['dominant_fraction']:.1%}</td><td>{r['mean_game_seconds']:.2f}</td><td>{r['qualifies']}</td></tr>"
            )
        sections.append(
            "</table></div><p>Frozen gates: at least 80% completed games, at least 50% mean coverage on positions with four or more legal moves, fewer than 90% dominant positions, at most 4 GiB peak RAM. A dominant position assigns at least 99% of visits to one move.</p>"
        )
    if evidence["training"]:
        sections.append(
            "<h2>All run-level learning records</h2><p>Online losses use changing self-play data and cannot be read as a fixed-data learning curve. Frozen checkpoint errors are measured separately. Fitting returns or limiting drift does not establish stronger play.</p>"
        )
        for name, training in evidence["training"].items():
            rows = training["history"]
            sections.append(f"<h3>{html.escape(name)}</h3>")
            sections.append(
                curve(
                    [r["frozen_checkpoint_outcome_mse"] for r in rows],
                    "Frozen-checkpoint outcome MSE on that batch",
                )
            )
            sections.append(
                curve(
                    [r["frozen_checkpoint_anchor_mse"] for r in rows],
                    "Deviation from frozen initializer on that batch",
                )
            )
            sections.append(
                '<div class="table-wrap"><table><tr><th>Batch</th><th>Completed/attempted</th><th>Online objective</th><th>Outcome MSE</th><th>Anchor MSE</th><th>Checkpoint SHA-256</th></tr>'
            )
            for r in rows:
                sections.append(
                    f"<tr><td>{r['batch']}</td><td>{r['completed']}/{r['attempted']}</td><td>{r['online_loss']:.5f}</td><td>{r['frozen_checkpoint_outcome_mse']:.5f}</td><td>{r['frozen_checkpoint_anchor_mse']:.5f}</td><td><code>{r['sha256']}</code></td></tr>"
                )
            sections.append("</table></div>")
    if "results" in evidence:
        result = evidence["results"]
        a = result["aggregate"]
        sections.append(
            f"<h2>Primary final result</h2><p>Anchored versus baseline: {a['score']:.2%}; paired 95% interval {a['ci95'][0]:.2%}–{a['ci95'][1]:.2%}. {html.escape(a['conclusion'])}. Seed SD: {a['seed_sd']:.4f}, separately from opening-pair uncertainty.</p>"
        )
        sections.append(
            "<table><tr><th>Seed</th><th>Score</th><th>Paired 95% interval</th><th>Caps</th></tr>"
        )
        for row in result["seeds"]:
            s = row["primary"]
            sections.append(
                f"<tr><td>{row['seed']}</td><td>{s['score']:.2%}</td><td>{s['ci95'][0]:.2%}–{s['ci95'][1]:.2%}</td><td>{s['capped_games']}</td></tr>"
            )
        sections.append(
            "</table><p>Final caps have provisional half-point scores, not rule-based draw claims. Uncapped-pair analysis and conservative cap bounds are in the evidence file.</p><h2>Secondary controls and practical cost</h2><table><tr><th>Seed</th><th>Treatment</th><th>Opponent</th><th>Score</th><th>95% interval</th></tr>"
        )
        for row in result["seeds"]:
            for treatment, controls in row["secondary"].items():
                for opponent, s in controls.items():
                    sections.append(
                        f"<tr><td>{row['seed']}</td><td>{treatment}</td><td>{opponent}</td><td>{s['score']:.2%}</td><td>{s['ci95'][0]:.2%}–{s['ci95'][1]:.2%}</td></tr>"
                    )
        sections.append("</table>")
        work = [
            m
            for path in root.glob("final-*/*/matches.json")
            for g in read_json(path)["games"]
            for moves in g["work"].values()
            for m in moves
        ]
        if work:
            ms = [r["ms"] for r in work]
            sections.append(
                f"<p>Across reported final searches: median latency {np.median(ms):.2f} ms; p95 {np.quantile(ms, 0.95):.2f} ms; MCTS simulations {sum(r.get('simulations', 0) for r in work):,}; neural calls {sum(r.get('network_calls', 0) for r in work):,}; alpha-beta nodes {sum(r.get('alpha_beta_nodes', 0) for r in work):,}. These work units are distinct; aggregation is for accounting, not an efficiency ranking.</p>"
            )
    else:
        sections.append(
            "<h2>Final evidence</h2><p>No completed final comparison is claimed. All six fits and selections, fresh finals and integrity audits are required before the study can be called complete.</p>"
        )
    sections.append(
        '<h2>Methods and limits</h2><p>Actor frozen per batch; no new teacher labels; exactly 32 sampled states per completed game; truncated games excluded from both losses. Three seeds share one initialization. All six development selections precede final games. Bootstrap shared color-swapped opening pairs jointly across fixed seeds; seed variation is reported separately. No Elo or calibrated win-probability claim. Lower drift and a head-to-head advantage can still mean less regression rather than improvement over initialization.</p><p><a href="protocol.json">Frozen protocol and identities</a> · <a href="evidence.json">All measured evidence</a></p>'
    )
    style = "body{font:16px/1.6 system-ui;max-width:1050px;margin:36px auto;padding:0 20px;color:#23382f}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccd;text-align:left}code{font-size:11px;overflow-wrap:anywhere}.table-wrap{overflow:auto}svg{max-width:600px;width:100%}h2{margin-top:36px}"
    (out / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Policy/value self-play</title><style>'
        + style
        + "</style>"
        + "".join(sections)
        + "</html>"
    )
    return public(evidence)
