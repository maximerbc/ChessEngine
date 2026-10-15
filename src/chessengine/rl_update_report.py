"""Compact development-only evidence for the fixed-trajectory update ablation."""

import argparse
from pathlib import Path
import shutil
import numpy as np
from .common import read_json, write_json
from .diagnostics_report import table


def generate(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    complete = (source / "results.json").exists()
    results = read_json(source / ("results.json" if complete else "fits.json"))
    shutil.copyfile(source / "protocol.json", output / "protocol.json")
    write_json(output / "results.json", results)
    if (source / "target-audit.json").exists():
        shutil.copyfile(source / "target-audit.json", output / "target-audit.json")
    runs = results["runs"]
    names = list(results["protocol"]["identity"]["config"]["treatments"])
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>RL update diagnosis</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:auto;padding:24px;color:#20352c;background:#f5f6f2}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccc}.table-wrap{overflow:auto}</style>'
    html += "<h1>RL regression: isolate the update</h1><p>This is an off-policy fixed-trajectory development ablation. All treatments use identical sampled training-game positions and outcomes. It is not a fresh on-policy study or independent final test. Released models remain unchanged.</p>"
    html += "<h2>Mechanism and mathematics</h2><p>The outcome-only objective fits realized returns from shallow self-play. These returns describe the generating players, not optimal chess. A useful Stockfish-trained estimate may disagree with those outcomes. The anchored objective is mean((v−z)²) + λ mean((v−v₀)²), with a detached frozen initializer v₀ and λ=1. Its unconstrained per-position optimum is (z+λv₀)/(1+λ), explicitly trading outcome fitting against preservation of initial estimates.</p>"
    html += f"<p>Baseline teacher-validation MSE {results['baseline_validation']['mse']:.6f}; prediction standard deviation {results['baseline_prediction_std']:.4f}. A lower teacher error here measures preserved agreement, not proven playing strength.</p>"
    html += table(
        ["Treatment", "Seed", "Teacher MSE", "Prediction drift MSE", "Prediction SD", "Development score"],
        [
            [
                x["treatment"],
                x["seed"],
                f"{x['validation']['mse']:.6f}",
                f"{x['validation_drift_mse']:.6f}",
                f"{x['prediction_std']:.4f}",
                f"{x['development']['score']:.2%}" if "development" in x else "pending",
            ]
            for x in runs
        ],
    )
    if (source / "target-audit.json").exists():
        audit = read_json(source / "target-audit.json")
        html += "<h2>Why return targets can pull values toward zero</h2><p>Among sampled positions where the initializer has |value| ≥ 0.75, about 22–24% lead to draws and 6% lead to an opposite-sign result in shallow self-play. These are valid outcomes, not demonstrated label errors. They illustrate that learning these players’ outcomes is a different objective from preserving strong-engine estimates.</p>"
        html += table(
            [
                "Seed",
                "Sampled positions",
                "Strong-value positions",
                "Opposite result fraction",
                "Draw fraction",
                "Unique openings",
            ],
            [
                [
                    x["seed"],
                    x["sampled_positions"],
                    x["strong_initial_positions"],
                    x["opposite_outcome_fraction_when_strong"],
                    x["draw_fraction_when_strong"],
                    x["unique_openings"],
                ]
                for x in audit["seeds"]
            ],
        )
    summaries = []
    for name in names:
        rows = [x for x in runs if x["treatment"] == name]
        summaries.append(
            dict(
                treatment=name,
                teacher_mse=float(np.mean([x["validation"]["mse"] for x in rows])),
                mean_drift=float(np.mean([x["validation_drift_mse"] for x in rows])),
                development_score=float(np.mean([x["development"]["score"] for x in rows]))
                if complete
                else None,
            )
        )
    if complete:
        baseline = {x["seed"]: x for x in runs if x["treatment"] == "reference"}
        screens = []
        for name in names:
            if name == "reference":
                continue
            rows = [x for x in runs if x["treatment"] == name]
            deltas = []
            paired = []
            for x in rows:
                seed = x["seed"]
                deltas.append(x["development"]["score"] - baseline[seed]["development"]["score"])
                sets = []
                for treatment in [name, "reference"]:
                    games = read_json(source / f"{treatment}-{seed}/development/matches.json")["games"]
                    pairs = {}
                    for g in games:
                        pairs.setdefault(g["opening_id"], []).append(g)
                    if any(
                        len(v) != 2 or {g["candidate_color"] for g in v} != {"white", "black"}
                        for v in pairs.values()
                    ):
                        raise ValueError("Incomplete diagnostic color pairs")
                    sets.append({k: np.mean([g["score"] for g in v]) for k, v in pairs.items()})
                keys = sorted(sets[0])
                if set(keys) != set(sets[1]):
                    raise ValueError("Mismatched diagnostic openings")
                paired.append([sets[0][k] - sets[1][k] for k in keys])
            values = np.mean(paired, axis=0)
            rng = np.random.default_rng(2026)
            ci = np.quantile(
                [rng.choice(values, len(values), replace=True).mean() for _ in range(10000)], [0.025, 0.975]
            ).tolist()
            mean = float(np.mean(deltas))
            wins = sum(d > 0 for d in deltas)
            screens.append(
                dict(
                    treatment=name,
                    mean_score_delta=mean,
                    improving_seeds=wins,
                    paired_difference_ci95=ci,
                    passes_screen=mean >= 0.02 and wins >= 2,
                )
            )
        write_json(output / "screening.json", screens)
        html += "<h2>Development screening</h2><p>The frozen screen requires a mean improvement of at least two percentage points against the matched replay reference, with at least two improving seeds. Differences compare performance against a common opponent; these are not direct treatment-versus-reference matches. Intervals resample shared opening pairs jointly across the three fixed seeds, conditional on these trained models.</p>"
        html += table(
            [
                "Treatment",
                "Mean score difference",
                "95% paired interval",
                "Improving seeds",
                "Online follow-up screen",
            ],
            [
                [
                    x["treatment"],
                    f"{x['mean_score_delta']:+.2%}",
                    str(x["paired_difference_ci95"]),
                    x["improving_seeds"],
                    x["passes_screen"],
                ]
                for x in screens
            ],
        )
        passing = {x["treatment"] for x in screens if x["passes_screen"]}
        eligible = sorted(
            [s for s in summaries if s["treatment"] in passing],
            key=lambda s: (-s["development_score"], s["treatment"] != "smaller-step"),
        )
        selected = eligible[0]["treatment"] if eligible else None
        write_json(
            output / "follow-up.json",
            dict(
                candidate=selected,
                evidence="Development screen only; requires live self-play and fresh confirmation before release",
            ),
        )
        html += f"<p>Candidate for a separate online study: <b>{selected or 'none'}</b>. Passing this screen does not establish improvement over the supervised initializer.</p>"
    else:
        html += "<p><b>Playing-strength comparisons are running.</b> The fitting diagnostics alone cannot select a stronger engine.</p>"
    write_json(output / "summary.json", summaries)
    html += '<h2>Interpretation and limitations</h2><p>The reference fits outcomes while losing teacher agreement and compressing its prediction range. Smaller steps and anchoring reduce that drift. This supports update-induced drift as a controllable mechanism, but does not prove it is the sole cause of chess regression. Correlated states, shallow behavior-policy errors, a changing game distribution and repeated development-set use remain limitations. No new labels, architecture search, final games or deployed weights were used.</p><p><a href="../../rl-update-diagnosis.md">Frozen methods</a> · <a href="results.json">Every seed and training curve</a> · <a href="protocol.json">Provenance</a></p></html>'
    (output / "index.html").write_text(html)
    return output / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    print(generate(args.source, args.output))
