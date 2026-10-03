from __future__ import annotations

import html
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from .common import read_json, write_json, digest


def generate(manifest_path, output):
    spec = read_json(manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    supervised = read_json(spec["supervised_metrics"]) if spec.get("supervised_metrics") else None
    rl = read_json(spec["rl_metrics"]) if spec.get("rl_metrics") else None
    matches = read_json(spec["matches"]) if spec.get("matches") else None
    summary = read_json(spec["summary"]) if spec.get("summary") else None
    measure = read_json(spec["measurement"]) if spec.get("measurement") else None
    calibration = read_json(spec["calibration"]) if spec.get("calibration") else None
    limitations = spec.get("limitations", [])
    data = {
        "status": spec["status"],
        "supervised": supervised,
        "rl": rl,
        "summary": summary,
        "games": matches["games"] if matches else [],
        "limitations": limitations,
        "measurement": measure,
        "calibration": calibration,
        "cost_cad": spec.get("cost_cad", 0),
    }
    write_json(output / "results.json", data)
    charts = []
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    if supervised:
        history = supervised["history"]
        fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
        ax.plot(
            [r["epoch"] for r in history],
            [r["train_loss"] for r in history],
            label="Training",
            color="#49683b",
        )
        ax.plot(
            [r["epoch"] for r in history],
            [r["validation_loss"] for r in history],
            label="Validation",
            color="#7d66a6",
        )
        ax.set(xlabel="Epoch", ylabel="Mean squared error", title="Supervised learning: teacher target fit")
        ax.legend(frameon=False)
        fig.savefig(output / "supervised.png", dpi=180)
        plt.close(fig)
        charts.append(
            ("supervised.png", "Teacher agreement is a regression metric, not proof of playing strength.")
        )
    if rl:
        history = rl["history"]
        fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
        usable = [r for r in history if r["loss"] is not None]
        if usable:
            ax.plot(
                [r["games_total"] for r in usable], [r["loss"] for r in usable], marker="o", color="#49683b"
            )
        else:
            ax.text(0.5, 0.5, "No completed-game updates", ha="center", transform=ax.transAxes)
        ax.set(
            xlabel="Self-play games generated",
            ylabel="Outcome regression MSE",
            title="RL training: outcome targets change with the policy",
        )
        fig.savefig(output / "rl.png", dpi=180)
        plt.close(fig)
        charts.append(
            (
                "rl.png",
                "Losses across batches use different on-policy data and are not directly comparable measures of strength.",
            )
        )
    if calibration and calibration["bins"]:
        fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
        ax.plot([0, 1], [0, 1], "--", color="#a1a1a1")
        ax.scatter(
            [b["prediction"] for b in calibration["bins"]],
            [b["observed"] for b in calibration["bins"]],
            s=[min(100, 15 + b["count"]) for b in calibration["bins"]],
            color="#49683b",
        )
        ax.set(
            xlabel="Predicted expected score: (value + 1) / 2",
            ylabel="Observed game score",
            title="Outcome calibration on recorded evaluation games",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        fig.savefig(output / "calibration.png", dpi=180)
        plt.close(fig)
        charts.append(
            (
                "calibration.png",
                "Positions within games are correlated; this diagnostic does not provide independent position-level confidence intervals.",
            )
        )
    sections = [
        (
            "Research question",
            "Does Monte Carlo self-play improve a supervised chess evaluator under a fixed search budget? This report separates training fit, playing performance, and practical runtime.",
        ),
        ("Evidence status", spec["status"] + ". " + spec.get("description", "")),
        (
            "Architecture",
            "A shared Python negamax search uses alpha-beta pruning, bounded quiescence, and history-aware transposition keys. python-chess supplies rules. A compact MLP evaluates positions; the web engine uses numerically checked NumPy inference. Stockfish is the teacher, not a hidden playing fallback.",
        ),
        (
            "Supervised method",
            "Source games are partitioned before position sampling. Duplicate positions across splits and reserved opening source games are excluded. Stockfish WDL estimates produce side-to-move targets y=(wins-losses)/total. The evaluator minimizes mean squared error with AdamW and validation-based checkpoint selection.",
        ),
        (
            "Reinforcement-learning method",
            "The frozen actor generates a batch of self-play games. Completed outcomes provide z=+1 for a win, 0 for a draw, and -1 for a loss, from each position's side-to-move perspective. We minimize mean((v_theta(s)-z)^2), then regenerate games with the updated actor. Each completed game receives equal sampling weight. Truncated games are excluded; no teacher labels enter RL.",
        ),
        (
            "Evaluation method",
            "A frozen protocol identifies both checkpoint hashes, the reserved opening set, and search budgets. Each opening is played twice with colors swapped. Score=(wins+0.5*draws)/games. The 95% interval resamples whole opening pairs, not individual games. Move-cap adjudications are reported separately. Checkpoints may be selected on development matches only.",
        ),
    ]
    if supervised:
        sections.append(
            (
                "Supervised results",
                f"Best validation MSE: {supervised['best_validation_mse']:.6f}. Epochs: {len(supervised['history'])}. Device: {supervised['device']}. Data hash: {supervised['data_sha256']}.",
            )
        )
    if measure:
        sections.append(
            (
                "Held-out teacher agreement",
                f"{measure['positions']} positions; MSE {measure['teacher_mse']:.6f}; {measure['ms_per_position_including_features']:.3f} ms per evaluation including features. Phase MSE: "
                + ", ".join(f"{k}: {v:.4f}" for k, v in measure["phase_mse"].items())
                + ".",
            )
        )
    if rl:
        h = rl["history"]
        completed = sum(r["completed"] for r in h)
        truncated = sum(r["truncated"] for r in h)
        sections.append(
            (
                "RL run",
                f"Generated {h[-1]['games_total'] if h else 0} games: {completed} completed, {truncated} truncated. Generation took {sum(r['generation_seconds'] for r in h):.1f} seconds; updates took {sum(r['update_seconds'] for r in h):.2f} seconds. Completed-game training does not by itself establish improvement.",
            )
        )
    if summary:
        sections.append(
            (
                "Match results",
                f"{summary['games']} games ({summary['pairs']} opening pairs): {summary['wins']} wins, {summary['draws']} draws, {summary['losses']} losses. Candidate score {summary['score']:.1%}; paired 95% interval [{summary['ci95'][0]:.1%}, {summary['ci95'][1]:.1%}]. Result: {summary['conclusion']}. {summary['capped_games']} capped games. Uncapped-pair score: {summary['uncapped_score']}; uncapped pairs: {summary['uncapped_pairs']}.",
            )
        )
    sections.append(("Limitations", " ".join(limitations)))
    sections.append(
        (
            "Reproducibility and cost",
            f"Paid compute expenditure: C${spec.get('cost_cad', 0):.2f}. Each run records configs, RNG seeds, dataset and checkpoint hashes, hardware, and Git state. Manifest SHA256: {digest(manifest_path)}. Main-study results must not be inferred from pilot evidence.",
        )
    )
    sections.append(
        (
            "Sources",
            "Lichess public database: https://database.lichess.org/ . Stockfish: https://github.com/official-stockfish/Stockfish . python-chess: https://python-chess.readthedocs.io/ . Monte Carlo policy evaluation follows the standard outcome-return formulation; the project is not an AlphaZero reproduction.",
        )
    )
    style = "body{font-family:system-ui,sans-serif;max-width:900px;margin:60px auto;padding:0 24px;color:#20352b;line-height:1.8}h1{font-size:42px;line-height:1.2}h2{font-size:21px;margin-top:32px}p{overflow-wrap:anywhere}img{max-width:100%}.badge{color:#546c47;letter-spacing:2px;font-size:12px}figcaption{font-size:13px;color:#647264}a{color:#456b38}"
    body = '<div class="badge">CHESSENGINE / RESEARCH NOTE</div><h1>Learning through self-play</h1><p><a href="report.pdf">Download PDF</a> · <a href="results.json">Machine-readable results</a></p>'
    for title, content in sections:
        body += f"<h2>{html.escape(title)}</h2><p>{html.escape(content)}</p>"
        if title == "Supervised results":
            body += '<figure><img src="supervised.png" alt="Training and validation curves"/><figcaption>Measured loss against teacher labels.</figcaption></figure>'
        if title == "RL run":
            body += '<figure><img src="rl.png" alt="RL training curve"/></figure>'
    if calibration:
        body += '<h2>Calibration diagnostic</h2><img src="calibration.png" alt="Outcome calibration"/>'
    (output / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ChessEngine research report</title><style>'
        + style
        + "</style><body>"
        + body
        + "</body></html>"
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyLab",
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            spaceAfter=12,
            textColor=colors.HexColor("#26392e"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeadLab",
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=20,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor("#355131"),
            keepWithNext=True,
        )
    )
    story = [
        Paragraph("CHESSENGINE / RESEARCH NOTE", styles["Heading3"]),
        Paragraph("Learning through self-play", styles["Title"]),
        Spacer(1, 12),
    ]
    for title, content in sections:
        story += [
            Paragraph(html.escape(title), styles["HeadLab"]),
            Paragraph(html.escape(content), styles["BodyLab"]),
        ]
        chart = "supervised.png" if title == "Supervised results" else "rl.png" if title == "RL run" else None
        if chart:
            story += [Image(str(output / chart), width=460, height=211), Spacer(1, 10)]
    if calibration:
        story += [
            Paragraph("Calibration diagnostic", styles["HeadLab"]),
            Image(str(output / "calibration.png"), width=460, height=211),
        ]

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#71836b"))
        canvas.drawString(42, 25, "ChessEngine | " + spec["status"])
        canvas.drawRightString(553, 25, str(doc.page))

    SimpleDocTemplate(
        str(output / "report.pdf"),
        pagesize=(595.28, 841.89),
        rightMargin=42,
        leftMargin=42,
        topMargin=40,
        bottomMargin=45,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    return {"html": str(output / "report.html"), "pdf": str(output / "report.pdf")}
