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
        if target.startswith(("https://", "#", "/research")) or target in allowed:
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
    original = read_json('artifacts/research/original-release.json')
    original_models = {m['id']: m['sha256'] for m in original['models']}
    rows = [
        dict(id='self-play-archive', title='Self-play: results and lessons',
             level='Archived research', result='inconclusive', availability='research only',
             question='What did the unsuccessful self-play experiments teach us?',
             takeaway='Self-play did not establish an improvement; the larger follow-up regressed. Reports, diagnoses and recorded matches remain available; live play is retired.',
             report='/research/self-play-archive/index.html', checkpoint_ids=original_models),
        dict(
            id="original-study",
            title="Original local study",
            level="Completed local study",
            result="inconclusive",
            availability="historical study",
            question="From supervision to self-play",
            takeaway="Historical three-seed evidence for the original released collection; later studies use different checkpoints.",
            report="/research/report.html",
            checkpoint_ids=original_models,
        )
    ]
    assets = {}
    archive = out/'self-play-archive'
    archive.mkdir(exist_ok=True)
    (archive/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Self-play: results and lessons</title><style>body{max-width:800px;margin:3rem auto;padding:1rem;font:18px/1.7 system-ui}li{margin:1rem 0}</style><nav><a href="/research">Research journal</a></nav><h1>Self-play: results and lessons</h1><p>Self-play did not establish an improvement; the larger follow-up regressed. These models are retired from live play. This archive preserves what we tried, measured and learned.</p><ul><li><a href="/research?study=original-study">Original local study</a>: inconclusive improvement across three seeds. Charts and recorded-match replay remain in the original-study archive; recorded replay does not invoke a live engine.</li><li><a href="/research/rl-improvement/index.html">Larger live self-play follow-up</a>: 44.125% mean final score against its initializer; a regression, using different checkpoints.</li><li><a href="/research/rl-update-diagnosis/index.html">Fixed-trajectory diagnosis</a>: prediction anchoring passed a development screen. This was offline replay, not successful live reinforcement learning.</li><li><a href="/research/policy-value/index.html">Policy/value feasibility</a>: a completed 60-attempt pilot; the full comparison was deferred and provides no strength conclusion.</li></ul><h2>What changed next</h2><p>Shallow-play returns and prediction drift motivated a separately preregistered evolutionary experiment. We preserve the negative evidence rather than calling it a successful model.</p><a href="/research?study=evolutionary">Evolutionary research</a></html>''')
    assets['self-play-archive/index.html'] = digest(archive/'index.html')
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

    # Completed historical reports are immutable inputs to publication.
    if not Path('docs/experiments/policy-value/evidence.json').exists():
        policy_value_report()
    pv = Path("docs/experiments/policy-value")
    state = read_json(pv / "evidence.json") if (pv / "evidence.json").exists() else None
    status = state["status"]["status"] if state else "planned"
    if state:
        target = out / "policy-value"
        target.mkdir(exist_ok=True)
        names = ["index.html", "evidence.json", "protocol.json"]
        if "release-scope" in state:
            names.append("release-scope.json")
        if (pv / "learning.svg").exists():
            names.append("learning.svg")
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
            level="Completed feasibility pilot"
            if status == "feasibility_complete"
            else "Completed final study"
            if status == "complete"
            else "Preregistered feasibility pilot",
            result=state["results"]["aggregate"]["conclusion"]
            if state and "results" in state
            else "pilot"
            if state
            else "planned",
            availability="research only",
            question="Does prediction anchoring improve policy/value self-play?",
            takeaway="All three search budgets passed the 60-game feasibility pilot. The full anchoring comparison was deferred because of its compute cost; no strength improvement is claimed."
            if status == "feasibility_complete"
            else f"Execution status: {status.replace('_', ' ')}. Two treatments and three paired seeds; no automatic model promotion.",
            report="/research/policy-value/index.html" if state else None,
        )
    )
    from .evolution_report import report as evolutionary_report
    legacy_evolution = Path('docs/experiments/evolutionary')
    if (legacy_evolution / 'evidence.json').exists():
        evolution = read_json(legacy_evolution / 'evidence.json')
        (out / 'evolutionary').mkdir(exist_ok=True)
        for name in ['index.html', 'evidence.json', 'protocol.json', 'learning.png']:
            if (legacy_evolution / name).exists():
                shutil.copyfile(legacy_evolution / name, out / 'evolutionary' / name)
    else:
        evolution = evolutionary_report(read_json('configs/evolutionary.json'), out/'evolutionary')
    for path in (out/'evolutionary').iterdir():
        if path.name in {'index.html', 'evidence.json', 'protocol.json', 'learning.png'}:
            assets[f'evolutionary/{path.name}'] = digest(path)
    rows.insert(1, dict(id='evolutionary', title='Evolutionary chess', level='Teacher-free evolutionary study',
                       result='pilot' if evolution['status'] == 'planned' else 'diagnostic' if not evolution['qualified'] else 'improvement',
                       availability='available' if 'evolutionary' in models else 'did not qualify' if evolution['status'] in ('pilot_failed', 'complete') else 'in research',
                       question='Can tournament-driven evolution learn a useful evaluator from scratch?',
                       takeaway='All 324 evaluation parameters start randomly; no teacher labels or fixed material score. See the measured gates and evidence.',
                       report='/research/evolutionary/index.html', checkpoint_ids={'evolutionary': models['evolutionary']} if 'evolutionary' in models else {}))
    gpu_source = Path('docs/experiments/gpu-value')
    if (gpu_source / 'evidence.json').exists():
        target = out / 'gpu-value'
        target.mkdir(exist_ok=True)
        from .pv_reporting import public
        for name in ['index.html', 'evidence.json']:
            if name.endswith('.html'):
                (target / name).write_text(sanitize((gpu_source / name).read_text(), {'index.html', 'evidence.json'}))
            else:
                write_json(target / name, public(read_json(gpu_source / name)))
            assets[f'gpu-value/{name}'] = digest(target / name)
        gpu_evidence = read_json(gpu_source / 'evidence.json')
        conclusion = gpu_evidence.get('evidence',{}).get('conclusion','planned')
        if conclusion not in {'planned','pilot','negative','inconclusive','regression','improvement'}:
            conclusion = 'inconclusive'
        gpu_ids = {'supervised-1m','evolutionary-gpu'}
        rows.insert(1, dict(id='gpu-value', title='GPU value learning studies',
            level='GPU studies: '+gpu_evidence['status'], result=conclusion,
            availability='available' if gpu_ids & set(models) else 'research only',
            question='Can broader supervision and GPU tournaments improve the value evaluator?',
            takeaway=(gpu_evidence.get('evidence',{}).get('summary',{}).get('same_search_ablation',{}).get('takeaway') or
                ('Augmented Supervised completed GPU training on 800,000 positions from a one-million-position dataset. It is playable experimentally; playing strength is not established. Evolutionary training remains planned.'
                if gpu_evidence.get('evidence',{}).get('training_executed') else
                'One million total positions (800,000 training); planned evolution uses 368,640 main training games. No NVIDIA training or playing-strength result is claimed.')),
            report='/research/gpu-value/index.html',
            checkpoint_ids={key:models[key] for key in gpu_ids if key in models}))
    # These aggregate reports are explicitly allowlisted; checkpoint publication is separate.
    for slug, study_id, title, names in [
        ('elite-policy', 'elite-pytorch-baseline-20261124', 'Learning moves from elite human games', ['index.html', 'evidence.json', 'curves.svg']),
        ('stockfish-move-policy', 'stockfish-move-policy', 'Stockfish move imitation', ['index.html', 'evidence.json']),
        ('supervised-variants', 'supervised-variants', 'Playable supervised variants', ['index.html', 'evidence.json']),
    ]:
        source = Path('docs/experiments') / slug
        if not (source / 'index.html').exists():
            continue
        target = out / slug
        target.mkdir(exist_ok=True)
        for name in names:
            if not (source / name).exists():
                continue
            if name.endswith('.html'):
                (target / name).write_text(sanitize((source / name).read_text(), set(names)))
            else:
                shutil.copyfile(source / name, target / name)
            assets[f'{slug}/{name}'] = digest(target / name)
        identities = (models if slug == 'supervised-variants' else
                      {key: models[key] for key in ['elite-policy' if slug == 'elite-policy' else 'stockfish-policy'] if key in models})
        rows.append(dict(id=study_id, title=title, level='Experimental playable models',
            result='diagnostic', availability='experimental play' if identities else 'research only',
            question='How do different supervised training choices behave?',
            takeaway='Previously selected checkpoints are playable experimentally. Agreement metrics do not establish playing strength; historical failures remain preserved.',
            report=f'/research/{slug}/index.html', checkpoint_ids=identities))
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
