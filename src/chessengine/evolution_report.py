"""Publish measured evolutionary evidence independently of model promotion."""
import html
import json
from pathlib import Path
import shutil
import numpy as np
from .common import digest, read_json, write_json
from .evolution_model import EvolutionEvaluator, export


def read_optional(path):
    return read_json(path) if path.exists() else None


def public_protocol(value):
    """Keep input hashes and scientific settings, remove private paths."""
    if value is None:
        return None
    identity = value['identity']
    settings = {k: v for k, v in identity['config'].items()
                if k not in ('output', 'archive', 'corpus', 'supervised', 'reserved')}
    return dict(settings=settings, packages=identity['packages'], sources=identity['sources'],
                input_sha256=list(identity['inputs'].values()), provenance=value['provenance'])


def report(config, public_destination='web/public/research/evolutionary'):
    root = Path(config['output'])
    destination = Path('docs/experiments/evolutionary')
    destination.mkdir(parents=True, exist_ok=True)
    pilot = read_optional(root/'pilot-decision.json')
    results = read_optional(root/'results.json')
    status = ('complete' if results else 'pilot_failed' if pilot and not pilot['passed']
              else 'training' if pilot else 'benchmark_complete' if (root/'benchmark.json').exists() else 'planned')
    qualified = bool(results and results['release_eligible'])
    histories = {}
    for name in ['pilot', *[f'seed-{s}' for s in config['seeds']]]:
        history = read_optional(root/name/'history.json')
        if history:
            histories[name] = history
    ledgers = []
    if root.exists():
        for path in root.rglob('*.json'):
            record = read_json(path)
            if isinstance(record, dict) and isinstance(record.get('games'), list) and record.get('signature'):
                ledgers.append((str(path.relative_to(root)), record))
    elapsed = read_optional(root/'elapsed.json') or {}
    evidence = dict(status=status, qualified=qualified, representative_seed=config['representative_seed'],
                    pilot=pilot, results=results, histories=histories,
                    benchmark=read_optional(root/'benchmark.json'),
                    forecasts={s: read_optional(root/f'forecast-{s}.json') for s in ('pilot', 'main')},
                    measured_active_seconds=elapsed.get('active_seconds', 0.),
                    tournament_active_seconds=sum(r.get('active_seconds', 0.) for _, r in ledgers),
                    matches=sum(len(r['games']) for _, r in ledgers),
                    peak_combined_bytes=max([r.get('peak_combined_bytes', 0) for _, r in ledgers] or [0]),
                    match_ledger_sha256={p: digest(root/p) for p, _ in ledgers}, paid_cost_cad=0,
                    limitation='Pilot evidence is not strength evidence. All weights are learned; input features are human-designed.')
    write_json(destination/'evidence.json', evidence)
    write_json(destination/'protocol.json', public_protocol(read_optional(root/'protocol.json')))
    charts = ''
    if histories:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        for name, rows in histories.items():
            x = [r['generation'] for r in rows]
            axes[0].plot(x, [r['mean_fitness'] for r in rows], label=name)
            axes[1].plot(x, [r['parameter_diversity'] for r in rows], label=name)
        axes[0].set(xlabel='Generation', ylabel='Population tournament fitness', ylim=(0, 1))
        axes[1].set(xlabel='Generation', ylabel='Mean parameter standard deviation')
        axes[0].legend()
        fig.savefig(destination/'learning.png', dpi=150)
        plt.close(fig)
        charts = '<img src="learning.png" alt="Measured tournament fitness and population parameter diversity by generation">'
    takeaway = ('Passed the preregistered qualification and confirmatory gates.' if qualified else
                'The pilot failed its frozen feasibility gates. Full training was not launched.' if status == 'pilot_failed' else
                'The completed study did not qualify a playable replacement.' if results else
                'No completed strength comparison is available. The model is not playable.')
    detail = html.escape(json.dumps(dict(pilot=pilot, results=results, forecasts=evidence['forecasts']), indent=2))
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evolutionary chess — methods and results</title><style>body{{max-width:950px;margin:3rem auto;padding:0 1rem;font:17px/1.6 system-ui;background:#10151c;color:#e8edf4}}a{{color:#8fd7cb}}img{{max-width:100%}}pre{{overflow:auto;font-size:13px}}section{{margin:2rem 0}}</style>
<nav><a href="/research?study=evolutionary">Research journal</a></nav><h1>Evolutionary chess</h1>
<p><strong>{html.escape(status.replace('_',' '))}</strong> · {'Qualified' if qualified else 'Research only'}</p><p>{takeaway}</p>
<section><h2>What learns</h2><p>A 63-input neural evaluator has 324 learned parameters: four ReLU units, a learned linear connection and a bounded scalar output. Counts, pawn structure, king safety and attack coverage are human-designed inputs. All evaluation weights start randomly; no fixed material score, Stockfish labels, pretrained weights or gradient updates enter training. Our alpha-beta search chooses moves.</p></section>
<section><h2>How it learns</h2><p>CMA-ES evaluates populations against Classical, the released Supervised evaluator and archived evolutionary opponents. Every candidate receives the same color-swapped positions and playing budgets within a generation. Historical opponents and changing curriculum make tournament fitness a development measure, not a stable strength rating.</p></section>
<section><h2>Evidence and limits</h2><p>{evidence['matches']} recorded matches; paid compute C$0. No Elo or calibrated win-probability claim. Capped games retain unfinished rule results and are reported separately. Opening-pair uncertainty and variation between training seeds are different quantities.</p>{charts}<pre>{detail}</pre></section>
<section><h2>Reproducible records</h2><p><a href="protocol.json">Frozen protocol</a> · <a href="evidence.json">Measured evidence</a></p></section></html>'''
    (destination/'index.html').write_text(document)
    public = Path(public_destination)
    public.mkdir(parents=True, exist_ok=True)
    for name in ['index.html', 'evidence.json', 'protocol.json', *(['learning.png'] if histories else [])]:
        shutil.copyfile(destination/name, public/name)
    return evidence


def release(config):
    root = Path(config['output'])
    results = read_optional(root/'results.json')
    if not results or not results['release_eligible']:
        return dict(published=False, reason='Completed passing qualification and fresh-final gates required')
    from .evolution_study import release_gate
    from .evolution_games import audit_ledger
    selections = read_json(root/'selections.json')
    # Independently recompute gates from saved match ledgers; a boolean is not evidence.
    for stage in ('qualification', 'final-production'):
        sets = {name: [read_json(root/f'seed-{seed}/{stage}-{name}.json') for seed in config['seeds']]
                for name in ('classical', 'supervised')}
        if not release_gate(sets, config)['passed']:
            raise ValueError('Release evidence failed independent gate recomputation')
        for ledgers in sets.values():
            for seed, ledger in zip(config['seeds'], ledgers):
                audit_ledger(ledger)
                selected = selections[str(seed)]
                if digest(selected['checkpoint']) != selected['sha256']:
                    raise ValueError('Release candidate changed')
                if any(g['candidate'] != selected['checkpoint'] for g in ledger['games']):
                    raise ValueError('Release match candidate identity mismatch')
    selected = selections[str(config['representative_seed'])]
    source = selected['checkpoint']
    value = EvolutionEvaluator(source)
    target = Path('artifacts/models/evolutionary.npz')
    export(value.parameters, target, dict(seed=config['representative_seed'], source_checkpoint_sha256=digest(source)))
    published = EvolutionEvaluator(target)
    if not np.array_equal(value.parameters, published.parameters):
        raise ValueError('Evolutionary export parity failure')
    shutil.copyfile(target, 'web/public/models/evolutionary.npz')
    manifest = read_json('artifacts/models/manifest.json')
    if manifest.get('release_version') != 3 or manifest['default'] != 'supervised':
        raise ValueError('Unexpected active release manifest')
    manifest['models'] = [r for r in manifest['models'] if r['id'] != 'evolutionary'] + [
        dict(id='evolutionary', kind='evolutionary', name='Evolutionary model', file='evolutionary.npz',
             sha256=digest(target), source_sha256=digest(source), documentation_path='/research?study=evolutionary')]
    manifest['required_model_ids'] = ['supervised', 'evolutionary']
    for path in ['artifacts/models/manifest.json', 'web/public/models/manifest.json']:
        write_json(path, manifest)
    return dict(published=True, checkpoint_sha256=digest(target), default='supervised')
