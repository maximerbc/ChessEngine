"""Publish only aggregate human-move evidence; raw games/checkpoints stay private."""
import argparse
import os
from html import escape
from pathlib import Path
import shutil
import tempfile
from .common import read_json, digest
from .elite_data import atomic_json


def learning_curves(results, destination):
    """Plot actual epoch records only; final-test scores never form a selection curve."""
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'chessengine-elite-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    colors = ['#276749', '#2563a6', '#ad5a20']
    has_history = False
    for i, seed in enumerate(results.get('seeds', [])):
        history = seed.get('history', [])
        if not history:
            continue
        has_history = True
        epochs = [r['epoch'] for r in history]
        label = f"Seed {seed['seed']}" + (' (failed)' if seed.get('status') == 'failed' else '')
        color = colors[i % len(colors)]
        axes[0].plot(epochs, [r['validation']['cross_entropy'] for r in history], color=color, label=label+' validation')
        axes[0].plot(epochs, [r['online_training_loss'] for r in history], color=color,
                     linestyle='--', alpha=.7, label=label+' online training')
        axes[1].plot(epochs, [r['validation']['top1'] for r in history], color=color, label=label)
        selected = next((r for r in history if r['epoch'] == seed.get('selected_epoch')), None)
        if selected:
            axes[1].scatter([selected['epoch']], [selected['validation']['top1']], color=color, marker='*', s=90)
    if not has_history:
        plt.close(fig)
        return False
    first = next(s for s in results['seeds'] if s.get('history'))
    baseline = first['history'][0]['validation']['uniform_top1']
    axes[1].axhline(baseline, color='#666666', linestyle=':', label='Uniform legal baseline')
    axes[0].set(title='Global-vocabulary cross-entropy', ylabel='Cross-entropy', xlabel='Epoch')
    axes[1].set(title='Validation human-move agreement', ylabel='Legal top-one agreement', xlabel='Epoch')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, frameon=False)
    fig.suptitle('Elite human-move CNN · frozen baseline')
    fig.savefig(destination / 'curves.svg', metadata={'Description':
        'Actual epoch records; dashed training loss is online, solid loss is validation; stars mark validation-CE selections.'})
    plt.close(fig)
    return True


def report(root, repository='.'):
    root, repository = Path(root), Path(repository)
    data = read_json(root / 'dataset.json') if (root / 'dataset.json').exists() else None
    execution = read_json(root / 'execution.json') if (root / 'execution.json').exists() else {}
    gate = read_json(root / 'gate.json') if (root / 'gate.json').exists() else dict(passed=False)
    results = read_json(root / 'results.json') if (root / 'results.json').exists() else None
    complete = bool(results and len(results.get('seeds', [])) == 3)
    running = not complete and execution.get('status') in ['training', 'benchmark-and-training-chain-launched']
    operations = read_json(root / 'run-summary.json') if (root / 'run-summary.json').exists() else {}
    manifest = read_json(repository / 'artifacts/models/manifest.json')
    installed = next((r for r in manifest['models'] if r['id'] == 'elite-policy'), None)
    playable = bool(installed and gate['passed'] and installed['sha256'] == gate.get('checkpoint_sha256'))
    evidence = dict(study='elite-pytorch-baseline-20261124',
        status='complete' if complete else 'training-in-progress' if running else 'awaiting-approved-runpod-training',
        playable=playable, qualified=False, strength_status='not-established',
        counts=data['counts'] if data else None,
        dataset_sha256=digest(root / 'dataset.json') if data else None,
        label_source='elite-human-played-moves', results=results, gate=gate, operations=operations)
    destination = repository / 'docs/experiments/elite-policy'
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / 'evidence.json', evidence)
    curves = complete and learning_curves(results, destination)
    rows = []
    for seed in (results or {}).get('seeds', []):
        if seed.get('status') == 'failed':
            rows.append(f"<tr><td>{seed['seed']}</td><td colspan='6'>Failed; preserved in evidence</td></tr>")
        else:
            scores = seed['test']
            interval = scores['intervals']['top1']
            rows.append(f"<tr><td>{seed['seed']}</td><td>{seed['selected_epoch']}</td>"
                        f"<td>{scores['cross_entropy']:.5f}</td><td>{scores['top1']:.2%}<br><small>95% CI {interval[0]:.2%}–{interval[1]:.2%}</small></td>"
                        f"<td>{scores['top3']:.2%}</td><td>{scores['mean_rank']:.2f}</td>"
                        f"<td>{'pass' if seed['export']['passed'] else 'FAIL'}</td></tr>")
    status = ('Completed baseline; experimental playable access ' + ('enabled.' if playable else 'not enabled.')
              if complete else 'Approved three-seed training is in progress. Test results remain sealed until every selection is frozen.'
              if running else 'Training has not started. The fourth demo choice remains unavailable.')
    validation_ci = gate.get('validation_top1_difference_ci')
    learning = (f"Seed 42's paired validation difference from uniform legal selection is "
                f"{validation_ci[0]:.2%} to {validation_ci[1]:.2%} (95% whole-game bootstrap)."
                if validation_ci else 'The validation learning gate has not been evaluated yet.')
    cost_text = operations.get('cost_description', 'Final billing and elapsed-time reconciliation are pending.')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Elite human-move baseline</title>
<style>body{{max-width:850px;margin:3rem auto;padding:1rem;font:18px/1.7 system-ui}}td,th{{text-align:left;padding:.5rem}}table{{width:100%;min-width:640px;border-collapse:collapse}}.table-scroll{{max-width:100%;overflow-x:auto}}tr{{border-bottom:1px solid #bbb}}code{{overflow-wrap:anywhere}}</style>
<nav><a href="/">Demo</a> · <a href="/research">Research</a></nav>
<h1>Learning moves from elite human games</h1><p>{escape(status)}</p>
<p>This independently implemented baseline follows the <a href="https://github.com/Skripkon/chess-engine/tree/97238ead1892adca5e8626c568cd8129d5993ee9/engines/torch">Skripkon PyTorch method</a>:
13 absolute board planes, two convolution layers, a 256-unit dense layer and a complete 4208-class move vocabulary.
Labels are moves actually played by humans, not Stockfish best moves.</p>
<p>The frozen dataset contains 800,000 training, 100,000 validation and 100,000 test examples from fresh
elite games. Whole-game splits, position deduplication and prior-corpus/reserved-pool exclusions precede training.
Each label is aligned to the board before its move.</p>
<h2>What the model does</h2><p>Choose the highest-scoring legal move directly, without search.
The original Supervised and Augmented Supervised models instead learn Stockfish position assessments and use alpha-beta.
Policy probabilities rank legal move predictions; they are not winning chances or position evaluations.</p>
<h2>Frozen training protocol</h2><p>Fresh seeds 42/43/44, FP32, Adam 0.0001, batch 64, gradient clipping 1.0,
100 epochs. Select minimum validation cross-entropy with earliest-epoch ties, seal all seed attempts,
then evaluate held-out tests. Seed 42 remains representative. No architecture search follows these results automatically.</p>
<h2>Results</h2>{('<div class="table-scroll" role="region" aria-label="Seed results" tabindex="0"><table><thead><tr><th>Seed</th><th>Selected epoch</th><th>Test CE</th><th>Top one</th><th>Top three</th><th>Mean target rank</th><th>CPU export</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>') if complete else '<p>No trained result is claimed. Preparation and synthetic correctness tests are not study outcomes.</p>'}
{('<figure style="margin:1rem 0"><img src="curves.svg" style="width:100%;height:auto" alt="Actual training and validation curves for every seed"><figcaption>Dashed loss is online training loss; solid loss is validation CE. Stars mark validation-CE checkpoint selections. Final tests do not select checkpoints.</figcaption></figure>') if curves else ''}
<p>{escape(learning)}</p>
<p>Full metrics, whole-game intervals, failures, hashes and the validation gate are in <a href="evidence.json">aggregate evidence</a>.
Between-seed variation is distinct from whole-game bootstrap uncertainty. Human-move agreement does not establish playing strength or Elo.</p>
<h2>Runtime and cost</h2><p>{escape(cost_text)}</p>
<h2>Limitations and access</h2><p>The reference input omits explicit turn, castling-right, en-passant,
halfmove-clock and repetition planes. Complete histories still enforce legality and automatic draws.
Experimental playable access requires correctness, CPU export parity and a positive lower 95% whole-game
bound on validation top-one accuracy minus the uniform-legal baseline. It does not confer strength qualification.</p>
<p>No old model weights initialize this CNN. Historical Stockfish-policy, self-play and evolutionary studies remain separate.</p></html>'''
    (destination / 'index.html').write_text(html)
    text = ('# Elite human-move baseline\n\n' + status + '\n\n'
            'Protocol: `docs/elite-policy.md`. Full aggregate evidence: `evidence.json`.\n'
            'No strength claim follows from human-move agreement.\n')
    (destination / 'README.md').write_text(text)
    public = repository / 'web/public/research/elite-policy'
    public.mkdir(parents=True, exist_ok=True)
    for name in ['index.html', 'evidence.json'] + (['curves.svg'] if curves else []):
        shutil.copy2(destination / name, public / name)
    index_path = repository / 'web/public/research/studies.json'
    index = read_json(index_path)
    index['studies'] = [s for s in index['studies'] if s['id'] != evidence['study']]
    index['studies'].insert(0, dict(id=evidence['study'], title='Elite human-move prediction',
        level='PyTorch direct-move baseline', result='complete' if complete else 'running' if running else 'planned',
        availability='experimental playable' if playable else 'awaiting validation',
        question='Can a small CNN learn useful move preferences from elite human games?',
        takeaway=status + ' Human-move agreement is separate from playing strength.',
        report='/research/elite-policy/index.html',
        checkpoint_ids={'elite-policy': installed['sha256']} if playable else {}))
    index['released_models'] = {r['id']: r['sha256'] for r in manifest['models']}
    index.setdefault('assets', {})['elite-policy/index.html'] = digest(public / 'index.html')
    index['assets']['elite-policy/evidence.json'] = digest(public / 'evidence.json')
    if curves:
        index['assets']['elite-policy/curves.svg'] = digest(public / 'curves.svg')
    atomic_json(index_path, index)
    return evidence


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    print(report(a.root)['status'])
