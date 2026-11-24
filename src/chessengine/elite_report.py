"""Publish only aggregate human-move evidence; raw games/checkpoints stay private."""
import argparse
from html import escape
from pathlib import Path
import shutil
from .common import read_json, digest
from .elite_data import atomic_json


def report(root, repository='.'):
    root, repository = Path(root), Path(repository)
    complete = (root / 'results.json').exists()
    data = read_json(root / 'dataset.json') if (root / 'dataset.json').exists() else None
    gate = read_json(root / 'gate.json') if (root / 'gate.json').exists() else dict(passed=False)
    results = read_json(root / 'results.json') if complete else None
    manifest = read_json(repository / 'artifacts/models/manifest.json')
    installed = next((r for r in manifest['models'] if r['id'] == 'elite-policy'), None)
    playable = bool(installed and gate['passed'] and installed['sha256'] == gate.get('checkpoint_sha256'))
    evidence = dict(study='elite-pytorch-baseline-20261124',
        status='complete' if complete else 'awaiting-approved-runpod-training',
        playable=playable, qualified=False, strength_status='not-established',
        counts=data['counts'] if data else None,
        dataset_sha256=digest(root / 'dataset.json') if data else None,
        label_source='elite-human-played-moves', results=results, gate=gate)
    destination = repository / 'docs/experiments/elite-policy'
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / 'evidence.json', evidence)
    rows = []
    for seed in (results or {}).get('seeds', []):
        if seed.get('status') == 'failed':
            rows.append(f"<tr><td>{seed['seed']}</td><td colspan='5'>Failed; preserved in evidence</td></tr>")
        else:
            scores = seed['test']
            rows.append(f"<tr><td>{seed['seed']}</td><td>{seed['selected_epoch']}</td>"
                        f"<td>{scores['cross_entropy']:.5f}</td><td>{scores['top1']:.2%}</td>"
                        f"<td>{scores['top3']:.2%}</td><td>{'pass' if seed['export']['passed'] else 'FAIL'}</td></tr>")
    status = ('Completed baseline; experimental playable access ' + ('enabled.' if playable else 'not enabled.')
              if complete else 'Training has not started. The fourth demo choice remains unavailable.')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Elite human-move baseline</title>
<style>body{{max-width:850px;margin:3rem auto;padding:1rem;font:18px/1.7 system-ui}}td,th{{text-align:left;padding:.5rem}}table{{width:100%;border-collapse:collapse}}tr{{border-bottom:1px solid #bbb}}code{{overflow-wrap:anywhere}}</style>
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
<h2>Results</h2>{('<table><thead><tr><th>Seed</th><th>Selected epoch</th><th>Test CE</th><th>Top one</th><th>Top three</th><th>CPU export</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>') if complete else '<p>No trained result is claimed. Preparation and synthetic correctness tests are not study outcomes.</p>'}
<p>Full metrics, whole-game intervals, failures, hashes and the validation gate are in <a href="evidence.json">aggregate evidence</a>.
Between-seed variation is distinct from whole-game bootstrap uncertainty. Human-move agreement does not establish playing strength or Elo.</p>
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
    for name in ['index.html', 'evidence.json']:
        shutil.copy2(destination / name, public / name)
    index_path = repository / 'web/public/research/studies.json'
    index = read_json(index_path)
    index['studies'] = [s for s in index['studies'] if s['id'] != evidence['study']]
    index['studies'].insert(0, dict(id=evidence['study'], title='Elite human-move prediction',
        level='PyTorch direct-move baseline', result='complete' if complete else 'planned',
        availability='experimental playable' if playable else 'awaiting validation',
        question='Can a small CNN learn useful move preferences from elite human games?',
        takeaway=status + ' Human-move agreement is separate from playing strength.',
        report='/research/elite-policy/index.html',
        checkpoint_ids={'elite-policy': installed['sha256']} if playable else {}))
    index['released_models'] = {r['id']: r['sha256'] for r in manifest['models']}
    index.setdefault('assets', {})['elite-policy/index.html'] = digest(public / 'index.html')
    index['assets']['elite-policy/evidence.json'] = digest(public / 'evidence.json')
    atomic_json(index_path, index)
    return evidence


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    print(report(a.root)['status'])
