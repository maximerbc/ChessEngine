"""Compare the frozen 80k baseline with the separately approved 200k-corpus extension."""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from .common import digest, read_json, write_json
from .data import records
from .diagnostics import grouped_bootstrap, metrics
from .diagnostics_report import table
from .features import encode, restore
from .model import NumpyEvaluator


def generate(previous, expanded, output, expansion_config):
    previous, expanded, output = Path(previous), Path(expanded), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / 'evidence'
    evidence.mkdir(exist_ok=True)
    old, new = read_json(previous / 'results.json'), read_json(expanded / 'results.json')
    before = {r['seed']: r for r in old['runs'] if r['experiment'] == 'baseline' and r['size'] == 'all'}
    after = {r['seed']: r for r in new['runs'] if r['experiment'] == 'baseline' and r['size'] == 'all'}
    if before.keys() != after.keys():
        raise ValueError('Training seed mismatch')
    old_source = old['protocol']['identity']['config']['source']
    new_source = new['protocol']['identity']['config']['source']
    validation = [r for r in records(old_source) if r['split'] == 'validation']
    if validation != [r for r in records(new_source) if r['split'] == 'validation']:
        raise ValueError('Validation set changed')
    x = np.stack([encode(restore(r)) for r in validation])
    y = np.array([r['target'] for r in validation], dtype=np.float32)
    games = np.array([r['game_id'] for r in validation])
    raw, comparisons = {'targets': y, 'game_ids': games}, []
    for seed in sorted(before):
        checkpoints = [previous / f'baseline-all-seed-{seed}' / 'best.npz',
                       expanded / f'baseline-all-seed-{seed}' / 'best.npz']
        if any(digest(p) != r['checkpoint_sha256'] for p, r in zip(checkpoints, [before[seed], after[seed]])):
            raise ValueError('Comparison checkpoint changed')
        with threadpool_limits(limits=1):
            a, b = [NumpyEvaluator(p).batch(x).astype(float) for p in checkpoints]
        for prediction, row in zip([a, b], [before[seed], after[seed]]):
            if not np.isclose(metrics(prediction, y)['mse'], row['validation']['mse'], atol=1e-8):
                raise ValueError('Saved validation metric does not match checkpoint')
        raw[f'old_{seed}'], raw[f'new_{seed}'] = a, b
        bootstrap = grouped_bootstrap({'supervised': b, 'old_baseline': a}, y.astype(float), games,
                                      resamples=2000, seed=2026)
        comparisons.append({'seed': seed, 'old': metrics(a, y), 'new': metrics(b, y),
                            'new_minus_old_mse': metrics(b, y)['mse'] - metrics(a, y)['mse'],
                            'bootstrap': bootstrap})
    np.savez_compressed(expanded / 'comparison-predictions.npz', **raw)
    old_errors = [r['validation']['mse'] for r in before.values()]
    new_errors = [r['validation']['mse'] for r in after.values()]
    comparison = {'old_positions': next(iter(before.values()))['positions'],
                  'new_positions': next(iter(after.values()))['positions'],
                  'validation_positions': len(y), 'validation_games': len(set(games)),
                  'old_mean_mse': float(np.mean(old_errors)), 'new_mean_mse': float(np.mean(new_errors)),
                  'old_seed_sd': float(np.std(old_errors, ddof=1)),
                  'new_seed_sd': float(np.std(new_errors, ddof=1)),
                  'relative_mse_reduction': float(1 - np.mean(new_errors) / np.mean(old_errors)),
                  'improving_seeds': sum(b < a for a, b in zip(old_errors, new_errors)),
                  'seed_comparisons': comparisons,
                  'raw_predictions_sha256': digest(expanded / 'comparison-predictions.npz')}
    write_json(evidence / 'comparison.json', comparison)
    write_json(evidence / 'previous-results.json', old)
    write_json(evidence / 'expanded-results.json', new)
    for source, name in [(Path(expansion_config['output']) / 'protocol.json', 'expansion-protocol'),
                         (Path(expansion_config['output']) / 'preparation.json', 'preparation'),
                         (Path(str(expansion_config['combined']) + '.manifest.json'), 'label-manifest')]:
        shutil.copyfile(source, evidence / f'{name}.json')
    curves = {}
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for seed in sorted(before):
        points = sorted([r for r in old['runs'] if r['experiment'] == 'baseline' and r['seed'] == seed],
                        key=lambda r: r['positions']) + [after[seed]]
        ax.plot([r['positions'] for r in points], [r['validation']['mse'] for r in points], 'o-', label=f'Seed {seed}')
    ax.set(xlabel='Training positions (held-out positions excluded)', ylabel='Validation MSE',
           title='More training data under the same epoch-based procedure')
    ax.legend()
    fig.savefig(output / 'data-size.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(before), figsize=(12, 4), squeeze=False, layout='constrained')
    for col, seed in enumerate(sorted(before)):
        ax = axes[0, col]
        for label, directory, row in [('Original', previous, before[seed]), ('Expanded', expanded, after[seed])]:
            history = read_json(directory / f'baseline-all-seed-{seed}' / 'metrics.json')['history']
            curves[f'{label}-{seed}'] = history
            ax.plot([v['optimizer_steps'] for v in history], [v['validation_loss'] for v in history],
                    'o-', markersize=3, label=f"{label}: {row['positions']:,}")
        ax.set(xlabel='Optimizer updates', ylabel='Validation MSE', title=f'Seed {seed}')
        ax.legend(fontsize=8)
    fig.savefig(output / 'training-curves.png', dpi=160)
    plt.close(fig)
    write_json(evidence / 'curves.json', curves)
    prep = read_json(evidence / 'preparation.json')
    labels = read_json(evidence / 'label-manifest.json')
    direction = 'fell' if comparison['relative_mse_reduction'] > 0 else 'rose'
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
    html += '<title>ChessEngine · 200k data extension</title><style>'
    html += ('body{background:#f5f6f2;color:#20322b;font:16px/1.6 system-ui;margin:0}'
             'main{max-width:1040px;margin:auto;padding:36px 24px}h1{font-size:38px;line-height:1.2}'
             'h2{margin-top:38px}img{max-width:100%}.table-wrap{overflow-x:auto}'
             'table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:left;'
             'border-bottom:1px solid #cad5cc}th{background:#dde8df}a{color:#14614e}</style><main>')
    html += '<p>CHESSENGINE · DEVELOPMENT EXPERIMENT</p><h1>Does more training data help?</h1>'
    html += (f"<p>Mean validation MSE {direction} from <b>{comparison['old_mean_mse']:.5f}</b> to "
             f"<b>{comparison['new_mean_mse']:.5f}</b> across three seeds "
             f"({abs(comparison['relative_mse_reduction']):.1%} {'reduction' if direction == 'fell' else 'increase'}). "
             f"{comparison['improving_seeds']} of three seeds improved. These are development results, "
             'not evidence of stronger chess play.</p>')
    html += (f"<p>The corpus now contains {prep['total_positions']:,} positions, including "
             f"{comparison['new_positions']:,} for training. Validation remains {len(y):,} positions from "
             f"{len(set(games)):,} source games. Existing test records are preserved and unused.</p>")
    html += '<img src="data-size.png" alt="Validation error at four training-set sizes across three seeds">'
    html += '<h2>Controlled settings and uncertainty</h2><p>The 128/64 MLP, AdamW settings, early stopping, '
    html += ('maximum epochs and validation set are fixed. New samples come from additional training-assigned '
             'games in the same January 2013 archive. New labels use the original Stockfish 19 binary, '
             '10,000 nodes, one thread and 32 MB hash per worker. Two independent workers label chunks.</p>')
    html += table(['Seed', 'Original MSE', 'Expanded MSE', 'Difference (new − old)', '95% game-bootstrap CI'], [
        [r['seed'], f"{r['old']['mse']:.5f}", f"{r['new']['mse']:.5f}", f"{r['new_minus_old_mse']:.5f}",
         '[' + ', '.join(f'{v:.5f}' for v in r['bootstrap']['paired_supervised_minus_baseline']['old_baseline']['mse']) + ']']
        for r in comparisons])
    html += (f"<p>Seed standard deviation: {comparison['old_seed_sd']:.5f} original; "
             f"{comparison['new_seed_sd']:.5f} expanded. Intervals use 2,000 paired source-game bootstrap "
             'resamples conditional on each selected checkpoint pair. They do not include retraining or '
             'model-selection uncertainty, and are not three independent test sets.</p>')
    html += '<h2>Optimization and compute</h2><img src="training-curves.png" alt="Validation curves against optimizer updates">'
    html += table(['Dataset', 'Seed', 'Frozen train MSE', 'Validation MSE', 'Selected epoch', 'Total updates',
                   'Epoch work (s)'], [[label, seed, f"{r['train']['mse']:.5f}", f"{r['validation']['mse']:.5f}",
                                      r['selected_epoch'], r['optimizer_steps'], f"{r['training_seconds']:.2f}"]
                                     for label, runs in [('Original', before), ('Expanded', after)]
                                     for seed, r in sorted(runs.items())])
    html += (f"<p>Preparation: {prep['wall_seconds']:.1f}s. Labeling elapsed time: "
             f"{labels['label_invocation_wall_seconds']:.1f}s; summed worker time: "
             f"{labels['label_worker_seconds']:.1f}s. Expanded training/validation epoch work: "
             f"{new['training_seconds']:.1f}s; diagnostic/training invocation: "
             f"{new['invocation_wall_seconds']:.1f}s. Report generation is excluded. Paid compute: C$0.</p>")
    html += ('<h2>Interpretation and next step</h2><p>Larger training sets receive more updates per epoch. '
             'The curves expose that difference, but this experiment is not an update-matched control. '
             'The result measures the combined effect of additional data and this training procedure. '
             'One sampling seed and one historical source limit conclusions about distribution coverage. '
             'Repeated use of validation data makes these development findings; reserve fresh final data '
             'before confirmatory evaluation.</p><p>Next, separate the effects of compute and data with '
             'a matched-update comparison, then assess any candidate in controlled chess matches. '
             'Released weights and public deployment remain unchanged.</p><h2>Reproducibility</h2><ul>')
    html += ''.join(f'<li><a href="evidence/{p.name}">{escape(p.stem)}</a></li>' for p in sorted(evidence.glob('*.json')))
    html += '</ul></main></html>'
    (output / 'index.html').write_text(html)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous')
    parser.add_argument('expanded')
    parser.add_argument('output')
    parser.add_argument('--expansion-config', default='configs/expansion-200k.json')
    args = parser.parse_args()
    print(generate(args.previous, args.expanded, args.output, read_json(args.expansion_config)))


if __name__ == '__main__':
    main()
