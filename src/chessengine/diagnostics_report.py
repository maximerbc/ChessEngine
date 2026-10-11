"""Generate a separate, static diagnostic report from saved study evidence."""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import shutil

import chess
import chess.svg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import read_json, write_json
from .data import records


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(
        f'<th>{escape(str(h))}</th>' for h in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{escape(str(c))}</td>' for c in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def number(value):
    return '—' if value is None else f'{value:.5f}'


def interval(values):
    return f'[{values[0]:.5f}, {values[1]:.5f}]'


def recommendations(result):
    promising = [r for r in result['summary'] if r['promising']]
    baseline = sorted([r for r in result['summary'] if r['experiment'] == 'baseline'], key=lambda r: r['positions'])
    statements = []
    if promising:
        winner = promising[0]
        statements.append(
            f"Confirm {winner['experiment']} in a later independent evaluation: validation MSE fell by "
            f"{winner['relative_improvement_vs_full_baseline']:.1%} versus the three-seed full-data baseline. "
            "This meets the predefined development threshold, but selection across multiple configurations "
            "can exaggerate the apparent benefit. No playing-strength improvement has been established."
        )
    if len(baseline) > 1 and baseline[-1]['mean_mse'] < baseline[0]['mean_mse']:
        statements.append(
            "Test broader training coverage next: the largest training subset has lower mean validation MSE "
            "than the smallest. More data also received more optimizer updates, so this is evidence for the "
            "combined training procedure rather than a fixed-compute causal effect."
        )
    if not promising:
        statements.append(
            "Do not increase regularization based on these results alone. Audit teacher-label stability and "
            "position coverage before changing capacity or launching a larger search."
        )
    statements.append(
        "Audit deeper Stockfish labels on a fixed development sample before mass relabelling. "
        "Large residuals alone cannot distinguish a noisy teacher, insufficient representation, "
        "optimization failure, or an unfamiliar position."
    )
    return statements


def generate(study, output):
    study, output = Path(study), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / 'evidence'
    evidence.mkdir(exist_ok=True)
    result, audit = read_json(study / 'results.json'), read_json(study / 'audit.json')
    raw = np.load(study / 'predictions.npz', allow_pickle=False)
    curves = {}
    for run in result['runs']:
        key = f"{run['experiment']}-{run['size']}-seed-{run['seed']}"
        curves[key] = read_json(study / key / 'metrics.json')['history']
    for name in ('results', 'audit', 'protocol', 'subsets', 'screening'):
        shutil.copyfile(study / f'{name}.json', evidence / f'{name}.json')
    write_json(evidence / 'curves.json', curves)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    train_targets = np.array([r['target'] for r in records(study / 'subset-all.jsonl') if r['split'] == 'train'])
    bins = np.linspace(-1, 1, 31)
    axes[0].hist(train_targets, bins=bins, density=True, histtype='step', label='Training', color='#b56130')
    axes[0].hist(raw['targets'], bins=bins, density=True, histtype='step', label='Validation', color='#1e766e')
    axes[0].legend()
    axes[0].set(xlabel='Teacher target', ylabel='Density', title='Target distributions')
    write_json(evidence / 'target-histograms.json', {'bin_edges': bins.tolist(),
               'train_counts': np.histogram(train_targets, bins)[0].tolist(),
               'validation_counts': np.histogram(raw['targets'], bins)[0].tolist()})
    density = axes[1].hexbin(raw['targets'], raw['supervised'], gridsize=35, mincnt=1, cmap='viridis')
    fig.colorbar(density, ax=axes[1], label='Positions per hexagon')
    axes[1].plot([-1, 1], [-1, 1], '--', color='#ad4134')
    axes[1].set(xlabel='Teacher target', ylabel='Frozen model prediction', title='Agreement with the teacher')
    fig.savefig(output / 'predictions.png', dpi=160)
    plt.close(fig)
    baseline = sorted([r for r in result['summary'] if r['experiment'] == 'baseline'], key=lambda r: r['positions'])
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for seed in result['protocol']['identity']['config']['seeds']:
        points = sorted([r for r in result['runs'] if r['experiment'] == 'baseline' and r['seed'] == seed],
                        key=lambda r: r['positions'])
        ax.plot([r['positions'] for r in points], [r['validation']['mse'] for r in points],
                'o-', alpha=.6, label=f'Seed {seed}')
    ax.errorbar([r['positions'] for r in baseline], [r['mean_mse'] for r in baseline],
                yerr=[r['std_mse'] for r in baseline], fmt='k.--', capsize=5, label='Mean ± seed SD')
    ax.set(xlabel='Actual training positions', ylabel='Validation MSE', title='Data-size learning curve')
    ax.legend()
    fig.savefig(output / 'data-size.png', dpi=160)
    plt.close(fig)
    names = list(dict.fromkeys(r['experiment'] for r in result['runs']))
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 3 * len(names)), squeeze=False, layout='constrained')
    for row, name in enumerate(names):
        for col, seed in enumerate(result['protocol']['identity']['config']['seeds']):
            ax = axes[row, col]
            for run in [r for r in result['runs'] if r['experiment'] == name and r['seed'] == seed]:
                key = f"{name}-{run['size']}-seed-{seed}"
                h = curves[key]
                line, = ax.plot([p['epoch'] for p in h], [p['validation_loss'] for p in h],
                                label=f"{run['positions']:,} validation")
                ax.plot([p['epoch'] for p in h], [p['train_loss'] for p in h], '--', color=line.get_color(),
                        alpha=.65, label=f"{run['positions']:,} online train")
            ax.set(title=f'{name} · seed {seed}', xlabel='Epoch', ylabel='MSE')
            ax.legend(fontsize=7)
    fig.savefig(output / 'training-curves.png', dpi=140)
    plt.close(fig)
    rank = recommendations(result)
    write_json(evidence / 'recommendations.json', rank)
    sections = [
        '<p class="eyebrow">CHESSENGINE · DEVELOPMENT STUDY · 09 OCTOBER 2026</p>',
        '<h1>What limits the supervised evaluator?</h1>',
        '<p class="lead">A frozen-model audit, nine data-size experiments, and a predefined conditional '
        'regularization comparison. This report evaluates approximation to Stockfish targets—not chess strength.</p>',
        '<div class="notice">The public demo and released checkpoints are unchanged. '
        'All results use existing training/validation data. No final-test results entered selection.</div>',
        '<h2>Findings and next steps</h2><ol>' + ''.join(f'<li>{escape(s)}</li>' for s in rank) + '</ol>',
        '<h2>Frozen checkpoint audit</h2>',
        f"<p>{audit['training_positions']:,} training positions; {audit['validation_positions']:,} validation "
        f"positions from {audit['validation_games']:,} source games. Training-mean target: "
        f"{audit['training_mean']:.5f}. Frozen-checkpoint training MSE: {audit['train']['mse']:.5f}.</p>",
        table(['Evaluator', 'MSE', '95% game-bootstrap CI', 'MAE', 'Signed error', 'MSE reduction vs mean'], [
            [name, number(m['mse']), interval(audit['bootstrap']['intervals'][name]['mse']),
             number(m['mae']), number(m['signed_error']),
             'undefined' if m['relative_mse_reduction_vs_training_mean'] is None
             else f"{m['relative_mse_reduction_vs_training_mean']:.1%}"]
            for name, m in audit['validation'].items()]),
        '<p>Signed error is prediction minus teacher target. Bootstrap intervals resample whole source games '
        'and retain position weighting; they describe validation-sample uncertainty, not retraining variability. '
        'The full MSE, MAE and signed-error intervals are in the evidence.</p>',
        '<h3>Paired uncertainty</h3>' + table(['Baseline', 'Supervised − baseline MSE, 95% CI',
                                            'MAE difference, 95% CI', 'Signed-error difference, 95% CI'], [
            [name, interval(v['mse']), interval(v['mae']), interval(v['signed_error'])]
            for name, v in audit['bootstrap']['paired_supervised_minus_baseline'].items()]),
        '<img src="predictions.png" alt="Validation target histogram and teacher versus model predictions">',
        '<h2>Data size, regularization and training seeds</h2>',
        '<p>Nested subsets add complete source games in a deterministic hash order (sampling seed 2026). '
        'The validation set is fixed. Larger datasets receive more updates per epoch; this is not a '
        'fixed-compute comparison. Historical training used a different record order, so the new full-data '
        'baseline need not reproduce the released checkpoint.</p>',
        '<img src="data-size.png" alt="Validation MSE against actual training set size across seeds">',
        table(['Experiment', 'Positions', 'Mean validation MSE', 'Seed SD', 'Improved seeds vs full baseline',
               'Promising'], [[r['experiment'], r['positions'], number(r['mean_mse']), number(r['std_mse']),
                              r['improved_seeds'], 'yes' if r['promising'] else 'no'] for r in result['summary']]),
        '<h3>Predefined screening rule</h3>',
        f"<p>Triggered: <strong>{result['screening']['triggered']}</strong>. At least two of three full-data "
        'seeds must have validation MSE more than 10% above frozen-checkpoint training MSE. '
        'This schedules an experiment; it is not proof of overfitting.</p>',
        table(['Seed', 'Frozen train MSE', 'Validation MSE', 'Exceeds 10% gap'], [
            [r['seed'], number(r['train_mse']), number(r['validation_mse']), r['exceeds']]
            for r in result['screening']['per_seed']]),
        '<p>A treatment is called promising only if mean validation MSE improves by at least 2% and '
        'at least two seeds improve. These are development criteria, not corrected significance tests.</p>',
        '<h3>Every run</h3>',
        table(['Experiment', 'Positions', 'Seed', 'Frozen train MSE', 'Validation MSE', 'Best / total epochs',
               'Total updates', 'Training seconds', 'Run wall seconds'], [
            [r['experiment'], r['positions'], r['seed'], number(r['train']['mse']), number(r['validation']['mse']),
             f"{r['selected_epoch']} / {r['epochs']}", r['optimizer_steps'], f"{r['training_seconds']:.2f}",
             f"{r['run_wall_seconds']:.2f}"] for r in result['runs']]),
        '<img src="training-curves.png" alt="Training and validation curves for every experiment and seed">',
        '<p>Dashed curves are online training losses, accumulated as weights change (and with dropout active '
        'where configured). Solid curves use frozen epoch-end weights in evaluation mode. Use the table’s '
        'frozen-checkpoint metrics to assess the gap; do not directly compare the dashed and solid curves.</p>',
        '<h2>Where errors concentrate</h2><p>Move-count labels are approximate phases, not material-based '
        'endgame classifications. Material phase units: knight/bishop 1, rook 2, queen 4. The high bin is '
        '17+ to include promoted pieces beyond the normal starting total of 24. Imbalance uses the '
        'engine’s existing pawn-unit piece values. Groups below 100 positions are marked sparse.</p>',
    ]
    for dimension, groups in audit['breakdown'].items():
        sections.append(f'<h3>{escape(dimension.replace("_", " "))}</h3>')
        sections.append(table(['Group', 'Positions / games', 'Evaluator', 'MSE', 'MAE', 'Signed error'], [
            [label + (' (sparse)' if group['sparse'] else ''), f"{group['positions']} / {group['games']}",
             name, number(m['mse']), number(m['mae']), number(m['signed_error'])]
            for label, group in groups.items() for name, m in group['metrics'].items()]))
    sections.append('<h2>Twenty largest validation errors</h2><p>At most one position per source game; '
                    'selected by squared error of the released supervised checkpoint. These are failure '
                    'examples, not a representative tactical test. Targets and predictions use side-to-move '
                    'perspective; boards are oriented to that player.</p><div class="boards">')
    for i, row in enumerate(audit['worst_positions'], 1):
        board = chess.Board(row['fen'])
        diagram = chess.svg.board(board, orientation=board.turn, size=300)
        sections.append(f'<article><h3>Example {i} · {"White" if board.turn else "Black"} to move</h3>'
                        + diagram + f"<p>Prediction <b>{row['prediction']:.4f}</b> · teacher "
                        f"<b>{row['target']:.4f}</b></p><code>{escape(row['fen'])}</code></article>")
    sections.append('</div><h2>Reproducibility and limitations</h2>')
    sections.append(f"<p>Training/validation epoch work: {result['training_seconds']:.1f} seconds. "
                    f"Study invocation wall time: {result['invocation_wall_seconds']:.1f} seconds. "
                    "Paid compute: C$0. Per-run wall times describe completed invocations and exclude any "
                    "interrupted attempts; epoch work excludes encoding, serialization and report generation.</p>")
    sections.append('<p>Targets are finite-search Stockfish assessments, not ground truth or observed win '
                    'probabilities. Validation has been used repeatedly for early stopping and development. '
                    'Group analysis is exploratory and multiple comparisons are not adjusted. Seed SD and '
                    'game-bootstrap uncertainty measure different sources of variation. Independent final '
                    'positions and controlled matches are required before claiming generalization or strength '
                    'improvements. Raw data, predictions and intermediate checkpoints remain local.</p>')
    sections.append('<ul>' + ''.join(f'<li><a href="evidence/{name}.json">{name}</a></li>'
                                    for name in ['protocol', 'results', 'audit', 'subsets', 'screening',
                                                 'curves', 'recommendations']) + '</ul>')
    css = '''body{margin:0;background:#f5f6f2;color:#1e2927;font:16px/1.6 system-ui,sans-serif}
    main{max-width:1120px;margin:auto;padding:48px 28px}h1{font-size:44px;line-height:1.15;max-width:850px}
    h2{margin-top:48px;border-top:1px solid #cad5cf;padding-top:24px}h3{margin-top:25px}
    .eyebrow{letter-spacing:.12em;font-size:12px;color:#36665c}.lead{font-size:20px;max-width:850px}
    .notice{background:#e0eee6;padding:18px;border-left:4px solid #237367}img{max-width:100%;height:auto}
    .table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}
    th{text-align:left;background:#dfe8e3}td,th{padding:10px;border-bottom:1px solid #ced9d2}
    tbody tr:nth-child(even){background:#edf1ec}.boards{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}
    article{background:white;padding:16px;min-width:0}article svg{width:100%;height:auto}code{font-size:11px;overflow-wrap:anywhere}
    li{margin-bottom:12px}a{color:#176a61}@media(max-width:700px){.boards{grid-template-columns:1fr}h1{font-size:32px}}
    '''
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
    html += '<title>ChessEngine · Supervised diagnostics</title><style>' + css + '</style><main>'
    html += '\n'.join(sections) + '</main></html>'
    (output / 'index.html').write_text(html)
    return output / 'index.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study')
    parser.add_argument('output')
    args = parser.parse_args()
    print(generate(args.study, args.output))


if __name__ == '__main__':
    main()
