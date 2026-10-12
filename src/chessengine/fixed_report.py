"""Render fixed-data experiments without reading final-test data."""
from pathlib import Path
import argparse
from html import escape
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import read_json, write_json
from .diagnostics_report import table


def generate(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / 'evidence'
    evidence.mkdir(exist_ok=True)
    result = read_json(source / 'results.json')
    for name in ['results', 'protocol', 'audit', 'combination-decision']:
        shutil.copyfile(source / f'{name}.json', evidence / f'{name}.json')
    curves = {f"{r['name']}-{r['seed']}": read_json(source / f"{r['name']}-{r['seed']}" / 'metrics.json')['history']
              for r in result['runs']}
    write_json(evidence / 'curves.json', curves)
    write_json(evidence / 'reproduction.json', {str(s): read_json(source / f'reproduction-{s}.json')
                                               for s in result['protocol']['identity']['config']['seeds']})
    config = result['protocol']['identity']['config']
    summary = result['summary']
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.barh([r['name'] for r in summary], [r['mean_mse'] for r in summary],
            xerr=[r['seed_sd'] for r in summary], color='#287c70', capsize=4)
    ax.invert_yaxis()
    ax.set(xlabel='Mean validation MSE (error bars: seed SD)', title='Fixed examples, different training and representations')
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for col, seed in enumerate(result['protocol']['identity']['config']['seeds']):
        for row in [r for r in result['runs'] if r['seed'] == seed]:
            h = curves[f"{row['name']}-{seed}"]
            axes[0, col].plot([v['epoch'] for v in h], [v['validation_loss'] for v in h], label=row['name'])
            axes[1, col].plot([v['epoch'] for v in h], [v['learning_rate'] for v in h], label=row['name'])
        axes[0, col].set(title=f'Seed {seed}', xlabel='Epoch', ylabel='Validation MSE')
        axes[1, col].set(xlabel='Epoch', ylabel='Learning rate', yscale='log')
        axes[0, col].legend(fontsize=7)
    fig.savefig(output / 'curves.png', dpi=160)
    plt.close(fig)
    reference = next(r for r in summary if r['name'] == 'reference')
    best = summary[0]
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
    html += '<title>ChessEngine · Fixed-data improvements</title><style>'
    html += ('body{margin:0;background:#f5f6f2;color:#22342e;font:16px/1.6 system-ui}main{max-width:1100px;'
             'margin:auto;padding:32px 24px}h1{font-size:38px;line-height:1.2}h2{margin-top:40px}img{max-width:100%}'
             '.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}'
             'td,th{padding:9px;border-bottom:1px solid #ccd8d1;text-align:left}th{background:#deebe3}'
             'a{color:#136e59}</style><main><p>CHESSENGINE · DEVELOPMENT STUDY</p>')
    html += '<h1>Improving the evaluator without adding examples</h1>'
    html += (f"<p>Reference mean MSE: <b>{reference['mean_mse']:.5f}</b>. Lowest mean MSE: "
             f"<b>{best['mean_mse']:.5f}</b> ({escape(best['name'])}; reduction "
             f"{best['relative_reduction']:.1%}). Completed {len(result['runs'])} runs. "
             'Released models and deployment are unchanged.</p>')
    html += (f'<h2>Method and mathematical interpretation</h2><p>All {config["expected_counts"][0]:,} training positions, their labels '
             f'and {config["expected_counts"][1]:,} validation positions are fixed. Every run minimizes mean((prediction − target)²) '
             'with a 128/64 ReLU MLP and tanh output. Learning-rate treatments change the size of updates. '
             'Relative inputs orient pieces to the player to move; material summaries make counts, balance '
             'and game phase explicit. Targets keep their original sign. No augmentation or new labels.</p>'
             '<p>The 12 material inputs are linear summaries of the existing piece indicators: if m = Ax, '
             'the first layer Wₓx + Wₘm equals (Wₓ + WₘA)x. They add no information and do not enlarge '
             'the unconstrained function class relative to v2. They change the parameterization, '
             'initialization and effect of weight decay, which can make useful relationships easier to learn.</p>'
             '<p>Three reference runs must reproduce historical weights and validation MSE within 10⁻⁶ before '
             'treatments proceed. The canonical-position audit rejects cross-split overlaps. A promising '
             'treatment must reduce mean MSE by at least 2% and improve at least two seeds. Only the best '
             'qualifying optimization and representation treatments may be combined.</p>')
    html += '<img src="comparison.png" alt="Validation MSE and seed standard deviations by treatment">'
    html += table(['Treatment', 'Mean MSE', 'Seed SD', 'Reduction', 'Improved seeds', 'Promising', 'Inference µs'],
                  [[r['name'], f"{r['mean_mse']:.5f}",f"{r['seed_sd']:.5f}",f"{r['relative_reduction']:.1%}",
                    r['improving_seeds'], r['promising'], f"{r['latency_us']:.1f}"] for r in summary])
    html += ('<p>Conditional combination: ' + escape(' + '.join(result['combination']['parents'])) +
             ' qualified independently and were combined for three additional runs.</p>'
             if result['combination'] else '<p>No combination qualified; the study stopped after 15 runs.</p>')
    html += '<h2>Findings and recommended follow-up</h2>'
    if best['promising']:
        html += (f"<p>Prioritize <b>{escape(best['name'])}</b> for a separately frozen confirmatory evaluation: "
                 f"it improves {best['improving_seeds']} seeds and meets the development criterion. "
                 'Keep all three seed checkpoints for that evaluation; do not pick a lucky seed or replace the demo yet.</p>')
    else:
        html += '<p>No configuration meets the development criterion. Retain the reference and diagnose remaining errors before proposing another experiment.</p>'
    rejected = [r['name'] for r in summary if r['name'] != 'reference' and not r['promising']]
    html += '<p>Configurations not meeting the criterion: ' + escape(', '.join(rejected) or 'none') + '.</p>'
    best_runs = [r for r in result['runs'] if r['name'] == best['name']]
    ref_runs = [r for r in result['runs'] if r['name'] == 'reference']
    html += ('<p>The following exploratory subgroup comparison averages the three seeds; it is not a '
             'separate hypothesis test. Near-equal overall performance can hide subgroup regressions.</p>')
    html += table(['Dimension', 'Group', 'Reference MSE', 'Best treatment MSE', 'Difference'], [
        [dimension, label,
         f"{np.mean([r['breakdown'][dimension][label]['mse'] for r in ref_runs]):.5f}",
         f"{np.mean([r['breakdown'][dimension][label]['mse'] for r in best_runs]):.5f}",
         f"{np.mean([r['breakdown'][dimension][label]['mse'] for r in best_runs]) - np.mean([r['breakdown'][dimension][label]['mse'] for r in ref_runs]):+.5f}"]
        for dimension, groups in ref_runs[0]['breakdown'].items() for label in groups])
    html += '<h2>Every seed and paired uncertainty</h2>'
    rows = []
    for r in result['runs']:
        ci = r.get('bootstrap', {}).get('paired_supervised_minus_baseline', {}).get('reference', {}).get('mse')
        rows.append([r['name'],r['seed'],f"{r['train']['mse']:.5f}",f"{r['validation']['mse']:.5f}",
                     'reference' if ci is None else '[' + ', '.join(f'{x:.5f}' for x in ci) + ']',
                     f"{r['selected_epoch']}/{r['epochs']}",r['optimizer_steps'],f"{r['training_seconds']:.2f}",f"{r['wall_seconds']:.2f}"])
    html += table(['Treatment','Seed','Frozen train MSE','Validation MSE','95% CI: treatment − reference MSE',
                   'Best/total epochs','Updates','Epoch work (s)','Fit + diagnostics (s)'],rows)
    html += (f'<p>Intervals use {config["bootstrap_resamples"]:,} paired source-game resamples per seed, conditional on the selected '
             'models. They are separate from seed standard deviation and do not correct for repeated '
             'validation selection or multiple comparisons.</p><img src="curves.png" alt="Validation and learning-rate curves">')
    html += '<h2>Subgroup errors</h2><p>Groups below 100 positions are sparse. These are exploratory, overlapping groups.</p>'
    for dimension in result['runs'][0]['breakdown']:
        html += '<h3>' + escape(dimension.replace('_',' ')) + '</h3>'
        html += table(['Treatment','Seed','Group','Positions/games','MSE','MAE','Signed error'], [
            [r['name'],r['seed'],label+(' (sparse)' if g['sparse'] else ''),f"{g['positions']}/{g['games']}",
             f"{g['mse']:.5f}",f"{g['mae']:.5f}",f"{g['signed_error']:.5f}"]
            for r in result['runs'] for label,g in r['breakdown'][dimension].items()])
    html += ('<h2>Compute, limitations and next step</h2>'
             f"<p>Epoch work: {result['training_seconds']:.1f}s. Completed invocation wall time: "
             f"{result['wall_seconds']:.1f}s. Paid compute: C$0. Report generation is excluded. "
             'Inference is the mean across seeds of median single-position latency over three passes '
             'through the first 64 validation boards, after warmup, with one BLAS thread; it includes encoding.</p>'
             '<p>Early stopping causes unequal update counts; feature v3 also adds input weights. This is '
             'not a fixed-compute comparison. Validation has been used repeatedly; no final-test targets '
             'were evaluated. Negative results remain in the report. Before replacing a public model, '
             'reserve fresh confirmatory data and evaluate controlled chess matches. Lower teacher-target '
             'MSE alone does not establish stronger play.</p><h2>Evidence</h2><ul>')
    html += ''.join(f'<li><a href="evidence/{p.name}">{escape(p.stem)}</a></li>' for p in sorted(evidence.glob('*.json')))
    (output / 'index.html').write_text(html + '</ul></main></html>')
    return output / 'index.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    print(generate(args.source, args.output))


if __name__ == '__main__':
    main()
