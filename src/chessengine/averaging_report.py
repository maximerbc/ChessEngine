"""Compact report for the predeclared averaging comparison."""
import argparse
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import read_json
from .diagnostics_report import table


def generate(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = read_json(source/'results.json')
    shutil.copyfile(source/'results.json', output/'results.json')
    shutil.copyfile(source/'protocol.json', output/'protocol.json')
    entries = result['entries']
    fig, ax = plt.subplots(figsize=(9,4),layout='constrained')
    ax.barh([e['name'] for e in entries],[e['mse'] for e in entries],color='#24796a')
    ax.set(xlabel='Validation MSE',title='Averaging on fixed development data')
    ax.invert_yaxis()
    fig.savefig(output/'comparison.png',dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,e in zip(axes,[e for e in entries if e['name'].startswith('average')]):
        h=e['history']
        ax.plot([v['epoch'] for v in h],[v['validation_loss'] for v in h],label='Ordinary')
        avg=[v for v in h if 'average_validation_loss' in v]
        ax.plot([v['epoch'] for v in avg],[v['average_validation_loss'] for v in avg],label='Weight average')
        ax.set(title=e['name'],xlabel='Epoch',ylabel='Validation MSE')
        ax.legend()
    fig.savefig(output/'curves.png',dpi=160)
    plt.close(fig)
    html='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Supervised averaging</title>'
    html+='<style>body{font:16px/1.6 system-ui;background:#f5f6f2;color:#23382f;max-width:1100px;margin:auto;padding:28px}img{max-width:100%}.table-wrap{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}</style>'
    html+='<h1>Supervised averaging: bounded development study</h1>'
    html+=f"<p>Reference mean MSE {result['reference_mean_mse']:.6f}; weight-average mean {result['averaged_mean_mse']:.6f}. Reduction {result['relative_reduction']:.2%}; improving seeds {result['improving_seeds']}/3. Adopt weight averaging: {result['adopt_averaging']}.</p>"
    html+='<p>Same positions, labels, architecture and optimizer. Ordinary-model losses and final/best weights reproduced within the frozen tolerance. Average snapshots beginning after epoch 8; ordinary validation controls early stopping. The ensemble is an average of predictions, not weights. Its error cannot exceed mean member error on the same records, but need not beat every member.</p>'
    html+=table(['Predictor','MSE','MAE','Signed error','Inference µs'],[[e['name'],f"{e['mse']:.6f}",f"{e['mae']:.6f}",f"{e['signed_error']:.6f}",f"{e['latency_us']:.1f}"] for e in entries])
    html+='<img src="comparison.png" alt="Validation error by predictor"><img src="curves.png" alt="Ordinary and averaged checkpoint learning curves">'
    html+='<h2>Paired source-game uncertainty</h2>'
    pairs=[]
    for e in entries:
        for baseline,ci in e.get('bootstrap',{}).get('paired_supervised_minus_baseline',{}).items():
            pairs.append([e['name'],baseline,str(ci['mse']),str(ci['mae'])])
    ci=entries[-1]['versus_mean_member_ci95']
    pairs.append(['ensemble','mean member loss',str(ci['mse']),str(ci['mae'])])
    html+=table(['Predictor','Comparator','95% MSE difference interval','95% MAE difference interval'],pairs)
    html+='<p>2,000 resamples of source games; predictions compared on the same draws. Intervals condition on these selected models and do not correct repeated-validation selection. Differences below zero favor averaging. Seed-level entries remain separate; the ensemble is one predictor.</p>'
    html+='<h2>Subgroups</h2>'
    for dimension in entries[0]['breakdown']:
        html+=f'<h3>{dimension}</h3>'
        html+=table(['Predictor','Group','Positions/games','MSE','MAE'],[[e['name'],label+(' (sparse)' if g['sparse'] else ''),f"{g['positions']}/{g['games']}",f"{g['mse']:.6f}",f"{g['mae']:.6f}"] for e in entries for label,g in e['breakdown'][dimension].items()])
    html+=f"<h2>Handoff to RL</h2><p>Common initialization: {result['rl_initial_checkpoint']}. This is seed 42 chosen before results, not the best seed. The ensemble will not initialize RL. Released models remain unchanged. Invocation time {result['wall_seconds']:.1f}s; paid compute C$0. Latency includes board encoding, three warmed passes over up to 64 validation boards and one BLAS thread.</p>"
    html+='<p>No stronger-play claim follows from these errors. Proceed to the approved RL pilot regardless of this outcome. Raw predictions and experimental checkpoints remain ignored; <a href="results.json">compact evidence</a> and <a href="protocol.json">frozen protocol</a> are available.</p></html>'
    (output/'index.html').write_text(html)
    return output/'index.html'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('output')
    a=p.parse_args()
    print(generate(a.source,a.output))
