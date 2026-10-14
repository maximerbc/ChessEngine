"""Render pilot gates and, only when completed, RL outcome evidence."""
import argparse
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import read_json, write_json
from .diagnostics_report import table


def generate(source, output):
    source,output=Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for name in ['study-protocol','pilot-summary','decision','compute','status','openings-manifest','results','compute-amendment','study-protocol-original','training-audit','final-audit']:
        path=source/f'{name}.json'
        if path.exists():
            shutil.copyfile(path,output/path.name)
    pilots=read_json(source/'pilot-summary.json')
    decision=read_json(source/'decision.json')
    status=read_json(source/'status.json')
    html='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Stronger self-play study</title>'
    html+='<style>body{font:16px/1.6 system-ui;background:#f5f6f2;color:#23382f;max-width:1100px;margin:auto;padding:28px}img{max-width:100%}.table-wrap{overflow:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}</style>'
    html+='<h1>Stronger self-play RL: quality before scale</h1>'
    html+=f"<p>Status: <b>{status['status']}</b>. Paid compute C$0. Released demo checkpoints are unchanged.</p>"
    html+='<h2>Pilot</h2><p>Twenty games per search budget, using the same preselected supervised seed-42 initialization. Require at least 90% of searched moves to finish depth one and no more than 20% truncated games. Exploration moves are excluded from the search-depth denominator. Pilot games never train the main models.</p>'
    html+=table(['Nodes/move','Games','Depth-one completion','Truncation','Mean game seconds','90th percentile plies'],
                [[p['nodes'],p['games'],f"{p['depth_one_moves']/p['searched_moves']:.2%}" if p['searched_moves'] else 'n/a',
                  f"{p['truncated']/p['games']:.1%}",f"{p['mean_game_seconds']:.2f}",p['p90_plies']] for p in pilots])
    if not decision['approved']:
        html+='<p><b>No approved main study.</b> The predefined quality/compute gates rejected the pilot options. No main RL training or fresh final matches were run. The next step is a separately approved investigation of search completion and game truncation, not relaxation of the thresholds or an unreported larger budget.</p>'
    else:
        html+=f"<p>Frozen protocol: {decision['nodes']} nodes/move; {decision['games_per_seed']} games per seed; forecast {decision['forecast_seconds']/3600:.2f} hours including safety margin, within the nine-hour planning allowance. The original planning protocol specified a persisted 12-hour allowance; any authorized amendment is recorded below.</p>"
    if (source/'compute-amendment.json').exists():
        html+='<p><b>Authorized amendment:</b> on 13 October the user removed the local CPU time cap. The original 12-hour limit above is historical; scientific settings and elapsed-time accounting remain unchanged.</p>'
    if (source/'training-audit.json').exists():
        audit=read_json(source/'training-audit.json')
        html+='<h2>Completed training audit</h2><p>All recorded games were replayed to verify legal histories and terminal outcomes. Actor exports were reconstructed from preceding weights and original actor metadata to verify each frozen batch.</p>'
        html+=table(['Seed','Games','Completed','Truncated'],[[x['seed'],x['games'],x['completed'],x['truncated']] for x in audit['seeds']])
    if status['status']=='complete':
        r=read_json(source/'results.json')
        a=r['aggregate']
        html+=f"<h2>Final playing strength</h2><p>Mean score across the three fixed RL seeds: <b>{a['score']:.2%}</b>, opening-pair 95% interval [{a['ci95'][0]:.2%}, {a['ci95'][1]:.2%}]. Conclusion: <b>{a['conclusion']}</b>. Seed standard deviation {a['seed_sd']:.2%}, reported separately from match uncertainty.</p>"
        html+=table(['Seed','Selected batch path','Development score','Final score','95% interval','Wins/draws/losses','Equal-time score'],[
            [x['seed'],Path(x['selection']['checkpoint']).name,f"{x['selection']['development_score']:.2%}",
             f"{x['final']['score']:.2%}",str(x['final']['ci95']),
             f"{x['final']['wins']}/{x['final']['draws']}/{x['final']['losses']}",f"{x['equal-time']['score']:.2%}"]
            for x in r['seeds']])
        curves, search_stats = [], []
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for x in r['seeds']:
            h=x['training']['history']
            axes[0].plot([t['games_total'] for t in h],[t['loss'] for t in h],label=str(x['seed']))
            scores=[]
            for batch in decision['development_batches']:
                summary=read_json(source/f"seed-{x['seed']}/development-{batch}/summary.json")
                scores.append(summary['score'])
                curves.append(dict(seed=x['seed'],batch=batch,**summary))
            for phase in ['selfplay','final','equal-time']:
                if phase=='selfplay':
                    games=[g for path in sorted((source/f"seed-{x['seed']}/training").glob('games-*.json'))
                           for g in read_json(path)]
                else:
                    games=read_json(source/f"seed-{x['seed']}/{phase}/matches.json")['games']
                count=sum(g['searched_moves'] for g in games)
                search_stats.append(dict(seed=x['seed'],phase=phase,searched_moves=count,
                                         depth_one_fraction=sum(g['depth_one_moves'] for g in games)/count if count else None,
                                         mean_completed_depth=sum(g['depth_sum'] for g in games)/count if count else None,
                                         nodes=sum(g['nodes'] for g in games)))
            axes[1].plot([b*decision['games_per_batch'] for b in decision['development_batches']],scores,marker='o',label=str(x['seed']))
        axes[0].set(xlabel='Self-play games',ylabel='Outcome training MSE',title='Changing self-play distribution')
        axes[1].set(xlabel='Self-play games',ylabel='Development match score',title='Checkpoint selection evidence')
        axes[1].axhline(.5,color='gray',linestyle='--')
        for ax in axes:
            ax.legend(title='RL seed')
        fig.savefig(output/'learning.png',dpi=160)
        plt.close(fig)
        write_json(output/'development-curves.json',curves)
        write_json(output/'search-statistics.json',search_stats)
        html+='<img src="learning.png" alt="RL training loss and development playing scores">'
        html+=f"<p>Final move-cap adjudications: {a['capped_games']}/{a['games']}; uncapped opening pairs across all seeds: {a['uncapped_pairs']}. Uncapped score: {a['uncapped_score']}. Assigning all capped games losses or wins gives score bounds {a['cap_score_bounds']}.</p>"
        html+='<h2>Teacher agreement is a separate diagnostic</h2>'
        html+=table(['Model','Validation Stockfish MSE','MAE'],[[name,f"{v['mse']:.6f}",f"{v['mae']:.6f}"] for name,v in r['teacher_agreement'].items()])
        html+='<p>These teacher errors are measured after selection and cannot choose RL checkpoints. Outcome training MSE uses actual self-play returns and is not comparable to teacher-target MSE. A lower outcome loss can reflect the changing game distribution rather than stronger play.</p>'
        if a['conclusion']=='regression':
            html+='<h2>Interpretation and next experiment</h2><p>The aggregate interval is entirely below 50%: this outcome-only RL procedure weakened play against its supervised initialization under the tested conditions. Keep the released model unchanged. Teacher-target MSE also increased, consistent with loss of teacher agreement; this does not establish why playing strength fell.</p><p>Possible explanations include noisy terminal targets from shallow self-play, changes in the visited positions, and updates that overwrite useful supervised estimates. These are hypotheses, not isolated causal findings. A separately preregistered follow-up could compare a smaller RL update rate or an explicit constraint toward the initial model, one change at a time. That would be a new treatment, not a repair to these results. This final set is now used evidence; reserve new confirmation data before a later confirmatory claim.</p>'
        html+='<h2>Search work</h2>'
        html+=table(['Seed','Phase','Searched moves','Depth-one fraction','Mean completed depth','Nodes'],
                    [[x['seed'],x['phase'],x['searched_moves'],x['depth_one_fraction'],x['mean_completed_depth'],x['nodes']]
                     for x in search_stats])
        html+='<h2>Generation and update costs</h2>'
        html+=table(['Seed','Generated','Completed','Truncated','Generation seconds','Update seconds'],[
            [x['seed'],x['training']['history'][-1]['games_total'],
             sum(h['completed'] for h in x['training']['history']),sum(h['truncated'] for h in x['training']['history']),
             f"{sum(h['generation_seconds'] for h in x['training']['history']):.2f}",
             f"{sum(h['update_seconds'] for h in x['training']['history']):.2f}"] for x in r['seeds']])
    if (source/'final-audit.json').exists():
        audit=read_json(source/'final-audit.json')
        html+=f"<p>Independent audit: all 1,920 evaluation PGNs, color pairs, outcome scores, checkpoint identities and aggregate bootstrap verified. Measured saved-game and training-update work: {audit['measured_game_and_update_seconds']/3600:.2f} hours; excludes preparation, report overhead and interrupted unsaved games.</p>"
    budget=read_json(source/'compute.json')
    html+=f"<h2>Limitations and reproducibility</h2><p>Recorded study ledger time (including conservatively charged interruption downtime) {budget['consumed']/3600:.3f} hours. Fresh final positions use unused games from the same historical archive, not a new era or population. Repeated development selection, only three training seeds and finite search budgets restrict conclusions. Aggregate bootstrap uncertainty is conditional on these fixed seeds. No public Elo or general chess-strength claim is made.</p>"
    html+='<p>Training/evaluation inputs and code hashes are frozen in the protocol. Graceful interruption saves completed games; an unclosed crash interval is charged conservatively to the allowance. See <a href="../../rl-improvement.md">methods and commands</a>.</p><ul>'
    html+=''.join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in sorted(output.glob('*.json')))
    (output/'index.html').write_text(html+'</ul></html>')
    return output/'index.html'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('output')
    a=p.parse_args()
    print(generate(a.source,a.output))
