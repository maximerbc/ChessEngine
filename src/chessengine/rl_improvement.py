"""Budgeted v3 self-play study; pilot gates precede main training and fresh final matches."""
import argparse
from pathlib import Path
import random
import chess
import chess.pgn
import numpy as np
from threadpoolctl import threadpool_limits
from .common import digest, json_hash, read_json, write_json, provenance
from .compute_budget import ComputeBudget, BudgetExceeded
from .data import open_pgn, records
from .features import restore, record_board, position_key, encode
from .model import NumpyEvaluator
from .rl import play_game, selfplay
from .evaluate import freeze_protocol, run_matches, select_checkpoint, paired_summary
from .fixed_data import predict
from .diagnostics import metrics


def canonical(board):
    return position_key(board if board.turn else board.mirror())


def prepare_openings(config, root, budget):
    path = root/'openings.json'
    inputs = {key: digest(config[key]) for key in ['old_openings', 'corpus', 'archive']}
    manifest_path = root/'openings-manifest.json'
    if path.exists():
        manifest = read_json(manifest_path)
        if manifest['inputs'] != inputs or manifest['sha256'] != digest(path):
            raise ValueError('Fresh opening inputs changed')
        return path
    old = read_json(config['old_openings'])
    excluded_games, excluded_positions = set(), set()
    # Only position and provenance fields are consumed, never corpus teacher targets.
    for row in records(config['corpus']):
        excluded_games.add(row['game_id'])
        excluded_positions.add(canonical(chess.Board(row['fen'])))
    for group in old.values():
        for row in group:
            excluded_games.add(row['source_game_id'])
            excluded_positions.add(canonical(restore(row)))
    fresh, scanned = [], 0
    with open_pgn(config['archive']) as stream:
        while len(fresh) < config['final_pairs']:
            budget.check()
            game = chess.pgn.read_game(stream)
            if game is None:
                raise ValueError('Insufficient fresh final openings; never reuse old final positions')
            scanned += 1
            moves = list(game.mainline_moves())
            if game.errors or len(moves) < 16 or game.headers.get('Variant','Standard') != 'Standard':
                continue
            board = game.board()
            if not board.is_valid() or board.chess960:
                continue
            game_id = json_hash({'initial': board.fen(), 'moves': [m.uci() for m in moves]})
            if game_id in excluded_games:
                continue
            for move in moves[:12]:
                board.push(move)
            key = canonical(board)
            if key in excluded_positions or board.is_game_over(claim_draw=True):
                continue
            fresh.append({**record_board(board), 'id': game_id[:12], 'source_game_id': game_id})
            excluded_games.add(game_id)
            excluded_positions.add(key)
    write_json(path, {'train': old['train'], 'development': old['development'], 'test': fresh})
    write_json(manifest_path, dict(inputs=inputs, sha256=digest(path), fresh_pairs=len(fresh),
                                   scanned_games=scanned, canonical_overlaps=0,
                                   note='12-ply starts; no test labels consumed; old test positions excluded'))
    budget.checkpoint()
    return path


def choose_protocol(pilots, config, elapsed):
    rejected = []
    feasible = []
    for p in pilots:
        depth_fraction = p['depth_one_moves']/p['searched_moves'] if p['searched_moves'] else 0.
        truncation = p['truncated']/p['games']
        if depth_fraction < .90 or truncation > .20:
            rejected.append(dict(nodes=p['nodes'], depth_one_fraction=depth_fraction, truncation=truncation))
        else:
            feasible.append(p)
    # Target game count takes precedence over a larger node budget.
    for games in config['game_targets']:
        for p in sorted(feasible, key=lambda p: p['nodes'], reverse=True):
            batches = 10 if games == 1000 else 5
            batch_size = games//batches
            seeds = len(config['seeds'])
            generation = seeds*games*p['mean_game_seconds']
            matches = seeds*(4*config['development_pairs']+2*config['final_pairs'])*p['mean_game_seconds']/0.9
            equal_time = seeds*2*config['development_pairs']*p['p90_plies']*config['equal_time_ms']/1000
            # Conservative update allowance plus 50% throughput margin, including evaluation.
            forecast = elapsed+1.5*(generation+matches+equal_time+seeds*batches*20)
            if forecast <= config['forecast_seconds']:
                return dict(approved=True, nodes=p['nodes'], games_per_seed=games, batches=batches,
                            games_per_batch=batch_size, development_batches=[(batches+1)//2,batches],
                            forecast_seconds=forecast, rejected_pilots=rejected)
    return dict(approved=False, reason='No pilot satisfies both quality and compute gates', rejected_pilots=rejected)


def aggregate_finals(game_sets, repetitions=10000, seed=2026):
    by_seed=[]
    for games in game_sets:
        paired_summary(games)  # Validate complete swapped-color pairs.
        values={}
        for g in games:
            values.setdefault(g['opening_id'],[]).append(g['score'])
        by_seed.append({key:np.mean(v) for key,v in values.items()})
    keys=sorted(by_seed[0])
    if any(set(row)!=set(keys) for row in by_seed):
        raise ValueError('Seeds must share final opening pairs')
    matrix=np.array([[row[key] for key in keys] for row in by_seed])
    means=matrix.mean(axis=0)
    rng=np.random.default_rng(seed)
    draws=[float(rng.choice(means,len(means),replace=True).mean()) for _ in range(repetitions)]
    ci=np.quantile(draws,[.025,.975]).tolist()
    uncapped={g['opening_id'] for games in game_sets for g in games if g['capped']}
    keep=[i for i,key in enumerate(keys) if key not in uncapped]
    capped_count=sum(g['capped'] for games in game_sets for g in games)
    total=sum(len(games) for games in game_sets)
    score=float(means.mean())
    return dict(score=score, ci95=ci, seed_scores=matrix.mean(axis=1).tolist(),
                seed_sd=float(matrix.mean(axis=1).std(ddof=1)), pairs=len(keys), games=total,
                capped_games=capped_count, uncapped_pairs=len(keep),
                uncapped_score=float(matrix[:,keep].mean()) if keep else None,
                cap_score_bounds=[score-.5*capped_count/total,score+.5*capped_count/total],
                conclusion='improvement' if ci[0]>.5 else 'regression' if ci[1]<.5 else 'inconclusive',
                note='Opening-pair uncertainty conditional on these three trained seeds; not a population-of-seeds interval')


def run(config, stage='all', unlimited_local=False):
    root=Path(config['output'])
    root.mkdir(parents=True,exist_ok=True)
    averaging=read_json(config['averaging_results'])
    baseline=averaging['rl_initial_checkpoint']
    if digest(baseline)!=averaging['rl_initial_sha256']:
        raise ValueError('Selected supervised checkpoint changed')
    if NumpyEvaluator(baseline).feature_version != 3:
        raise ValueError('This study requires the approved v3 initialization')
    identity=dict(config=config,baseline_sha256=digest(baseline),
                  averaging_sha256=digest(config['averaging_results']),
                  sources={k:digest(config[k]) for k in ['old_openings','corpus','development_data','archive']},
                  code={n:digest(Path(__file__).with_name(n+'.py')) for n in
                        ['rl_improvement','compute_budget','rl','evaluate','engine','features','model','train','common']})
    snapshot=root/'study-protocol.json'
    if snapshot.exists():
        if read_json(snapshot)['identity']!=identity:
            raise ValueError('Frozen study mismatch')
    else:
        write_json(snapshot,dict(identity=identity,provenance=provenance()))
    try:
        with ComputeBudget(root/'compute.json',config['allowance_seconds'], enforce=not unlimited_local) as budget, threadpool_limits(limits=1):
            budget.check()
            openings=read_json(config['old_openings'])['train']
            actor=NumpyEvaluator(baseline)
            pilots=[]
            for nodes in config['nodes']:
                path=root/f'pilot-{nodes}.json'
                games=read_json(path)['games'] if path.exists() else []
                for i in range(len(games),config['pilot_games']):
                    rng=random.Random(config['pilot_seed']+i)
                    opening=openings[rng.randrange(len(openings))]
                    cfg={**config['training'],'max_plies':config['max_plies'],'nodes':nodes}
                    game,_=play_game(actor,opening,cfg,rng,budget)
                    games.append(game)
                    write_json(path,dict(nodes=nodes,checkpoint_sha256=digest(baseline),games=games))
                    budget.checkpoint()
                    print(f"Pilot {nodes}: {len(games)}/{config['pilot_games']}",flush=True)
                pilots.append(dict(nodes=nodes,games=len(games),truncated=sum(g['truncated'] for g in games),
                                   searched_moves=sum(g['searched_moves'] for g in games),
                                   depth_one_moves=sum(g['depth_one_moves'] for g in games),
                                   mean_game_seconds=float(np.mean([g['seconds'] for g in games])),
                                   p90_plies=float(np.quantile([g['plies'] for g in games],.9))))
            write_json(root/'pilot-summary.json',pilots)
            decision_path=root/'decision.json'
            if decision_path.exists():
                decision=read_json(decision_path)
            else:
                # Fresh-opening preparation is part of the allowance and precedes the final forecast.
                if any(p['searched_moves'] and p['depth_one_moves']/p['searched_moves']>=.9
                       and p['truncated']/p['games']<=.2 for p in pilots):
                    prepare_openings(config,root,budget)
                budget.checkpoint()
                decision=choose_protocol(pilots,config,config['allowance_seconds']-budget.remaining)
                write_json(decision_path,decision)
            sealed = dict(decision_sha256=digest(decision_path),
                          pilot_sha256=digest(root/'pilot-summary.json'),
                          openings_sha256=digest(root/'openings.json') if (root/'openings.json').exists() else None)
            seal_path=root/'decision-seal.json'
            if seal_path.exists() and read_json(seal_path)!=sealed:
                raise ValueError('Frozen pilot decision changed')
            if not seal_path.exists():
                write_json(seal_path,sealed)
            print(decision,flush=True)
            if not decision['approved'] or stage=='pilot':
                write_json(root/'status.json',dict(status='pilot_complete' if decision['approved'] else 'pilot_rejected',
                                                  decision=decision))
                return decision
            opening_path=prepare_openings(config,root,budget)
            batches=decision['batches']
            for seed in config['seeds']:
                directory=root/f'seed-{seed}'/'training'
                cfg={**config['training'],'seed':seed,'max_plies':config['max_plies'],
                     'nodes':decision['nodes'],'batches':batches,'games_per_batch':decision['games_per_batch']}
                print(f'Self-play seed {seed}: {decision["games_per_seed"]} games',flush=True)
                selfplay(baseline,opening_path,directory,cfg,resume=True,budget=budget)
            eval_cfg=dict(seed=2026,pairs=config['development_pairs'],nodes=decision['nodes'],qdepth=2,
                          time_ms=None,max_plies=config['max_plies'],evidence='rl-improvement-study')
            for seed in config['seeds']:
                dirs=[]
                for batch in decision['development_batches']:
                    checkpoint=str(root/f'seed-{seed}/training/batch-{batch:03}.npz')
                    protocol=root/f'seed-{seed}/development-{batch}.protocol.json'
                    directory=root/f'seed-{seed}/development-{batch}'
                    if not protocol.exists():
                        freeze_protocol(checkpoint,baseline,str(opening_path),eval_cfg,protocol,'development')
                    run_matches(protocol,directory,budget)
                    dirs.append(directory)
                select_checkpoint(dirs,root/f'seed-{seed}/selection.json')
            # All checkpoint selections are now locked, before any fresh final result is inspected.
            selections={str(s):digest(root/f'seed-{s}/selection.json') for s in config['seeds']}
            lock=root/'selection-lock.json'
            if lock.exists() and read_json(lock)!=selections:
                raise ValueError('Final checkpoint selections changed')
            if not lock.exists():
                write_json(lock,selections)
            results,game_sets=[],[]
            for seed in config['seeds']:
                selected=read_json(root/f'seed-{seed}/selection.json')
                record=dict(seed=seed,selection=selected)
                for name,split,cfg in [
                    ('final','test',{**eval_cfg,'pairs':config['final_pairs']}),
                    ('equal-time','development',{**eval_cfg,'nodes':1000000000,'time_ms':config['equal_time_ms']})]:
                    protocol=root/f'seed-{seed}/{name}.protocol.json'
                    directory=root/f'seed-{seed}/{name}'
                    if not protocol.exists():
                        freeze_protocol(selected['checkpoint'],baseline,str(opening_path),cfg,protocol,split)
                    record[name]=run_matches(protocol,directory,budget)
                    if name=='final':
                        game_sets.append(read_json(directory/'matches.json')['games'])
                record['training']=read_json(root/f'seed-{seed}/training/metrics.json')
                results.append(record)
            # Secondary teacher agreement is computed after selection and never enters it.
            budget.check()
            val=[r for r in records(config['development_data']) if r['split']=='validation']
            xv=np.stack([encode(restore(r),3) for r in val])
            yv=np.array([r['target'] for r in val])
            teacher={'baseline':metrics(predict(NumpyEvaluator(baseline),xv),yv)}
            for r in results:
                teacher[str(r['seed'])]=metrics(predict(NumpyEvaluator(r['selection']['checkpoint']),xv),yv)
            budget.checkpoint()
            result=dict(protocol=read_json(snapshot),decision=decision,pilots=pilots,seeds=results,
                        aggregate=aggregate_finals(game_sets,config['bootstrap_resamples'],config['bootstrap_seed']),
                        teacher_agreement=teacher,compute_seconds=config['allowance_seconds']-budget.remaining,
                        paid_cost_cad=0)
            write_json(root/'results.json',result)
            write_json(root/'status.json',dict(status='complete'))
            return result
    except BudgetExceeded as error:
        write_json(root/'status.json',dict(status='budget_exhausted',reason=str(error)))
        print(str(error),flush=True)
        return dict(status='budget_exhausted')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('--stage',choices=['all','pilot'],default='all')
    p.add_argument('--unlimited-local', action='store_true',
                   help='Explicitly authorized local run without a time cap; retain compute accounting')
    a=p.parse_args()
    run(read_json(a.config),a.stage,unlimited_local=a.unlimited_local)


if __name__=='__main__':
    main()
