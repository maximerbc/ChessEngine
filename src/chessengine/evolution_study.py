"""Frozen, teacher-free CMA-ES study with resumable paired tournaments."""
import argparse
import importlib.metadata
import pickle
from pathlib import Path
import random
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .common import digest, json_hash, provenance, read_json, write_json
from .engine import Search
from .features import restore
from .evolution_data import prepare
from .evolution_games import audit_ledger, evaluator, run_tasks, tasks_for
from .evolution_model import EvolutionEvaluator, PARAMETER_COUNT, export
from .evaluate import paired_summary

SCIENCE = ['evolution_study', 'evolution_data', 'evolution_model', 'evolution_games',
           'engine', 'features', 'model', 'common', 'evaluate']


def freeze(config):
    if not 1 <= config['workers'] <= 2:
        raise ValueError('At most two tournament workers are allowed')
    packages = {p: importlib.metadata.version(p) for p in ['cma', 'numpy', 'chess']}
    if packages['cma'] != '4.5.0':
        raise ValueError('This experiment requires cma==4.5.0')
    inputs = [config['archive'], config['corpus'], config['supervised'], *config['reserved']]
    identity = dict(config=config, packages=packages, inputs={p: digest(p) for p in inputs},
                    sources={n: digest(Path(__file__).with_name(n+'.py')) for n in SCIENCE})
    path = Path(config['output'])/'protocol.json'
    if path.exists() and read_json(path)['identity'] != identity:
        raise ValueError('Frozen evolutionary protocol changed; never silently amend or restart')
    if not path.exists():
        write_json(path, dict(identity=identity, provenance=provenance(),
                             representative_seed=config['representative_seed'],
                             note='All evaluation weights random; no teacher targets or fixed material score.'))


def checkpoint(directory, generation):
    return str(Path(directory)/f'mean-{generation:03}.npz')


def initialize(directory, seed, sd):
    path = checkpoint(directory, 0)
    parameters = np.random.default_rng(seed).normal(0, sd, PARAMETER_COUNT)
    if not Path(path).exists():
        export(parameters, path, dict(seed=seed, generation=0))
    elif not np.array_equal(EvolutionEvaluator(path).parameters, parameters.astype(np.float32)):
        raise ValueError('Random initializer changed')
    return parameters


def generation_starts(pool, generation, seed):
    rng = random.Random(seed*10000+generation)
    groups = {k: [r for r in pool if r['stratum'] == k] for k in ('endgame', 'middlegame', 'opening')}
    allocation = {'endgame': 6} if generation <= 10 else (
        {'endgame': 3, 'middlegame': 3} if generation <= 20 else {'middlegame': 3, 'opening': 3})
    result = [r for kind, n in allocation.items() for r in rng.sample(groups[kind], n)]
    rng.shuffle(result)
    return result


def fitness(games, population):
    weights = {'classical': .5, 'supervised': .25, 'archive': .25}
    scores = []
    for index in range(population):
        by_group = {k: [r for r in games if r['candidate_index'] == index and r['opponent_group'] == k]
                    for k in weights}
        if any(len(rows) != 4 for rows in by_group.values()):
            raise ValueError('Incomplete candidate fitness tournament')
        for rows in by_group.values():
            paired_summary(rows, repetitions=10)
        scores.append(sum(weights[k]*np.mean([r['score'] for r in rows]) for k, rows in by_group.items()))
    return np.asarray(scores)


def save_state(path, state):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(path)


def evolve(config, pool, seed, generations, nodes, directory):
    import cma  # Training only. Never imported by inference or API startup.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    initial = initialize(directory, seed, config['initial_sd'])
    signature = json_hash(dict(protocol=digest(Path(config['output'])/'protocol.json'),
                               pool=pool, seed=seed, generations=generations, nodes=nodes))
    state_path = directory/'optimizer.pkl'
    if state_path.exists():
        # Only our private local state is unpickled, never public/user checkpoints.
        with state_path.open('rb') as stream:
            state = pickle.load(stream)
        if state['signature'] != signature:
            raise ValueError('Evolution optimizer resume identity mismatch')
        np.random.set_state(state['rng'])
    else:
        es = cma.CMAEvolutionStrategy(initial, config['sigma'], dict(seed=seed, popsize=config['population'],
                                     bounds=config['bounds'], verbose=-9, verb_log=0))
        state = dict(signature=signature, es=es, rng=np.random.get_state(), history=[], pending=None)
        save_state(state_path, state)
    while len(state['history']) < generations:
        generation = len(state['history'])+1
        folder = directory/f'generation-{generation:03}'
        folder.mkdir(exist_ok=True)
        if state['pending'] is None:
            state['pending'] = state['es'].ask()
            state['rng'] = np.random.get_state()
            save_state(state_path, state)
        candidates = state['pending']
        starts = generation_starts(pool, generation, seed)
        archive = [checkpoint(directory, g) for g in range(generation) if g == 0 or g % 5 == 0]
        ancestor = random.Random(seed*10000+generation).choice(archive)
        match_config = dict(nodes=nodes, qdepth=config['qdepth'], max_plies=config['max_plies'])
        tasks = []
        for index, parameters in enumerate(candidates):
            candidate = str(folder/f'candidate-{index:02}.npz')
            if not Path(candidate).exists():
                export(parameters, candidate, dict(seed=seed, generation=generation, candidate=index))
            elif not np.array_equal(EvolutionEvaluator(candidate).parameters, np.asarray(parameters, dtype=np.float32)):
                raise ValueError('Pending evolutionary candidate changed')
            for group, opponent, openings in [('classical', 'handcrafted', starts[:2]),
                    ('supervised', config['supervised'], starts[2:4]), ('archive', ancestor, starts[4:])]:
                tasks += tasks_for(candidate, opponent, openings, match_config,
                                   prefix=f'{index}:{group}:', candidate_index=index, opponent_group=group)
        ledger = run_tasks(tasks, folder/'matches.json', config['workers'])
        scores = fitness(ledger['games'], config['population'])
        state['es'].tell(candidates, -scores)
        mean = np.asarray(state['es'].result.xfavorite)
        if not np.isfinite(mean).all():
            raise ValueError('Non-finite CMA-ES mean; release blocked')
        export(mean, checkpoint(directory, generation), dict(seed=seed, generation=generation))
        games = ledger['games']
        state['history'].append(dict(generation=generation, fitness=scores.tolist(),
                                    best_candidate=int(np.argmax(scores)), mean_fitness=float(scores.mean()),
                                    parameter_diversity=float(np.std(candidates, axis=0).mean()),
                                    sigma=float(state['es'].sigma), games=len(games),
                                    capped=sum(g['capped'] for g in games),
                                    decisive=sum(g['result'] in ('1-0', '0-1') for g in games),
                                    active_seconds=ledger['active_seconds'],
                                    peak_combined_bytes=ledger['peak_combined_bytes'],
                                    mean_sha256=digest(checkpoint(directory, generation)), ancestor_sha256=digest(ancestor)))
        state['pending'] = None
        state['rng'] = np.random.get_state()
        save_state(state_path, state)
        write_json(directory/'history.json', state['history'])
        print(f'Evolution seed {seed}: generation {generation}/{generations}; mean fitness {scores.mean():.3f}', flush=True)
    # Repair a crash between the authoritative optimizer commit and JSON reporting.
    write_json(directory/'history.json', state['history'])
    return state['history']


def comparison(config, candidate, opponent, openings, path, nodes, time_ms=None):
    settings = dict(nodes=nodes, time_ms=time_ms, qdepth=config['qdepth'], max_plies=config['max_plies'])
    ledger = run_tasks(tasks_for(candidate, opponent, openings, settings), path, config['workers'])
    audit_ledger(ledger)
    return {**paired_summary(ledger['games'], seed=config['bootstrap_seed'],
                             repetitions=config['bootstrap_resamples']),
            'candidate_sha256': digest(candidate), 'opponent_sha256': 'handcrafted-v1' if opponent == 'handcrafted' else digest(opponent),
            'peak_combined_bytes': ledger.get('peak_combined_bytes', 0), 'active_seconds': ledger.get('active_seconds', 0)}


def benchmark(config, groups):
    root = Path(config['output'])
    path = root/'benchmark.json'
    if path.exists():
        return read_json(path)
    initial = initialize(root/'pilot', config['pilot_seed'], config['initial_sd'])
    models = [('classical', evaluator('handcrafted')), ('supervised', evaluator(config['supervised'])),
              ('random', EvolutionEvaluator(parameters=initial))]
    starts = [r for kind in ('endgame', 'middlegame', 'opening')
              for r in [x for x in groups['pilot'] if x['stratum'] == kind][:4]]
    budgets = []
    for nodes in config['nodes']:
        rows = []
        for start in starts:
            for name, model in models:
                result = Search(model, nodes=nodes, qdepth=config['qdepth']).run(restore(start))
                rows.append(dict(model=name, opening_id=start['id'], stratum=start['stratum'],
                                 depth=result.depth, nodes=result.nodes, seconds=result.elapsed_ms/1000))
        d1 = float(np.mean([r['depth'] >= 1 for r in rows]))
        d2 = float(np.mean([r['depth'] >= 2 for r in rows if r['stratum'] != 'endgame']))
        budgets.append(dict(nodes=nodes, depth_one=d1, depth_two=d2, rows=rows))
        print(f'Benchmark {nodes} nodes: depth one {d1:.1%}, depth two {d2:.1%}', flush=True)
    passing = [r for r in budgets if r['depth_one'] >= config['gates']['depth_one']
               and r['depth_two'] >= config['gates']['depth_two']]
    result = dict(passed=bool(passing), nodes=passing[0]['nodes'] if passing else None, budgets=budgets)
    write_json(path, result)
    return result


def forecast(config, groups, nodes, stage):
    root = Path(config['output'])
    path = root/f'forecast-{stage}.json'
    if path.exists():
        return read_json(path)
    initial = checkpoint(root/'pilot', 0)
    starts = [r for kind in ('endgame', 'middlegame', 'opening')
              for r in [x for x in groups['pilot'] if x['stratum'] == kind][:2]]
    settings = dict(nodes=nodes, qdepth=config['qdepth'], max_plies=config['max_plies'])
    sample = run_tasks(tasks_for(initial, 'handcrafted', starts, settings), root/'forecast-matches.json', config['workers'])
    production = dict(settings, nodes=config['production_nodes'], time_ms=config['production_time_ms'])
    timing = run_tasks(tasks_for(initial, config['supervised'], starts[:2], production),
                       root/'forecast-production-matches.json', config['workers'])
    seconds = float(np.quantile([r['seconds'] for r in sample['games']], .9))
    production_seconds = float(np.quantile([r['seconds'] for r in timing['games']], .9))
    if stage == 'main':
        history = read_json(root/'pilot/history.json')
        seconds = max(seconds, sum(r['active_seconds'] for r in history)*config['workers']/sum(r['games'] for r in history))
    training_games = config['population']*12*config['generations']*len(config['seeds'])
    screen_games = len(config['seeds'])*len(config['checkpoints'])*2*2*len(groups['development'])
    qualification_games = len(config['seeds'])*2*2*len(groups['qualification'])
    final_per_budget = len(config['seeds'])*2*2*len(groups['final'])
    main = 1.5*((training_games+screen_games+final_per_budget)*seconds
                +(qualification_games+final_per_budget)*production_seconds)/config['workers']
    pilot = 1.5*(config['pilot_generations']*config['population']*12+4*len(groups['pilot-check']))*seconds/config['workers']
    result = dict(stage=stage, measured_p90_node_game_seconds=seconds,
                  measured_p90_production_game_seconds=production_seconds, pilot_hours=pilot/3600,
                  main_hours=main/3600, safety_multiplier=1.5, main_matches=training_games+screen_games+qualification_games+2*final_per_budget,
                  paid_cost_cad=0, note='Measured forecast, not a CPU time cap; two workers and no paid compute.')
    write_json(path, result)
    print(f'Measured forecast: pilot {pilot/3600:.1f} hours; main {main/3600:.1f} hours (50% margin; C$0)', flush=True)
    return result


def pilot(config, groups, decision):
    root = Path(config['output'])
    saved = root/'pilot-decision.json'
    if saved.exists():
        return read_json(saved)
    if not decision['passed']:
        result = dict(passed=False, reason='No search budget passed the frozen depth quality gates')
        write_json(saved, result)
        return result
    nodes = decision['nodes']
    forecast(config, groups, nodes, 'pilot')
    history = evolve(config, groups['pilot'], config['pilot_seed'], config['pilot_generations'], nodes, root/'pilot')
    audits = [audit_ledger(read_json(path)) for path in sorted((root/'pilot').glob('generation-*/matches.json'))]
    write_json(root/'pilot-audit.json', dict(generations=audits))
    scores = [comparison(config, checkpoint(root/'pilot', g), 'handcrafted', groups['pilot-check'],
                         root/f'pilot-check-{g}.json', nodes) for g in (0, config['pilot_generations'])]
    total = sum(r['games'] for r in history)
    caps = sum(r['capped'] for r in history)/total
    decisive = sum(r['decisive'] for r in history)/total
    gain = scores[1]['score']-scores[0]['score']
    peak = max([r['peak_combined_bytes'] for r in history]+[r['peak_combined_bytes'] for r in scores])
    gates = dict(caps=caps <= config['gates']['pilot_caps'], decisive=decisive >= config['gates']['pilot_decisive'],
                 improvement=gain >= config['gates']['pilot_gain'], memory=peak < config['gates']['memory_gib']*1024**3)
    result = dict(passed=all(gates.values()), gates=gates, cap_fraction=caps, decisive_fraction=decisive,
                  initial_score=scores[0]['score'], trained_score=scores[1]['score'], gain=gain,
                  peak_combined_bytes=peak, nodes=nodes,
                  note='Feasibility screen only; not final strength evidence. No automatic retries or threshold changes.')
    write_json(saved, result)
    return result


def train(config, groups, nodes):
    root = Path(config['output'])
    for seed in config['seeds']:
        folder = root/f'seed-{seed}'
        evolve(config, groups['train'], seed, config['generations'], nodes, folder)
        audits = [audit_ledger(read_json(path)) for path in sorted(folder.glob('generation-*/matches.json'))]
        write_json(folder/'training-audit.json', dict(generations=audits))
        options = []
        for generation in config['checkpoints']:
            candidate = checkpoint(folder, generation)
            scores = [comparison(config, candidate, opponent, groups['development'],
                                 folder/f'screen-{generation}-{name}.json', nodes)
                      for name, opponent in [('classical', 'handcrafted'), ('supervised', config['supervised'])]]
            options.append(dict(generation=generation, checkpoint=candidate, sha256=digest(candidate),
                                score=float(np.mean([s['score'] for s in scores])), comparisons=scores))
        selected = max(options, key=lambda r: r['score'])  # Input order preserves earlier-generation ties.
        write_json(folder/'selection.json', dict(selected=selected, candidates=options, split='development'))


def aggregate(ledgers, config, pessimistic=False):
    # One bootstrap draw samples the same opening pairs jointly across all seeds.
    tables = []
    for ledger in ledgers:
        paired_summary(ledger['games'], repetitions=10)
        pairs = {}
        for game in ledger['games']:
            pairs.setdefault(game['opening_id'], []).append(0. if pessimistic and game['capped'] else game['score'])
        tables.append({k: float(np.mean(v)) for k, v in pairs.items()})
    keys = sorted(tables[0])
    if any(set(t) != set(keys) for t in tables):
        raise ValueError('Aggregate seeds must share opening pairs')
    matrix = np.array([[t[k] for k in keys] for t in tables])
    means = matrix.mean(axis=0)
    rng = np.random.default_rng(config['bootstrap_seed'])
    draws = [float(rng.choice(means, len(means), replace=True).mean()) for _ in range(config['bootstrap_resamples'])]
    return dict(score=float(matrix.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                seed_scores=matrix.mean(axis=1).tolist(), seed_sd=float(matrix.mean(axis=1).std(ddof=1)),
                note='Opening-pair interval conditional on these seeds; seed SD is separate.')


def release_gate(ledgers, config):
    result = {}
    representative = config['seeds'].index(config['representative_seed'])
    for name, sets in ledgers.items():
        threshold = config['gates']['classical_lower' if name == 'classical' else 'supervised_lower']
        normal, conservative = aggregate(sets, config), aggregate(sets, config, True)
        games = sets[representative]['games']
        pessimistic = [dict(g, score=0. if g['capped'] else g['score']) for g in games]
        individual = paired_summary(pessimistic, seed=config['bootstrap_seed'], repetitions=config['bootstrap_resamples'])
        cap_fraction = sum(g['capped'] for s in sets for g in s['games'])/sum(len(s['games']) for s in sets)
        lower = individual['ci95'][0]
        passed = ((lower > threshold and conservative['ci95'][0] > threshold) if name == 'classical'
                  else (lower >= threshold and conservative['ci95'][0] >= threshold))
        passed &= cap_fraction <= config['gates']['release_caps']
        if name == 'classical':
            passed &= sum(s > .5 for s in normal['seed_scores']) >= 2
        result[name] = dict(passed=bool(passed), aggregate=normal, pessimistic=conservative,
                            representative=individual, cap_fraction=cap_fraction)
    return dict(passed=all(r['passed'] for r in result.values()), opponents=result)


def evaluate(config, groups, nodes):
    root = Path(config['output'])
    selected = {str(seed): read_json(root/f'seed-{seed}/selection.json')['selected'] for seed in config['seeds']}
    seal = root/'selections.json'
    if seal.exists() and read_json(seal) != selected:
        raise ValueError('Selections changed after qualification/final evaluation began')
    write_json(seal, selected)
    results = {}
    for stage, pool, budget, time_ms in [('qualification', 'qualification', config['production_nodes'], config['production_time_ms']),
            ('final-production', 'final', config['production_nodes'], config['production_time_ms']),
            ('final-nodes', 'final', nodes, None)]:
        sets = {'classical': [], 'supervised': []}
        for seed in config['seeds']:
            candidate = selected[str(seed)]['checkpoint']
            if digest(candidate) != selected[str(seed)]['sha256']:
                raise ValueError('Selected checkpoint identity changed')
            for name, opponent in [('classical', 'handcrafted'), ('supervised', config['supervised'])]:
                path = root/f'seed-{seed}/{stage}-{name}.json'
                comparison(config, candidate, opponent, groups[pool], path, budget, time_ms)
                sets[name].append(read_json(path))
        results[stage] = release_gate(sets, config)
    results['release_eligible'] = results['qualification']['passed'] and results['final-production']['passed']
    results['representative_seed'] = config['representative_seed']
    write_json(root/'results.json', results)
    return results


def run(config, stage='all'):
    root = Path(config['output'])
    freeze(config)
    started = time.perf_counter()
    elapsed_path = root/'elapsed.json'
    elapsed = read_json(elapsed_path) if elapsed_path.exists() else dict(active_seconds=0., paid_cost_cad=0)
    try:
        groups = prepare(config)
        if stage == 'prepare':
            return {'prepared': {k: len(v) for k, v in groups.items()}}
        if stage in ('report', 'release'):
            from .evolution_report import report, release
            return release(config) if stage == 'release' else report(config)
        decision = benchmark(config, groups)
        if stage == 'benchmark':
            return decision
        outcome = pilot(config, groups, decision)
        if stage == 'pilot' or not outcome['passed']:
            from .evolution_report import report
            report(config)
            return outcome
        forecast(config, groups, decision['nodes'], 'main')
        if stage in ('all', 'train'):
            train(config, groups, decision['nodes'])
        if stage in ('all', 'evaluate'):
            evaluate(config, groups, decision['nodes'])
        from .evolution_report import report, release
        report(config)
        if stage == 'all':
            release(config)
        return {'status': 'complete' if (root/'results.json').exists() else 'trained'}
    finally:
        elapsed['active_seconds'] += time.perf_counter()-started
        write_json(elapsed_path, elapsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/evolutionary.json')
    parser.add_argument('--stage', choices=['prepare', 'benchmark', 'pilot', 'train', 'evaluate', 'report', 'release', 'all'], default='all')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(read_json(args.config), args.stage), flush=True)


if __name__ == '__main__':
    main()
