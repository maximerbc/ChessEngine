"""Separately identified GPU-first evolutionary calibration and pilot.

No teacher, gradients, inherited initializer, full-study launch or promotion.
"""
import argparse
import json
from pathlib import Path
import time

import chess
import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms import Sep_CMA_ES

from ..features import restore, encode
from ..model import NumpyEvaluator
from ..rootcover import RootCoverageValueSearch, STRATEGY
from . import native_games, rules
from .evolution import fitness
from .games import tasks_for
from .network import initialize, export, import_numpy, apply
from .protocol import freeze, sha, atomic_json, save_state, load_state, nvidia_available
from .native_search import make_search


def export_candidate(params, path, seed, generation, protocol, purpose):
    # Reuse array/schema validation, then give this separate lane its own identity.
    export(params, path, 'evolutionary', purpose,
           dict(seed=seed, generation=generation, protocol_id=protocol, qualified=False))
    path = Path(path)
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {k: bundle[k].copy() for k in bundle.files}
    metadata = json.loads(str(arrays['metadata']))
    metadata['study'] = 'evolutionary-gpu-native-v1'
    metadata['search'] = STRATEGY
    arrays['metadata'] = np.array(json.dumps(metadata))
    temporary = path.with_suffix('.identity.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def preflight(supervised, output):
    """Fixed correctness probes; final pools and position labels stay sealed."""
    boards = [chess.Board(), chess.Board('4k3/P7/8/8/8/8/8/4K3 w - - 0 1'),
              chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1'),
              chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 100 1'),
              chess.Board('8/8/8/8/8/8/6k1/4K2R w - - 99 1')]
    repeated = chess.Board()
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1']:
        repeated.push_uci(move)
    boards.append(repeated)
    params = import_numpy(supervised)
    cpu = NumpyEvaluator(supervised)
    errors = []
    for board in boards:
        state = rules.from_board(board)
        rules.assert_parity(board, state)
        error = abs(float(apply(params, encode(board, 3))) - cpu(board))
        if error > 1e-5:
            raise ValueError('CPU/GPU inference tolerance exceeded')
        errors.append(error)
    # A fixed shallow tree must agree beyond mere scalar inference.
    root = boards[0]
    result = make_search(16)(jax.tree.map(lambda x: x[None], params),
        jax.tree.map(lambda x: x[None], rules.from_board(root)), jnp.ones(1, jnp.int32))
    reference = RootCoverageValueSearch(cpu, 16, None).run(root)
    if rules.action_map(root)[int(result['actions'][0])].uci() != reference.move:
        raise ValueError('CPU/GPU search action mismatch')
    if abs(float(result['value'][0]) - reference.value) > 1e-5:
        raise ValueError('CPU/GPU search value mismatch')
    atomic_json(output / 'preflight.json', dict(passed=True, positions=len(boards),
        inference_max_absolute_error=max(errors), search_parity=True, devices=[str(d) for d in jax.devices()]))


def optimizer_probe(config, output):
    """Measure the actual 64-member ask/tell separately from the 8-member pilot."""
    strategy = Sep_CMA_ES(config['main']['population'], initialize(jax.random.PRNGKey(1)))
    settings = strategy.default_params.replace(std_init=config['sigma'])
    start = time.perf_counter()
    state = strategy.init(jax.random.PRNGKey(2), initialize(jax.random.PRNGKey(3)), settings)
    population, state = strategy.ask(jax.random.PRNGKey(4), state, settings)
    state, _ = strategy.tell(jax.random.PRNGKey(5), population, jnp.zeros(config['main']['population']), state, settings)
    jax.block_until_ready(state)
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for k in range(3):
        population, state = strategy.ask(jax.random.PRNGKey(10 + k), state, settings)
        state, _ = strategy.tell(jax.random.PRNGKey(20 + k), population, jnp.zeros(config['main']['population']), state, settings)
    jax.block_until_ready(state)
    record = dict(population=64, compile_and_first_update_seconds=compile_seconds,
                  seconds_per_generation=(time.perf_counter() - start) / 3,
                  synthetic_fitness=True, not_learning_evidence=True)
    atomic_json(output / 'optimizer-probe.json', record)
    return record


def check_deadline(deadline):
    if deadline is not None and time.time() >= deadline:
        raise native_games.DeadlineReached('Budget deadline reached before next stage')


def calibrate(config, pools, supervised, output, deadline):
    out = output / 'calibration'
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / 'selection.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    parameters = initialize(jax.random.PRNGKey(config['pilot']['seed']))
    roots = pools['coverage']
    boards = [restore(o) for o in roots]
    imported = [rules.from_board(b) for b in boards]
    # Same roots and three evaluator classes at every simulation budget.
    states = jax.tree.map(lambda *x: jnp.stack(x * 3), *imported)
    params = jax.tree.map(lambda a, b: jnp.stack([a] * len(roots) + [b] * len(roots) + [a] * len(roots)),
                         parameters, import_numpy(supervised))
    kinds = jnp.array([1] * (2 * len(roots)) + [0] * len(roots), jnp.int32)
    trials = []
    selected = None
    for simulations in config['simulation_candidates']:
        check_deadline(deadline)
        before = time.perf_counter()
        result = make_search(simulations)(params, states, kinds)
        jax.block_until_ready(result)
        coverage = np.asarray(result['root_coverage'])
        passed = all(float(np.mean(c >= config['coverage_root_fraction'])) >= config['coverage_position_fraction']
                     for c in np.split(coverage, 3))
        if np.asarray(result['overflow']).any() or not np.all(np.asarray(result['visits']).sum(1) == simulations):
            raise ValueError('Coverage trial failed history/budget integrity')
        trials.append(dict(simulations=simulations, compilation_and_search_seconds=time.perf_counter()-before,
            root_coverages=coverage.tolist(), passed=passed))
        atomic_json(out / 'coverage.json', dict(trials=trials, selected_simulations=simulations if passed else None))
        if passed:
            selected = simulations
            break
    if selected is None:
        atomic_json(result_path, dict(passed=False, reason='No simulation budget passed frozen coverage', trials=trials))
        return json.loads(result_path.read_text())
    timings = []
    # Fixed roots, search effort and models; only scheduling dimensions vary.
    for batch in config['batch_candidates']:
        state = jax.tree.map(lambda *x: jnp.stack([x[i % len(x)] for i in range(batch)]), *imported)
        bank = jax.tree.map(lambda x: x[None], parameters)
        for chunk in config['chunk_candidates']:
            check_deadline(deadline)
            advance = native_games.rollout(selected, chunk, config['max_plies'])
            args = (bank, state, jnp.zeros(batch, jnp.int32), jnp.zeros(batch, jnp.int32),
                    jnp.ones(batch, bool), jnp.zeros(batch, jnp.int32), jnp.ones(batch, bool), jnp.zeros(batch, jnp.int32))
            start = time.perf_counter()
            first = advance(*args)
            jax.block_until_ready(first)
            compile_seconds = time.perf_counter() - start
            samples = []
            for _ in range(2):
                check_deadline(deadline)
                start = time.perf_counter()
                result = advance(*args)
                jax.block_until_ready(result)
                # Tune the execution path, including the actual transfer/fsync
                # that chunking amortizes, rather than isolated GPU kernels.
                save_state(out/'layout-checkpoint.pkl', jax.device_get(result), 'layout-only')
                if np.asarray(result['invalid']).any() or np.asarray(result['overflow']).any():
                    raise ValueError('Scheduling trial failed integrity')
                samples.append(time.perf_counter() - start)
            moves = int((np.asarray(result['actions']) >= 0).sum())
            timings.append(dict(batch_size=batch, chunk_plies=chunk, moves=moves,
                compile_seconds=compile_seconds, seconds=float(np.median(samples)),
                moves_per_second=moves/float(np.median(samples)), includes_checkpoint_io=True))
            atomic_json(out / 'timings.json', timings)
    best = max(timings, key=lambda r: r['moves_per_second'])
    record = dict(passed=True, simulations=selected, batch_size=best['batch_size'],
        chunk_plies=best['chunk_plies'], coverage_trials=trials, timing_trials=timings,
        timing_only=True, scientific_workload_changed=False)
    atomic_json(result_path, record)
    return record


def evolve(config, pools, supervised, output, selected, protocol, deadline):
    settings = config['pilot']
    seed, population_size = settings['seed'], settings['population']
    strategy = Sep_CMA_ES(population_size, initialize(jax.random.PRNGKey(seed)))
    es_params = strategy.default_params.replace(std_init=config['sigma'])
    path = output / 'optimizer.pkl'
    if path.exists():
        saved = load_state(path, protocol)
        saved = jax.tree.map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, saved)
    else:
        key, init = jax.random.split(jax.random.PRNGKey(seed))
        mean = initialize(init)
        initial = output / 'initial.npz'
        export_candidate(mean, initial, seed, 0, protocol, 'random-independent-initializer')
        saved = dict(es=strategy.init(init, mean, es_params), rng=key, generation=0,
                     pending=None, history=[], archive=[str(initial)], archive_hashes={str(initial):sha(initial)})
        save_state(path, jax.device_get(saved), protocol)
    for p, expected in saved['archive_hashes'].items():
        if sha(p) != expected:
            raise ValueError('Evolutionary archive changed')
    while saved['generation'] < settings['generations']:
        check_deadline(deadline)
        generation = saved['generation'] + 1
        directory = output / f'generation-{generation:03}'
        directory.mkdir(parents=True, exist_ok=True)
        if saved['pending'] is None:
            saved['rng'], key = jax.random.split(saved['rng'])
            saved['pending'], saved['es'] = strategy.ask(key, saved['es'], es_params)
            save_state(path, jax.device_get(saved), protocol)
        import random
        starts = random.Random(seed * 100 + generation).sample(pools['pilot_train'], 24)
        archive = saved['archive'][generation % len(saved['archive'])]
        tasks = []
        for i in range(population_size):
            candidate = directory / f'candidate-{i:03}.npz'
            params = jax.tree.map(lambda x: x[i], saved['pending'])
            if candidate.exists():
                if any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in
                       zip(jax.tree.leaves(params),jax.tree.leaves(import_numpy(candidate)))):
                    raise ValueError('Pending candidate identity changed')
            else:
                export_candidate(params, candidate, seed, generation, protocol, 'pilot-candidate')
            for index, (group, opponent) in enumerate([('classical','handcrafted'), ('supervised',supervised), ('archive',archive)]):
                tasks.extend(tasks_for(candidate, opponent, starts[index*8:index*8+8], group, i))
        ledger = native_games.run(tasks, directory, selected['simulations'], config['max_plies'],
            selected['batch_size'], selected['chunk_plies'], deadline)
        scores = fitness(ledger['games'], population_size, 16)
        if not all(g['replay_verified'] for g in ledger['games']):
            raise ValueError('Unverified fitness record')
        saved['rng'], key = jax.random.split(saved['rng'])
        saved['es'], _ = strategy.tell(key, saved['pending'], -jnp.asarray(scores), saved['es'], es_params)
        mean = strategy.get_mean(saved['es'])
        target = output / f'mean-{generation:03}.npz'
        export_candidate(mean, target, seed, generation, protocol, 'pilot-mean-not-qualified')
        saved['archive'].append(str(target))
        saved['archive_hashes'][str(target)] = sha(target)
        saved['generation'], saved['pending'] = generation, None
        saved['history'].append(dict(generation=generation, mean_fitness=float(scores.mean()),
            games=len(ledger['games']), caps=sum(g['capped'] for g in ledger['games']),
            active_seconds=ledger['active_seconds'], checkpoint_sha256=sha(target)))
        save_state(path, jax.device_get(saved), protocol)
        atomic_json(output / 'history.json', saved['history'])
    return saved['history']


def forecast(output, config, rate_usd):
    ledgers = list(output.rglob('games.json'))
    games, seconds, moves, caps = 0, 0., 0, 0
    for path in ledgers:
        if '/checks/' in str(path):
            continue
        d = json.loads(path.read_text())
        games += len(d['games'])
        seconds += d['active_seconds'] + d.get('checkpoint_seconds', 0.)
        moves += sum(g['plies'] for g in d['games'])
        caps += sum(g['capped'] for g in d['games'])
    # Includes time spent on current unfinished games: intentionally conservative.
    rate = games / seconds if games and seconds else None
    scenarios = []
    for count in [184320, 368640, 737280]:
        hours = count/rate/3600 if rate else None
        scenarios.append(dict(main_training_games=count, measured_rate_projection_hours=hours,
            projected_compute_usd=hours*rate_usd if hours is not None else None,
            with_20_percent_contingency_usd=hours*rate_usd*1.2 if hours is not None else None))
    record = dict(evidence='pilot-throughput-not-playing-strength', completed_games=games,
        measured_seconds=seconds, games_per_second=rate, completed_plies=moves, caps=caps,
        scenarios=scenarios, full_training_authorized=False,
        additional_costs=dict(labeling_usd=0, supervised_fits_usd=0,
            development_and_qualification='unmeasured; full cost forecast incomplete',
            storage='provider receipt; not included in compute projections'),
        limitation='Small random populations and unfinished games cannot establish stable main-study throughput.')
    atomic_json(output / 'forecast.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/evolutionary-gpu-native.json')
    parser.add_argument('--openings', required=True)
    parser.add_argument('--supervised', default='artifacts/models/supervised-1m.npz')
    parser.add_argument('--output', required=True)
    parser.add_argument('--deadline-unix', required=True, type=float)
    parser.add_argument('--hourly-usd', default=.89, type=float)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    if config['full_training_authorized'] or config['study'] != 'evolutionary-gpu-native-v1':
        raise ValueError('This command is exclusively the separately authorized pilot')
    if not nvidia_available():
        raise RuntimeError('Actual NVIDIA execution required; no silent CPU fallback')
    if sha(args.supervised) != config['supervised_sha256']:
        raise ValueError('Frozen augmented opponent identity changed')
    protocol = freeze(output, config, [args.openings, args.supervised, args.config])
    pools = json.loads(Path(args.openings).read_text())
    status = dict(status='started', protocol_id=protocol, started_unix=time.time(),
                  deadline_unix=args.deadline_unix, budget_cad=config['budget_cad'])
    atomic_json(output / 'status.json', status)
    def stage(name):
        status['stage'] = name
        atomic_json(output / 'status.json', status)
        print(json.dumps(dict(stage=name, unix=time.time())), flush=True)
    try:
        check_deadline(args.deadline_unix)
        stage('correctness-preflight')
        preflight(args.supervised, output)
        check_deadline(args.deadline_unix)
        stage('full-population-optimizer-probe')
        optimizer_probe(config, output)
        stage('coverage-and-scheduling-calibration')
        selected = calibrate(config, pools, args.supervised, output, args.deadline_unix)
        if not selected['passed']:
            status.update(status='coverage-failed', stage='calibration')
            return
        candidate = output / 'benchmark-random.npz'
        export_candidate(initialize(jax.random.PRNGKey(1234)), candidate, config['pilot']['seed'], 0, protocol, 'benchmark-only')
        stage('interruption-recovery-check')
        checks = tasks_for(candidate, 'handcrafted', pools['benchmark'][:1], 'classical')
        native_games.run(checks, output/'checks/resumed', selected['simulations'], 2, 2, 1,
                         args.deadline_unix, stop_after_chunks=1)
        resumed = native_games.run(checks, output/'checks/resumed', selected['simulations'], 2, 2, 1, args.deadline_unix)
        whole = native_games.run(checks, output/'checks/whole', selected['simulations'], 2, 2, 1, args.deadline_unix)
        if resumed['games'] != whole['games']:
            raise ValueError('Interrupted GPU games differ from uninterrupted execution')
        atomic_json(output/'resume-check.json', dict(passed=True, capped_smoke_only=True))
        tasks = []
        for group, opponent in [('classical','handcrafted'), ('supervised',args.supervised), ('archive',str(candidate))]:
            tasks.extend(tasks_for(candidate, opponent, pools['benchmark'], group))
        stage('complete-game-throughput-benchmark')
        ledger = native_games.run(tasks, output/'benchmark', selected['simulations'], config['max_plies'],
            selected['batch_size'], selected['chunk_plies'], args.deadline_unix)
        coverage = sum(g['coverage_hits'] for g in ledger['games']) / max(1, sum(g['plies'] for g in ledger['games']))
        if coverage < config['coverage_position_fraction']:
            status.update(status='coverage-failed', stage='complete-games', coverage=coverage)
            return
        stage('small-evolution-pilot')
        evolve(config, pools, args.supervised, output/'pilot', selected, protocol, args.deadline_unix)
        status.update(status='pilot-complete', games=768, qualified=False)
    except native_games.DeadlineReached as exc:
        status.update(status='budget-stopped-partial', reason=str(exc))
    except BaseException as exc:
        status.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        status['ended_unix'] = time.time()
        atomic_json(output/'status.json', status)
        forecast(output, config, args.hourly_usd)


if __name__ == '__main__':
    main()
