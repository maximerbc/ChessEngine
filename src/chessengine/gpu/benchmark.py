"""Complete-game hardware measurements, frozen PUCT coverage and resume checks."""
import json
from pathlib import Path
import resource
import time
import jax
import jax.numpy as jnp
import numpy as np
from ..features import restore
from ..model import NumpyEvaluator
from ..puct_serving import ValueSearch
from . import rules, games
from .network import initialize, export
from .search import make_search
from .protocol import freeze, atomic_json, identity, nvidia_available


def benchmark(config, openings_path, directory, batch_size=16, smoke=False, supervised=None):
    directory = Path(directory)
    protocol = freeze(directory, dict(config=config, smoke=smoke), [openings_path]+([supervised] if supervised else []))
    if (directory / 'benchmark.json').exists():
        return json.loads((directory/'benchmark.json').read_text())
    pool = json.loads(Path(openings_path).read_text())['pilot']
    pairs = 1 if smoke else config['search']['benchmark_pairs']
    if len(pool) < pairs:
        raise ValueError('Not enough preregistered benchmark opening pairs')
    starts = pool[:pairs]
    params = initialize(jax.random.PRNGKey(20261121))
    candidate = directory / 'random-benchmark.npz'
    export(params, candidate, 'evolutionary', 'benchmark-only-random-network')
    proxy = directory / 'random-supervised-proxy.npz'
    export(initialize(jax.random.PRNGKey(42)),proxy,'supervised','benchmark-only-untrained-proxy')
    opponent = str(supervised) if supervised else str(proxy)
    board = restore(starts[0])
    state = rules.from_board(board)
    states = jax.tree.map(lambda x: jnp.stack([x] * min(batch_size, pairs * 2)), state)
    parameters = jax.tree.map(lambda x: jnp.stack([x] * len(states.game.color)), params)
    results = []
    for simulations in ([2] if smoke else config['search']['simulation_candidates']):
        timing_path = directory / f'timing-{simulations}.json'
        if timing_path.exists():
            results.append(json.loads(timing_path.read_text()))
            continue
        target = directory / f'simulations-{simulations}'
        if (target/'games.json').exists():
            raise ValueError('An interrupted timing trial cannot be timed from cached games; retain it and use a fresh benchmark directory')
        search = make_search(simulations)
        start = time.perf_counter()
        first = search(parameters, states, jnp.ones(len(states.game.color), jnp.int32))
        jax.block_until_ready(first)
        compile_seconds = time.perf_counter() - start
        tasks = []
        for index in range(1 if smoke else config['search']['benchmark_candidates']):
            path = directory / f'random-candidate-{index:03}.npz'
            export(initialize(jax.random.PRNGKey(20261121 + index)),path,'evolutionary','benchmark-only')
            for group,reference in [('classical','handcrafted'),('supervised-proxy' if not supervised else 'supervised',opponent),
                                    ('archive',str(candidate))]:
                tasks.extend(games.tasks_for(str(path),reference,starts,group,index))
        started = time.perf_counter()
        ledger = games.run(tasks, target, simulations, 2 if smoke else config['evolution']['max_plies'], batch_size)
        total_seconds = time.perf_counter() - started
        coverage = []
        # Fixed roots include all benchmark starts; full-game coverage is recorded by the runner.
        for row in starts:
            b = restore(row)
            s = rules.from_board(b)
            r = search(jax.tree.map(lambda x: x[None], params), jax.tree.map(lambda x: x[None], s), jnp.ones(1, jnp.int32))
            coverage.append(float(r['root_coverage'][0]))
        searched_moves = sum(r['plies'] for r in ledger['games'])
        move_coverage = sum(r['coverage_hits'] for r in ledger['games']) / max(1, searched_moves)
        coverage_qualified = move_coverage >= config['search']['coverage_move_fraction']
        cpu = ValueSearch(NumpyEvaluator(candidate), simulations, None).run(board)
        gpu = search(jax.tree.map(lambda x: x[None], params), jax.tree.map(lambda x: x[None], state), jnp.ones(1, jnp.int32))
        parity = cpu.move == rules.action_map(board)[int(gpu['actions'][0])].uci() and np.isclose(cpu.value, float(gpu['value'][0]), atol=1e-5)
        # Actual interrupted batch, then resume; compare complete move histories and outcomes.
        resumed_dir = directory / f'resume-{simulations}'
        games.run(tasks[:2], resumed_dir, simulations, 2, batch_size=2, max_moves=1)
        resumed = games.run(tasks[:2], resumed_dir, simulations, 2, batch_size=2)
        whole = games.run(tasks[:2], directory / f'whole-{simulations}', simulations, 2, batch_size=2)
        resume = resumed['games'] == whole['games']
        stats = jax.devices()[0].memory_stats() or {}
        results.append(dict(simulations=simulations, games=len(ledger['games']),
            compile_and_first_search_seconds=compile_seconds, total_seconds=total_seconds,
            tournament_games_per_second=len(ledger['games']) / total_seconds,
            caps=sum(r['capped'] for r in ledger['games']),
            root_coverages=coverage, coverage_qualified=bool(coverage_qualified),
            full_game_move_coverage=move_coverage, searched_moves=searched_moves,
            rules_parity_passed=bool(parity), resume_verified=resume,
            cpu_serving_elapsed_ms=cpu.elapsed_ms, memory_stats={k: int(v) for k, v in stats.items()}))
        atomic_json(timing_path,results[-1])
    qualifying = [r for r in results if r['coverage_qualified'] and r['rules_parity_passed'] and r['resume_verified']]
    record = dict(protocol_id=protocol, hardware=[str(d) for d in jax.devices()],
        measured_on_nvidia=nvidia_available() and not smoke,
        smoke=smoke, batch_size=batch_size, results=results,
        tournament_candidates=1 if smoke else config['search']['benchmark_candidates'],
        supervised_opponent='frozen-augmented-supervised' if supervised else 'untrained-random-proxy',
        selected_simulations=min([r['simulations'] for r in qualifying], default=None),
        peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        workload_sha256=identity(config), timing_includes='full game batches, host rule audit, checkpoint I/O',
        warning='Random-network forecast is provisional; remeasure with frozen supervised opponent before main evolution.')
    atomic_json(directory / 'benchmark.json', record)
    return record
