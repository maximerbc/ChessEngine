"""Independent Sep-CMA-ES with common openings and separately frozen pilot."""
import json
from pathlib import Path
import random
import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms import Sep_CMA_ES
from . import games
from .network import initialize, export, import_numpy
from .protocol import freeze, save_state, load_state, atomic_json, sha, identity


def fitness(rows, population, games_per_group):
    scores = []
    for candidate in range(population):
        weighted = 0.
        for group, weight in [('classical', .5), ('supervised', .25), ('archive', .25)]:
            group_rows = [r for r in rows if r['candidate_index'] == candidate and r['group'] == group]
            if len(group_rows) != games_per_group:
                raise ValueError('Incomplete or duplicated candidate tournament')
            pairs = {}
            for row in group_rows:
                pairs.setdefault(row['opening']['id'], set()).add(row['color'])
                if row['capped'] != (row['result'] == '*'):
                    raise ValueError('Cap must retain unfinished rule outcome')
            if len(pairs) != games_per_group // 2 or any(c != {'white', 'black'} for c in pairs.values()):
                raise ValueError('Incomplete color-swapped opening pairs')
            weighted += weight * np.mean([r['score'] for r in group_rows])
        scores.append(weighted)
    return np.asarray(scores)


def evolve(config, openings_path, supervised, output, seed, pilot=False, batch_size=32, stop_after=None):
    output = Path(output)
    pool = json.loads(Path(openings_path).read_text())['pilot_train' if pilot else 'train']
    if not pilot and seed not in config['seeds']:
        raise ValueError('Unregistered seed')
    if pilot and seed != config['pilot_seed']:
        raise ValueError('The 30,720-game pilot uses only its separate frozen pilot seed')
    count = config['games_per_candidate'] // 3
    if count % 2 or count < 2:
        raise ValueError('Three equal opponent groups require complete opening pairs')
    frozen = dict(config=config, seed=seed, stage='pilot' if pilot else 'main')
    from ..model import NumpyEvaluator
    if NumpyEvaluator(supervised).metadata.get('seed') != 42:
        raise ValueError('Opponent must be the frozen augmented Supervised seed-42 checkpoint')
    protocol = freeze(output, frozen, [openings_path, supervised] + ([] if pilot else [config['pilot_decision']]))
    if not pilot:
        gate = json.loads(Path(config['pilot_decision']).read_text())
        if not gate.get('passed') or gate.get('config_sha256') != identity(dict(config, games_per_candidate=48)):
            raise ValueError('Frozen pilot failed or does not match this study')
    strategy = Sep_CMA_ES(config['population'], initialize(jax.random.PRNGKey(seed)))
    es_params = strategy.default_params.replace(std_init=config['sigma'])
    state_path = output / 'optimizer.pkl'
    if state_path.exists():
        state = load_state(state_path, protocol)
    else:
        rng, init_key = jax.random.split(jax.random.PRNGKey(seed))
        mean = initialize(init_key)
        export(mean, output / 'initial.npz', 'evolutionary', 'gpu-evolution-random-initialization',
               dict(seed=seed, protocol_id=protocol, generation=0))
        state = dict(es=strategy.init(init_key, mean, es_params), rng=rng, generation=0,
                     pending=None, archive=[str(output / 'initial.npz')], history=[],
                     archive_hashes={str(output/'initial.npz'):sha(output/'initial.npz')})
        save_state(state_path, jax.device_get(state), protocol)
    generations = config['pilot_generations'] if pilot else config['generations']
    if any(sha(path) != expected for path,expected in state['archive_hashes'].items()):
        raise ValueError('Archived evolutionary checkpoint changed')
    while state['generation'] < generations:
        generation = state['generation'] + 1
        directory = output / f'generation-{generation:03}'
        directory.mkdir(parents=True, exist_ok=True)
        if state['pending'] is None:
            state['rng'], ask_key = jax.random.split(state['rng'])
            population, state['es'] = strategy.ask(ask_key, state['es'], es_params)
            state['pending'] = population
            save_state(state_path, jax.device_get(state), protocol)
        population = jax.tree.map(jnp.asarray, state['pending'])
        starts = random.Random(seed * 1000 + generation).sample(pool, 3 * count // 2)
        archive = random.Random(f'{seed}:{generation}:archive').choice(state['archive'])
        tasks = []
        for index in range(config['population']):
            candidate = directory / f'candidate-{index:03}.npz'
            params = jax.tree.map(lambda x: x[index], population)
            if not candidate.exists():
                export(params, candidate, 'evolutionary', 'gpu-evolution-tournament-candidate',
                       dict(seed=seed, protocol_id=protocol, generation=generation, candidate=index))
            else:
                restored = import_numpy(candidate)
                if any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in
                       zip(jax.tree.leaves(params),jax.tree.leaves(restored))):
                    raise ValueError('Pending candidate differs from the frozen population')
            for group_index,(group, opponent) in enumerate([('classical', 'handcrafted'), ('supervised', supervised), ('archive', archive)]):
                group_starts = starts[group_index * count // 2:(group_index + 1) * count // 2]
                tasks.extend(games.tasks_for(candidate, opponent, group_starts, group, index))
        ledger = games.run(tasks, directory, config['simulations'], config['max_plies'], batch_size)
        scores = fitness(ledger['games'], config['population'], count)
        state['rng'], tell_key = jax.random.split(state['rng'])
        # evosax minimizes; tournament fitness is maximized.
        state['es'], _ = strategy.tell(tell_key, population, -jnp.asarray(scores), state['es'], es_params)
        mean = strategy.get_mean(state['es'])
        checkpoint = output / f'mean-{generation:03}.npz'
        export(mean, checkpoint, 'evolutionary', 'gpu-evolution-development',
               dict(seed=seed, protocol_id=protocol, generation=generation))
        state['archive'].append(str(checkpoint))
        state['archive_hashes'][str(checkpoint)] = sha(checkpoint)
        state['history'].append(dict(generation=generation, mean_fitness=float(scores.mean()),
            best_fitness=float(scores.max()), caps=sum(r['capped'] for r in ledger['games']),
            games=len(ledger['games']), checkpoint_sha256=sha(checkpoint)))
        state.update(generation=generation, pending=None)
        save_state(state_path, jax.device_get(state), protocol)
        atomic_json(output / 'history.json', state['history'])
        if stop_after is not None and generation >= stop_after:
            break
    return state['history']
