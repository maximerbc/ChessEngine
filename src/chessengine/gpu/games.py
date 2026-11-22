"""Resumable complete-game GPU batches, with truthful rule outcomes and caps."""
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from ..features import restore, record_board
from . import rules
from .network import import_numpy, initialize
from .search import make_search
from .protocol import identity, sha, save_state, load_state, atomic_json

BATCH_STEP = jax.jit(jax.vmap(lambda s, a, go: jax.lax.cond(
    go, lambda _: rules.step(s, a), lambda _: s, None)))


def task(candidate, opponent, opening, color, group, index=0):
    row = dict(candidate=str(candidate), opponent=str(opponent), opening=opening,
               color=color, group=group, candidate_index=index,
               candidate_sha256=sha(candidate),
               opponent_sha256='handcrafted-v1' if opponent == 'handcrafted' else sha(opponent))
    return dict(row, task_id=identity(row))


def tasks_for(candidate, opponent, openings, group, index=0):
    return [task(candidate, opponent, row, color, group, index)
            for row in openings for color in ['white', 'black']]


def run(tasks, directory, simulations, max_plies=512, batch_size=32, max_moves=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if batch_size < 1 or not tasks or len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Nonempty unique tasks and positive batch size required')
    for row in tasks:
        if sha(row['candidate']) != row['candidate_sha256']:
            raise ValueError('Candidate changed')
        if row['opponent'] != 'handcrafted' and sha(row['opponent']) != row['opponent_sha256']:
            raise ValueError('Opponent changed')
    protocol = identity(dict(tasks=tasks, simulations=simulations, max_plies=max_plies))
    ledger_path = directory / 'games.json'
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger['protocol_id'] != protocol:
            raise ValueError('Game protocol changed')
    else:
        ledger = dict(protocol_id=protocol, games=[], active_seconds=0.)
    completed = {g['task_id'] for g in ledger['games']}
    remaining = [t for t in tasks if t['task_id'] not in completed]
    search = make_search(simulations)
    moves_completed = 0
    while remaining:
        pending_path = directory / 'pending.pkl'
        if pending_path.exists():
            pending = load_state(pending_path, protocol)
            group = pending['tasks']
            boards = [restore(row) for row in pending['boards']]
            states = jax.tree.map(jnp.asarray, pending['states'])
            plies = pending['plies']
            coverage_hits = pending['coverage_hits']
            if any(t['task_id'] not in {r['task_id'] for r in remaining} for t in group):
                # A completed ledger write can precede removal of the pending batch.
                if all(t['task_id'] in completed for t in group):
                    pending_path.unlink()
                    continue
                raise ValueError('Pending game batch disagrees with the ledger')
        else:
            group = remaining[:batch_size]
            boards = [restore(t['opening']) for t in group]
            states = jax.tree.map(lambda *x: jnp.stack(x), *[rules.from_board(b) for b in boards])
            plies = np.zeros(len(group), dtype=np.int32)
            coverage_hits = np.zeros(len(group), dtype=np.int32)
        candidate = jax.tree.map(lambda *x: jnp.stack(x), *[import_numpy(t['candidate']) for t in group])
        opponent = jax.tree.map(lambda *x: jnp.stack(x), *[
            initialize(jax.random.PRNGKey(0)) if t['opponent'] == 'handcrafted' else import_numpy(t['opponent'])
            for t in group])
        candidate_white = jnp.array([t['color'] == 'white' for t in group])
        opponent_kind = jnp.array([0 if t['opponent'] == 'handcrafted' else 1 for t in group])
        while True:
            ended = np.array([b.is_game_over(claim_draw=True) for b in boards])
            finished = ended | (plies >= max_plies)
            if finished.all():
                break
            started = time.perf_counter()
            acting_candidate = (states.game.color == 0) == candidate_white
            params = jax.tree.map(lambda a, b: jnp.where(acting_candidate.reshape((-1,) + (1,) * (a.ndim - 1)), a, b),
                                  candidate, opponent)
            kinds = jnp.where(acting_candidate, 1, opponent_kind)
            result = search(params, states, kinds)
            actions = np.asarray(result['actions'])
            coverage_hits += ((np.asarray(result['root_coverage']) >= .8) & ~finished)
            if np.any(np.asarray(result['visits']).sum(1)[~finished] != simulations):
                raise ValueError('Incomplete fixed-budget GPU search')
            if np.asarray(result['overflow']).any():
                raise ValueError('GPU history capacity exceeded')
            if not np.array_equal(np.asarray(result['terminal'])[~finished], ended[~finished]):
                raise ValueError('GPU/python-chess automatic draw mismatch')
            active = jnp.asarray(~finished)
            states = BATCH_STEP(states, jnp.asarray(actions), active)
            for i, board in enumerate(boards):
                if finished[i]:
                    continue
                mapping = rules.action_map(board)
                if actions[i] not in mapping:
                    raise ValueError('GPU selected a python-chess-illegal action')
                board.push(mapping[actions[i]])
                plies[i] += 1
            jax.block_until_ready(states)
            ledger['active_seconds'] += time.perf_counter() - started
            pending = dict(tasks=group, boards=[record_board(b) for b in boards],
                           states=jax.device_get(states), plies=plies, coverage_hits=coverage_hits)
            save_state(pending_path, pending, protocol)
            atomic_json(ledger_path, ledger)
            moves_completed += 1
            if max_moves is not None and moves_completed >= max_moves:
                return ledger
        for index, (row, board, ply) in enumerate(zip(group, boards, plies)):
            rules.assert_parity(board, jax.tree.map(lambda x: x[index], states))
            outcome = board.outcome(claim_draw=True)
            capped = outcome is None
            score = .5 if capped or outcome.winner is None else float(outcome.winner == (row['color'] == 'white'))
            ledger['games'].append(dict(row, **record_board(board), result=outcome.result() if outcome else '*',
                score=score, capped=capped, plies=int(ply), coverage_hits=int(coverage_hits[index]),
                fitness_adjudication='neutral-cap' if capped else 'rules'))
        atomic_json(ledger_path, ledger)
        pending_path.unlink(missing_ok=True)
        completed = {g['task_id'] for g in ledger['games']}
        remaining = [t for t in tasks if t['task_id'] not in completed]
    return ledger
