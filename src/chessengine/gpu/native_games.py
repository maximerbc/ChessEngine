"""GPU-resident PUCT rollouts; independent CPU replay gates every fitness record.

Historical gpu.games is intentionally unchanged. One atomic continuation is the
authoritative transaction, including completed games, active slots and timing.
"""
from functools import lru_cache
from pathlib import Path
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..features import restore, record_board
from . import rules
from .network import import_numpy
from .protocol import freeze, identity, sha, save_state, load_state, atomic_json
from .native_search import make_search


class DeadlineReached(RuntimeError):
    """A saved, resumable partial run; never a completed pilot."""


@lru_cache(maxsize=32)
def rollout(simulations, chunk_plies, max_plies):
    """Static shapes, no callbacks, file writes or host reads inside the scan."""
    search = make_search(simulations)

    @jax.jit
    def advance(bank, states, candidate_ids, opponent_ids, candidate_white,
                opponent_kinds, enabled, plies):
        def move(carry, _):
            state, count = carry
            ended, _ = jax.vmap(rules.terminal)(state)
            active = enabled & ~ended & (count < max_plies)

            def play(_):
                acting = (state.game.color == 0) == candidate_white
                indices = jnp.where(acting, candidate_ids, opponent_ids)
                params = jax.tree.map(lambda x: x[indices], bank)
                kinds = jnp.where(acting, 1, opponent_kinds)
                result = search(params, state, kinds)
                actions = result['actions']
                legal = jnp.take_along_axis(state.game.legal_action_mask, actions[:, None], 1)[:, 0]
                invalid = active & (~legal | (result['visits'].sum(1) != simulations)
                                    | result['overflow'] | result['terminal'])
                after = jax.vmap(lambda s, a, go: jax.lax.cond(
                    go, lambda _: rules.step(s, a), lambda _: s, None))(state, actions, active)
                return after, jnp.where(active, actions, -1), invalid, active & (result['root_coverage'] >= .8)

            def idle(_):
                return state, jnp.full(enabled.shape, -1, jnp.int32), jnp.zeros_like(enabled), jnp.zeros_like(enabled)

            after, actions, invalid, coverage = jax.lax.cond(active.any(), play, idle, None)
            return (after, count + active.astype(jnp.int32)), (actions, invalid, coverage)

        (states, plies), (actions, invalid, coverage) = jax.lax.scan(
            move, (states, plies), None, length=chunk_plies)
        ended, _ = jax.vmap(rules.terminal)(states)
        return dict(states=states, plies=plies, actions=actions, invalid=invalid.any(0),
                    coverage=coverage.sum(0), finished=enabled & (ended | (plies >= max_plies)),
                    overflow=enabled & states.overflow)
    return advance


def replay(task, actions, state, max_plies, coverage):
    """Reject even an intermediate automatic draw that a GPU trajectory skipped."""
    board = restore(task['opening'])
    for action in actions:
        if board.is_game_over(claim_draw=True):
            raise ValueError('GPU continued beyond a rules termination')
        mapping = rules.action_map(board)
        if action not in mapping:
            raise ValueError('GPU action failed independent legal replay')
        board.push(mapping[action])
    rules.assert_parity(board, state)
    outcome = board.outcome(claim_draw=True)
    capped = outcome is None
    if capped and len(actions) != max_plies:
        raise ValueError('Unfinished game is not at the declared cap')
    score = .5 if capped or outcome.winner is None else float(outcome.winner == (task['color'] == 'white'))
    return dict(task, **record_board(board), result=outcome.result() if outcome else '*',
                score=score, capped=capped, plies=len(actions), coverage_hits=int(coverage),
                replay_verified=True, fitness_adjudication='neutral-cap' if capped else 'rules')


def run(tasks, directory, simulations, max_plies=512, batch_size=32,
        chunk_plies=8, deadline=None, stop_after_chunks=None):
    if not tasks or batch_size < 1 or chunk_plies < 1 or max_plies < 1:
        raise ValueError('Positive workload and execution dimensions required')
    batch_size = min(batch_size, len(tasks))
    ids = [t['task_id'] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate tournament task')
    paths = sorted({t[k] for t in tasks for k in ['candidate', 'opponent'] if t[k] != 'handcrafted'})
    for task in tasks:
        if sha(task['candidate']) != task['candidate_sha256']:
            raise ValueError('Candidate changed')
        if task['opponent'] != 'handcrafted' and sha(task['opponent']) != task['opponent_sha256']:
            raise ValueError('Opponent changed')
    output = Path(directory)
    protocol = freeze(output, dict(runner='gpu-native-chunk-v2', tasks=tasks,
        simulations=simulations, max_plies=max_plies, batch_size=batch_size, chunk_plies=chunk_plies), paths)
    state_path = output / 'continuation.pkl'
    if state_path.exists():
        saved = load_state(state_path, protocol)
    else:
        saved = dict(games=[], slots=[None] * batch_size, labels=[[] for _ in range(batch_size)],
            states=None, plies=np.zeros(batch_size, np.int32), coverage=np.zeros(batch_size, np.int32),
            next_task=0, chunks=0, active_seconds=0., rollout_seconds=0., verification_seconds=0.,
            checkpoint_seconds=0., batch_histogram={}, started_unix=time.time())
    if len({g['task_id'] for g in saved['games']}) != len(saved['games']):
        raise ValueError('Duplicate completed records')
    completed = {g['task_id'] for g in saved['games']}
    assigned = {t['task_id'] for t in saved['slots'] if t is not None}
    if completed & assigned or completed | assigned != set(ids[:saved['next_task']]):
        raise ValueError('Pending slots disagree with the completed ledger')
    params = [import_numpy(p) for p in paths]
    # Handcrafted rows use kind=0; their unused network still needs a valid bank ID.
    bank = jax.tree.map(lambda *x: jnp.stack(x), *params)
    lookup = {p: i for i, p in enumerate(paths)}
    roots = {}
    def root_state(opening):
        key = identity(opening)
        if key not in roots:
            roots[key] = rules.from_board(restore(opening))
        return roots[key]
    initial = root_state(tasks[0]['opening'])
    states = (jax.tree.map(jnp.asarray, saved['states']) if saved['states'] is not None else
              jax.tree.map(lambda x: jnp.stack([x] * batch_size), initial))
    advance = rollout(simulations, chunk_plies, max_plies)
    session_chunks = 0

    def persist():
        before = time.perf_counter()
        saved['states'] = jax.device_get(states)
        save_state(state_path, saved, protocol)
        # This is a derived view. A crash between these writes is repaired on resume.
        atomic_json(output / 'games.json', dict(protocol_id=protocol,
            **{k: v for k, v in saved.items() if k not in {'states', 'labels', 'slots', 'plies', 'coverage'}},
            active_games=sum(s is not None for s in saved['slots']), total_games=len(tasks)))
        saved['checkpoint_seconds'] += time.perf_counter() - before

    while len(saved['games']) < len(tasks):
        if deadline is not None and time.time() >= deadline:
            persist()
            raise DeadlineReached('Budget deadline reached; tournament saved without fabricated results')
        tick = time.perf_counter()
        for i, slot in enumerate(saved['slots']):
            if slot is None and saved['next_task'] < len(tasks):
                task = tasks[saved['next_task']]
                saved['next_task'] += 1
                saved['slots'][i] = task
                saved['labels'][i] = []
                saved['plies'][i] = saved['coverage'][i] = 0
                root = root_state(task['opening'])
                states = jax.tree.map(lambda x, y: x.at[i].set(y), states, root)
        enabled = np.array([s is not None for s in saved['slots']])
        active_ids = np.flatnonzero(enabled)
        # Static buckets avoid running a full generation's search for its last
        # few unfinished games. Unique padding indices make scatter lossless.
        buckets = sorted({min(batch_size, 2 ** i) for i in range(batch_size.bit_length()+1)})
        width = next(b for b in buckets if b >= len(active_ids))
        indices = np.concatenate([active_ids, np.flatnonzero(~enabled)[:width-len(active_ids)]])
        slots = [saved['slots'][i] if saved['slots'][i] is not None else tasks[0] for i in indices]
        active_states = jax.tree.map(lambda x: x[indices], states)
        candidate_ids = jnp.array([lookup[s['candidate']] for s in slots], jnp.int32)
        opponent_ids = jnp.array([lookup.get(s['opponent'], 0) for s in slots], jnp.int32)
        start = time.perf_counter()
        result = advance(bank, active_states, candidate_ids, opponent_ids,
            jnp.array([s['color'] == 'white' for s in slots]),
            jnp.array([0 if s['opponent'] == 'handcrafted' else 1 for s in slots], jnp.int32),
            jnp.asarray(enabled[indices]), jnp.asarray(saved['plies'][indices]))
        jax.block_until_ready(result)
        saved['rollout_seconds'] += time.perf_counter() - start
        if np.asarray(result['invalid']).any() or np.asarray(result['overflow']).any():
            raise ValueError('GPU rollout failed legal/budget/history checks; fitness rejected')
        states = jax.tree.map(lambda x, y: x.at[indices].set(y), states, result['states'])
        saved['plies'][indices] = np.asarray(result['plies'])
        saved['coverage'][indices] += np.asarray(result['coverage'])
        saved['batch_histogram'][str(width)] = saved['batch_histogram'].get(str(width), 0) + 1
        labels = np.asarray(result['actions'])
        for j, i in enumerate(indices[:len(active_ids)]):
            saved['labels'][i].extend(int(a) for a in labels[:, j] if a >= 0)
        before = time.perf_counter()
        for j in np.flatnonzero(np.asarray(result['finished'])):
            i = indices[j]
            row = replay(saved['slots'][i], saved['labels'][i],
                jax.tree.map(lambda x: x[i], states), max_plies, saved['coverage'][i])
            saved['games'].append(row)
            saved['slots'][i] = None
        saved['verification_seconds'] += time.perf_counter() - before
        saved['chunks'] += 1
        saved['active_seconds'] += time.perf_counter() - tick
        persist()
        session_chunks += 1
        if stop_after_chunks is not None and session_chunks >= stop_after_chunks:
            break
    persist()
    return json.loads((output / 'games.json').read_text())
