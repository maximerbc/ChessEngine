"""Timing-only main-population shapes; no tournament fitness or learning."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms import Sep_CMA_ES
from chessengine.features import restore
from chessengine.gpu import rules, native_games
from chessengine.gpu.network import initialize, import_numpy
from chessengine.gpu.protocol import freeze, atomic_json, save_state, nvidia_available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--openings', required=True)
    parser.add_argument('--supervised', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deadline-unix', required=True, type=float)
    args = parser.parse_args()
    if not nvidia_available():
        raise RuntimeError('NVIDIA timing required')
    config = json.loads(Path(args.config).read_text())
    selection = json.loads(Path(args.selection).read_text())
    if not selection['passed']:
        raise ValueError('Coverage selection must pass first')
    output = Path(args.output)
    batches = [384, 768, 1536, 3072]
    protocol = freeze(output, dict(config=config, purpose='timing-only-main-shapes', batches=batches),
        [args.config, args.selection, args.openings, args.supervised, __file__])
    key = jax.random.PRNGKey(2026112202)
    mean = initialize(key)
    strategy = Sep_CMA_ES(64, mean)
    settings = strategy.default_params.replace(std_init=config['sigma'])
    state = strategy.init(key, mean, settings)
    population, _ = strategy.ask(jax.random.fold_in(key, 1), state, settings)
    bank = jax.tree.map(lambda a, b, c: jnp.concatenate([a, b[None], c[None]], 0),
                        population, import_numpy(args.supervised), mean)
    pool = json.loads(Path(args.openings).read_text())['coverage']
    roots = [rules.from_board(restore(row)) for row in pool]
    root_bank = jax.tree.map(lambda *x: jnp.stack(x), *roots)
    rows = []
    advance = native_games.rollout(selection['simulations'], selection['chunk_plies'], config['max_plies'])
    for batch in batches:
        if time.time() >= args.deadline_unix:
            break
        index = jnp.arange(batch, dtype=jnp.int32)
        states = jax.tree.map(lambda x: x[index % len(roots)], root_bank)
        kinds = jnp.where(index % 3 == 0, 0, 1)
        opponents = jnp.where(index % 3 == 1, 64, 65)
        values = (bank, states, index % 64, opponents, index % 2 == 0, kinds,
                  jnp.ones(batch, bool), jnp.zeros(batch, jnp.int32))
        started = time.perf_counter()
        result = advance(*values)
        jax.block_until_ready(result)
        cold = time.perf_counter() - started
        times = []
        for _ in range(2):
            if time.time() >= args.deadline_unix:
                break
            started = time.perf_counter()
            result = advance(*values)
            jax.block_until_ready(result)
            save_state(output/'layout-checkpoint.pkl', jax.device_get(result), protocol)
            times.append(time.perf_counter() - started)
        if not times:
            break
        if np.asarray(result['invalid']).any() or np.asarray(result['overflow']).any():
            raise ValueError('Timing probe violated search invariants')
        moves = int((np.asarray(result['actions']) >= 0).sum())
        rows.append(dict(batch_size=batch, simulations=selection['simulations'],
            chunk_plies=selection['chunk_plies'], compile_and_first_call_seconds=cold,
            median_seconds=float(np.median(times)), moves=moves,
            moves_per_second=moves/float(np.median(times)), includes_checkpoint_io=True,
            memory_stats={k:int(v) for k,v in (jax.devices()[0].memory_stats() or {}).items()}))
        atomic_json(output/'measurement.json', dict(protocol_id=protocol, rows=rows,
            model_bank_size=66, completed_training_games=0, fitness_updates=0,
            evidence='synthetic-root throughput, not complete-game throughput',
            limitation='Larger-batch main-game speedup remains an extrapolation until complete tournaments are measured.'))
        print(json.dumps(rows[-1]), flush=True)


if __name__ == '__main__':
    main()
