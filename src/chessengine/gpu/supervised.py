"""AdamW from scratch; validation selects epochs, final test stays sealed."""
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from .network import initialize, apply, export
from .protocol import freeze, load_state, save_state, atomic_json, sha


def fit(config, data, output, seed, max_batches=None):
    data, output = Path(data), Path(output)
    if seed not in config['seeds']:
        raise ValueError('Seed is not in the frozen study')
    inputs = [data / 'features.json', data / 'labels.json']
    manifest = json.loads((data / 'features.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(data / name) != expected:
            raise ValueError('Feature cache changed')
    protocol = freeze(output, dict(config=config, seed=seed), inputs)
    train_x, train_y, val_x, val_y = [np.load(data / name, mmap_mode='r') for name in
        ['train-x.npy', 'train-y.npy', 'validation-x.npy', 'validation-y.npy']]
    if len(train_y) != config['counts']['train'] or len(val_y) != config['counts']['validation']:
        raise ValueError('Dataset split counts disagree with the frozen protocol')
    optimizer = optax.adamw(config['learning_rate'], weight_decay=config['weight_decay'])
    state_path = output / 'optimizer.pkl'
    if state_path.exists():
        state = load_state(state_path, protocol)
    else:
        params = initialize(jax.random.PRNGKey(seed))
        state = dict(params=params, optimizer=optimizer.init(params), epoch=0, batch=0,
            epoch_rng=None, rng=np.random.default_rng(seed).bit_generator.state, best=float('inf'),
            best_params=params, stale=0, history=[], elapsed_seconds=0., complete=False)

    @jax.jit
    def update(params, opt, x, y, weight):
        def loss(p):
            error = (apply(p, x) - y) ** 2
            return (error * weight).sum() / weight.sum()
        loss_value, gradient = jax.value_and_grad(loss)(params)
        changes, opt = optimizer.update(gradient, opt, params)
        return optax.apply_updates(params, changes), opt, loss_value

    predict = jax.jit(apply)
    started, batches = time.perf_counter(), 0
    batch_size = config['batch_size']
    rng = np.random.default_rng()
    rng.bit_generator.state = state['rng']
    while state['epoch'] < config['epochs'] and state['stale'] < config['patience'] and not state['complete']:
        if state['epoch_rng'] is None:
            state['epoch_rng'] = state['rng']
        rng.bit_generator.state = state['epoch_rng']
        order = rng.permutation(len(train_y))
        state['rng'] = rng.bit_generator.state
        for offset in range(state['batch'], len(train_y), batch_size):
            index = order[offset:offset + batch_size]
            n = len(index)
            # Padding has zero loss weight; all real positions appear exactly once.
            index = np.pad(index, (0, batch_size - n), mode='edge')
            weight = np.arange(batch_size) < n
            state['params'], state['optimizer'], _ = update(state['params'], state['optimizer'],
                jnp.asarray(train_x[index]), jnp.asarray(train_y[index]), jnp.asarray(weight))
            jax.block_until_ready(state['params'])
            state['batch'] = offset + n
            state['elapsed_seconds'] += time.perf_counter() - started
            started = time.perf_counter()
            save_state(state_path, jax.device_get(state), protocol)
            batches += 1
            if max_batches is not None and batches >= max_batches:
                return state
        squared = 0.
        for offset in range(0, len(val_y), batch_size):
            predicted = np.asarray(predict(state['params'], jnp.asarray(val_x[offset:offset + batch_size])))
            squared += float(np.sum((predicted - val_y[offset:offset + batch_size]) ** 2, dtype=np.float64))
        mse = squared / len(val_y)
        state['history'].append(dict(epoch=state['epoch'] + 1, validation_mse=mse))
        if mse < state['best']:
            state.update(best=mse, best_params=state['params'], stale=0)
            export(state['best_params'], output / 'best.npz', 'supervised', 'gpu-supervised-development',
                   dict(seed=seed, protocol_id=protocol, training_positions=len(train_y)))
        else:
            state['stale'] += 1
        state.update(epoch=state['epoch'] + 1, batch=0, epoch_rng=None)
        save_state(state_path, jax.device_get(state), protocol)
        atomic_json(output / 'history.json', state['history'])
    state['complete'] = True
    save_state(state_path, jax.device_get(state), protocol)
    atomic_json(output / 'selection.json', dict(seed=seed, validation_mse=state['best'],
        epochs=state['epoch'], history=state['history'], elapsed_seconds=state['elapsed_seconds'],
        representative=seed == 42, final_test_opened=False, protocol_id=protocol,
        checkpoint_sha256=sha(output / 'best.npz')))
    return state


def final_test(config, data, selections, output):
    from .network import import_numpy
    from .protocol import sha
    data = Path(data)
    seal = json.loads(Path(selections).read_text())
    if seal.get('seeds') != config['seeds']:
        raise ValueError('All development selections must be frozen before final testing')
    marker = data / 'final-test-selection.json'
    assessment = {'selection_sha256':sha(selections)}
    if marker.exists() and json.loads(marker.read_text()) != assessment:
        raise ValueError('Final test already opened for a different frozen selection')
    atomic_json(marker,assessment)
    x, y = np.load(data / 'test-x.npy', mmap_mode='r'), np.load(data / 'test-y.npy', mmap_mode='r')
    if len(y) != config['counts']['test']:
        raise ValueError('Final split count mismatch')
    rows = []
    for row in seal['models']:
        if sha(row['file']) != row['sha256']:
            raise ValueError('Selected checkpoint changed')
        predict = jax.jit(apply)
        params = import_numpy(row['file'])
        error = 0.
        for start in range(0, len(y), config['batch_size']):
            stop = start + config['batch_size']
            error += float(np.sum((np.asarray(predict(params, x[start:stop])) - y[start:stop]) ** 2, dtype=np.float64))
        rows.append(dict(seed=row['seed'], mse=error / len(y), checkpoint_sha256=row['sha256']))
    atomic_json(output, dict(final_position_assessment=rows, samples=len(y), selection_sha256=sha(selections)))
    return rows
