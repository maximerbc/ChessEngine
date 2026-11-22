"""Additional measured forecast inputs; no benchmark model is promoted."""
import json
import math
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ..data import records
from ..expand_data import label_chunk
from .evaluation import cpu_matches
from .network import initialize, apply, export
from .protocol import freeze, sha, atomic_json, identity, LEGACY_SHA, nvidia_available, save_state


def measure(config, tournament_path, positions, teacher_path, binary, openings_path,
            legacy_supervised, directory, workers=1):
    directory = Path(directory)
    protocol = freeze(directory, dict(config=config,workers=workers),
        [tournament_path,positions,teacher_path,openings_path,legacy_supervised,binary])
    result_path = directory / 'measurement.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    tournament = json.loads(Path(tournament_path).read_text())
    if tournament.get('workload_sha256') != identity(config):
        raise ValueError('Tournament benchmark belongs to another workload')
    if tournament.get('measured_on_nvidia') and not nvidia_available():
        raise ValueError('Training component timings must run on the NVIDIA backend used for the forecast')
    selected = tournament.get('selected_simulations')
    if tournament.get('smoke') or selected not in [32,64,128]:
        raise ValueError('Complete qualifying tournament measurements are required')
    gpu = next(r for r in tournament['results'] if r['simulations'] == selected)
    teacher = json.loads(Path(teacher_path).read_text())
    if sha(binary) != teacher['binary_sha256'] or any(sha(p) != h for p,h in teacher['nnue'].items()):
        raise ValueError('Teacher identity changed')
    if sha(legacy_supervised) != LEGACY_SHA:
        raise ValueError('CPU reference must be the preserved 80k release')
    sample = []
    for row in records(positions):
        if row['split'] == 'train':
            sample.append(row)
            if len(sample) == 1024:
                break
    if len(sample) != 1024:
        raise ValueError('Labeling benchmark needs 1,024 training-only positions')
    labeled = label_chunk(sample,directory/'teacher-sample.jsonl',binary,teacher)
    # Synthetic arrays exercise the exact architecture/optimizer and transfer path.
    # They are not study examples, and benchmark weights are never study checkpoints.
    rng = np.random.default_rng(20261121)
    batch = config['supervised']['batch_size']
    x = rng.normal(0,.1,(batch,853)).astype('float32')
    y = rng.uniform(-1,1,batch).astype('float32')
    params = initialize(jax.random.PRNGKey(20261121))
    optimizer = optax.adamw(.001,weight_decay=.0001)
    state = optimizer.init(params)

    @jax.jit
    def update(p,o,inputs,targets):
        gradient = jax.grad(lambda weights:jnp.mean((apply(weights,inputs)-targets)**2))(p)
        changes,o = optimizer.update(gradient,o,p)
        return optax.apply_updates(p,changes),o

    started = time.perf_counter()
    params,state = update(params,state,x,y)
    jax.block_until_ready(params)
    compilation = time.perf_counter()-started
    started = time.perf_counter()
    best_params = initialize(jax.random.PRNGKey(20261121))
    for _ in range(50):
        params,state = update(params,state,jnp.asarray(x.copy()),jnp.asarray(y.copy()))
        jax.block_until_ready(params)
        save_state(directory/'batch-checkpoint.pkl',jax.device_get(dict(params=params,optimizer=state,
                   best_params=best_params,rng=rng.bit_generator.state)),protocol)
    per_batch = (time.perf_counter()-started)/50
    candidate = directory/'cpu-reference-random.npz'
    export(initialize(jax.random.PRNGKey(20261121)),candidate,'evolutionary','benchmark-only')
    openings = json.loads(Path(openings_path).read_text())['pilot'][:config['search']['benchmark_pairs']]
    cpu_games = 0
    cpu_seconds = 0.
    for name,opponent in [('classical','handcrafted'),('supervised',legacy_supervised)]:
        ledger = cpu_matches(str(candidate),opponent,openings,directory/f'cpu-{name}.json',
            config['evolution']['cpu_match_time_ms'],selected,config['evolution']['max_plies'],workers=workers)
        cpu_games += len(ledger['games'])
        cpu_seconds += ledger['active_seconds']
    # Charge validation as additional full update batches, conservatively.
    epoch_batches = sum(math.ceil(config['supervised']['counts'][s]/batch) for s in ['train','validation'])
    result = dict(protocol_id=protocol,workload_sha256=identity(config),
        measured_on_nvidia=tournament['measured_on_nvidia'],hardware=tournament['hardware'],
        tournament_games_per_second=gpu['tournament_games_per_second'],
        label_positions_per_second=1024/labeled['label_seconds'],
        supervised_seconds_per_epoch=per_batch*epoch_batches,
        cpu_evaluation_games_per_second=cpu_games/(cpu_seconds/workers),
        compilation_seconds=sum(r['compile_and_first_search_seconds'] for r in tournament['results'])+compilation,
        cpu_workers=workers,label_sample_positions=1024,training_batch_benchmark_steps=50,
        epoch_estimate_basis='measured synthetic update batches including checkpoint I/O plus a conservative update-equivalent validation allowance',
        limitations=['Feature-cache preparation and first-epoch disk I/O require additional measured allowance.',
            'CPU parallel throughput is estimated from summed per-game active time and configured workers; scheduling overhead is extra.',
            'Random-network game lengths may differ from trained opponents; update with the pilot before main evolution.'])
    atomic_json(result_path,result)
    return result
