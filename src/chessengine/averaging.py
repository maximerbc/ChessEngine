"""Bounded supervised prediction and weight averaging study."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .common import digest, read_json, write_json, provenance
from .diagnostics import development_rows, metrics, strata, grouped_bootstrap
from .features import encode, restore
from .fixed_data import canonical_audit, predict
from .model import NumpyEvaluator
from .train import fit


def ensemble_predictions(predictions):
    return np.stack(predictions).mean(axis=0)


def paired_mean_member(prediction, members, targets, games, resamples, seed):
    """Difference from mean member losses, not loss of the mean prediction."""
    _, group = np.unique(games, return_inverse=True)
    counts = np.bincount(group)
    differences = []
    for power in [2, 1]:
        delta = abs(prediction - targets)**power - np.mean(abs(members - targets)**power, axis=0)
        sums = np.bincount(group, weights=delta)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(resamples):
            ids = rng.integers(len(counts), size=len(counts))
            draws.append(sums[ids].sum() / counts[ids].sum())
        differences.append(np.quantile(draws, [.025, .975]).tolist())
    return dict(mse=differences[0], mae=differences[1])


def latency(evaluators, boards):
    for b in boards:
        for e in evaluators:
            e(b)
    timings = []
    for _ in range(3):
        for b in boards:
            start = time.perf_counter_ns()
            sum(e(b) for e in evaluators) / len(evaluators)
            timings.append((time.perf_counter_ns() - start) / 1000)
    return float(np.median(timings))


def run(config, resume=False):
    start = time.perf_counter()
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=True)
    previous = read_json(Path(config['previous']) / 'results.json')
    prior = {r['seed']: r for r in previous['runs'] if r['name'] == 'combined'}
    identity = {'config': config, 'source_sha256': digest(config['source']),
                'previous_sha256': digest(Path(config['previous']) / 'results.json'),
                'code': {n: digest(Path(__file__).with_name(n + '.py')) for n in
                         ['averaging', 'train', 'features', 'model', 'diagnostics', 'fixed_data', 'data', 'common']}}
    if identity['source_sha256'] != previous['protocol']['identity']['source_sha256']:
        raise ValueError('Historical dataset mismatch')
    path = root / 'protocol.json'
    if path.exists():
        if not resume or read_json(path)['identity'] != identity:
            raise ValueError('Study protocol mismatch')
    else:
        write_json(path, {'identity': identity, 'provenance': provenance()})
    print('Auditing and encoding fixed development data', flush=True)
    rows = development_rows(config['source'])
    audit = canonical_audit(rows)
    selected = {s: [r for r in rows if r['split'] == s] for s in ['train', 'validation']}
    if [len(selected[s]) for s in selected] != config['expected_counts']:
        raise ValueError('Split counts changed')
    subsets = {s: (torch.from_numpy(np.stack([encode(restore(r), 3) for r in rr])),
                   torch.tensor([r['target'] for r in rr], dtype=torch.float32))
               for s, rr in selected.items()}
    prepared = dict(source_sha256=identity['source_sha256'], feature_version=3, subsets=subsets)
    xv, yv = [t.numpy() for t in subsets['validation']]
    games = np.array([r['game_id'] for r in selected['validation']])
    groups = [strata(restore(r), r) for r in selected['validation']]
    boards = [restore(r) for r in selected['validation'][:64]]
    predictions, entries, evaluators = {}, [], []
    def describe(name, pred, models):
        breakdown = {}
        for dimension in groups[0]:
            breakdown[dimension] = {}
            for label in sorted({g[dimension] for g in groups}):
                ids = [i for i, g in enumerate(groups) if g[dimension] == label]
                breakdown[dimension][label] = dict(positions=len(ids), games=len(set(games[ids])),
                    sparse=len(ids) < 100, **metrics(pred[ids], yv[ids]))
        return dict(name=name, **metrics(pred, yv), latency_us=latency(models, boards), breakdown=breakdown)
    with threadpool_limits(limits=1):
        for seed in config['seeds']:
            old_dir = Path(config['previous']) / f'combined-{seed}'
            if digest(old_dir / 'best.npz') != prior[seed]['checkpoint_sha256']:
                raise ValueError('Historical checkpoint mismatch')
            original = NumpyEvaluator(old_dir / 'best.npz')
            original_pred = predict(original, xv)
            np.testing.assert_allclose(metrics(original_pred, yv)['mse'], prior[seed]['validation']['mse'], atol=config['tolerance'], rtol=0)
            evaluators.append(original)
            predictions[f'reference-{seed}'] = original_pred
            entries.append(describe(f'reference-{seed}', original_pred, [original]))
            cfg = {**prior[seed]['config'], 'average_start_epoch': config['average_start_epoch']}
            directory = root / f'seed-{seed}'
            print(f'Weight averaging seed {seed}', flush=True)
            fit(config['source'], directory, cfg, resume=(directory / 'resume.pt').exists(), prepared=prepared)
            # Verify entire ordinary loss trajectory and final/best weights.
            old_history = read_json(old_dir / 'metrics.json')['history']
            new_history = read_json(directory / 'metrics.json')['history']
            assert len(new_history) == len(old_history), 'Ordinary stopping trajectory changed'
            delta = 0.
            for filename in ['best.npz', 'last.npz']:
                a, b = NumpyEvaluator(old_dir / filename), NumpyEvaluator(directory / filename)
                delta = max(delta, max(float(np.max(abs(x-y))) for p, q in zip(a.layers, b.layers) for x,y in zip(p,q)))
            for a,b in zip(old_history, new_history):
                for key in ['train_loss', 'validation_loss']:
                    delta = max(delta, abs(a[key]-b[key]))
            if delta > config['tolerance']:
                raise ValueError('Ordinary trajectory reproduction failed')
            average = NumpyEvaluator(directory / 'average-best.npz')
            pred = predict(average, xv)
            predictions[f'average-{seed}'] = pred
            item = describe(f'average-{seed}', pred, [average])
            item.update(seed=seed, checkpoint=str(directory / 'average-best.npz'),
                        checkpoint_sha256=digest(directory / 'average-best.npz'),
                        ordinary_reproduction_delta=delta, history=new_history,
                        train=metrics(predict(average, subsets['train'][0].numpy()), subsets['train'][1].numpy()))
            item['bootstrap'] = grouped_bootstrap({'supervised': pred, 'reference': original_pred}, yv, games,
                                                   config['resamples'], config['bootstrap_seed'])
            entries.append(item)
        members = np.stack([predictions[f'reference-{s}'] for s in config['seeds']])
        ensemble = ensemble_predictions(members)
        predictions['ensemble'] = ensemble
        item = describe('ensemble', ensemble, evaluators)
        item['bootstrap'] = grouped_bootstrap({'supervised': ensemble, **{str(s): members[i] for i,s in enumerate(config['seeds'])}},
                                               yv, games, config['resamples'], config['bootstrap_seed'])
        item['versus_mean_member_ci95'] = paired_mean_member(ensemble, members, yv, games, config['resamples'], config['bootstrap_seed'])
        entries.append(item)
    baseline = float(np.mean([e['mse'] for e in entries if e['name'].startswith('reference')]))
    averaged = [e for e in entries if e['name'].startswith('average')]
    mean = float(np.mean([e['mse'] for e in averaged]))
    improved = sum(e['mse'] < metrics(predictions[f"reference-{e['seed']}"], yv)['mse'] for e in averaged)
    adopt = bool(1-mean/baseline >= .02 and improved >= 2)
    checkpoint = root / 'seed-42/average-best.npz' if adopt else Path(config['previous']) / 'combined-42/best.npz'
    np.savez_compressed(root / 'predictions.npz', **predictions, targets=yv, game_ids=games)
    result = dict(protocol=read_json(path), audit=audit, entries=entries, reference_mean_mse=baseline,
                  averaged_mean_mse=mean, relative_reduction=1-mean/baseline, improving_seeds=improved,
                  adopt_averaging=adopt, rl_initial_checkpoint=str(checkpoint), rl_initial_sha256=digest(checkpoint),
                  wall_seconds=time.perf_counter()-start, paid_cost_cad=0,
                  predictions_sha256=digest(root/'predictions.npz'))
    write_json(root/'results.json', result)
    print({k:result[k] for k in ['reference_mean_mse','averaged_mean_mse','adopt_averaging','rl_initial_checkpoint']}, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    run(read_json(a.config), a.resume)


if __name__ == '__main__':
    main()
