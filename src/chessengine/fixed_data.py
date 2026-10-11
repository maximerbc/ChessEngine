"""Frozen-data optimization/representation study with a baseline reproduction gate."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import chess
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .common import digest, json_hash, provenance, read_json, write_json
from .diagnostics import development_rows, metrics, strata, grouped_bootstrap
from .features import encode, restore, position_key, transform_features
from .model import NumpyEvaluator
from .train import fit


def canonical_audit(rows):
    seen = {}
    for row in rows:
        board = chess.Board(row['fen'])
        key = position_key(board if board.turn else board.mirror())
        if key in seen and seen[key] != row['split']:
            raise ValueError('Canonical train/validation overlap')
        seen[key] = row['split']
    return {'canonical_cross_split_overlaps': 0, 'positions': len(rows)}


def summaries(runs, config):
    reference = {r['seed']: r for r in runs if r['name'] == 'reference'}
    base = np.mean([r['validation']['mse'] for r in reference.values()])
    result = []
    for name in dict.fromkeys(r['name'] for r in runs):
        group = [r for r in runs if r['name'] == name]
        errors = [r['validation']['mse'] for r in group]
        reduction = 1 - np.mean(errors) / base if base else 0
        improved = sum(r['validation']['mse'] < reference[r['seed']]['validation']['mse'] for r in group)
        result.append({'name': name, 'category': group[0]['category'], 'mean_mse': float(np.mean(errors)),
                       'seed_sd': float(np.std(errors, ddof=1)), 'relative_reduction': float(reduction),
                       'improving_seeds': improved, 'latency_us': float(np.mean([r['latency_us'] for r in group])),
                       'promising': bool(name != 'reference' and reduction >= config['minimum_reduction']
                       and improved >= config['minimum_seeds'])})
    return sorted(result, key=lambda r: (r['mean_mse'], r['latency_us'], r['name']))


def combination(summary, treatments):
    winners = []
    for category in ['optimization', 'representation']:
        eligible = [r for r in summary if r['category'] == category and r['promising']]
        if not eligible:
            return None
        winners.append(next(t for t in treatments if t['name'] == eligible[0]['name']))
    return {'name': 'combined', 'category': 'combination',
            'settings': {**winners[0]['settings'], **winners[1]['settings']},
            'parents': [t['name'] for t in winners]}


def predict(evaluator, x):
    return np.concatenate([evaluator.batch(x[i:i+2048]) for i in range(0, len(x), 2048)]).astype(float)


def run(config, resume=False):
    started = time.perf_counter()
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    old = read_json(Path(config['historical']) / 'results.json')
    prior = {r['seed']: r for r in old['runs'] if r['experiment'] == 'baseline' and r['size'] == 'all'}
    source_sha = digest(config['source'])
    if any(r['data_sha256'] != source_sha for r in prior.values()):
        raise ValueError('Historical dataset mismatch')
    identity = {'config': config, 'source_sha256': source_sha,
                'historical_results_sha256': digest(Path(config['historical']) / 'results.json'),
                'code': {name: digest(Path(__file__).with_name(name + '.py')) for name in
                         ['fixed_data', 'features', 'model', 'train', 'diagnostics', 'data', 'common', 'engine']}}
    protocol = out / 'protocol.json'
    if protocol.exists():
        if not resume or read_json(protocol)['identity'] != identity:
            raise ValueError('Study exists or protocol mismatch; resume requires identical inputs/code')
    else:
        write_json(protocol, {'identity': identity, 'provenance': provenance()})
    rows = development_rows(config['source'])
    audit = canonical_audit(rows)
    train_ids = [i for i, r in enumerate(rows) if r['split'] == 'train']
    val_ids = [i for i, r in enumerate(rows) if r['split'] == 'validation']
    if [len(train_ids), len(val_ids)] != config['expected_counts']:
        raise ValueError('Frozen split counts changed')
    print(f'Encoding {len(rows):,} fixed positions', flush=True)
    base = np.stack([encode(restore(r)) for r in rows])
    y = np.array([r['target'] for r in rows], dtype=np.float32)
    val_rows = [rows[i] for i in val_ids]
    groups = [strata(restore(r), r) for r in val_rows]
    latency_boards = [restore(r) for r in val_rows[:64]]
    games = np.array([r['game_id'] for r in val_rows])
    write_json(out / 'audit.json', {**audit, 'split_counts': config['expected_counts'],
                                  'training_records_sha256': json_hash([rows[i] for i in train_ids]),
                                  'validation_records_sha256': json_hash(val_rows)})
    completed = []
    reference_predictions = {}

    def experiment(treatment):
        name = treatment['name']
        version = treatment['settings'].get('feature_version', 1)
        xt = transform_features(base[train_ids], version)
        xv = transform_features(base[val_ids], version)
        prepared = {'source_sha256': source_sha, 'feature_version': version,
                    'subsets': {'train': (torch.from_numpy(xt), torch.from_numpy(y[train_ids])),
                                'validation': (torch.from_numpy(xv), torch.from_numpy(y[val_ids]))}}
        for seed in config['seeds']:
            training = {**config['training'], **treatment['settings'], 'seed': seed}
            directory = out / f'{name}-{seed}'
            result_path = directory / 'result.json'
            if result_path.exists():
                result = read_json(result_path)
                if (result['config'] != training or digest(directory / 'best.npz') != result['checkpoint_sha256']
                        or digest(directory / 'predictions.npz') != result['predictions_sha256']):
                    raise ValueError('Completed run mismatch')
                prediction = np.load(directory / 'predictions.npz')['prediction']
            else:
                print(f'Training {name}, seed {seed}', flush=True)
                tick = time.perf_counter()
                fit(config['source'], directory, training, resume=(directory / 'resume.pt').exists(), prepared=prepared)
                evaluator = NumpyEvaluator(directory / 'best.npz')
                with threadpool_limits(limits=1):
                    prediction = predict(evaluator, xv)
                    train_error = metrics(predict(evaluator, xt), y[train_ids])
                    for board in latency_boards:
                        evaluator(board)
                    times = []
                    for _ in range(3):
                        for board in latency_boards:
                            t = time.perf_counter_ns()
                            evaluator(board)
                            times.append((time.perf_counter_ns()-t)/1000)
                history = read_json(directory / 'metrics.json')['history']
                breakdown = {}
                for dimension in groups[0]:
                    breakdown[dimension] = {}
                    for label in sorted({g[dimension] for g in groups}):
                        ids = [i for i, g in enumerate(groups) if g[dimension] == label]
                        breakdown[dimension][label] = {'positions': len(ids), 'games': len(set(games[ids])),
                                                      'sparse': len(ids) < 100,
                                                      **metrics(prediction[ids], y[val_ids][ids])}
                np.savez_compressed(directory / 'predictions.npz', prediction=prediction, targets=y[val_ids], game_ids=games)
                result = {'name': name, 'category': treatment['category'], 'seed': seed, 'config': training,
                          'train': train_error, 'validation': metrics(prediction, y[val_ids]), 'breakdown': breakdown,
                          'selected_epoch': evaluator.metadata['epoch'], 'epochs': len(history),
                          'optimizer_steps': history[-1]['optimizer_steps'],
                          'training_seconds': history[-1]['update_seconds'], 'wall_seconds': time.perf_counter()-tick,
                          'latency_us': float(np.median(times)), 'checkpoint_sha256': digest(directory / 'best.npz'),
                          'predictions_sha256': digest(directory / 'predictions.npz'), 'provenance': provenance()}
                if name != 'reference':
                    result['bootstrap'] = grouped_bootstrap({'supervised': prediction, 'reference': reference_predictions[seed]},
                                                            y[val_ids].astype(float), games,
                                                            config['bootstrap_resamples'], config['bootstrap_seed'])
                write_json(result_path, result)
            if name == 'reference':
                old_path = Path(config['historical']) / f'baseline-all-seed-{seed}' / 'best.npz'
                if digest(old_path) != prior[seed]['checkpoint_sha256']:
                    raise ValueError('Historical checkpoint changed')
                a, b = NumpyEvaluator(directory / 'best.npz'), NumpyEvaluator(old_path)
                delta = max(float(np.max(abs(x-z))) for pair, other in zip(a.layers, b.layers) for x, z in zip(pair, other))
                if delta > config['reproduction_tolerance'] or abs(result['validation']['mse'] - prior[seed]['validation']['mse']) > config['reproduction_tolerance']:
                    write_json(out / 'reproduction-failure.json', {'seed': seed, 'max_weight_difference': delta})
                    raise ValueError('Historical reference reproduction failed; treatments blocked')
                reference_predictions[seed] = prediction
                write_json(out / f'reproduction-{seed}.json', {'max_weight_difference': delta, 'passed': True})
            completed.append(result)
            print(f"  validation={result['validation']['mse']:.6f}", flush=True)

    for treatment in config['treatments']:
        experiment(treatment)
    extra = combination(summaries(completed, config), config['treatments'])
    write_json(out / 'combination-decision.json', {'treatment': extra, 'initial_summary': summaries(completed, config)})
    if extra:
        experiment(extra)
    result = {'protocol': read_json(protocol), 'runs': completed, 'summary': summaries(completed, config),
              'combination': extra, 'wall_seconds': time.perf_counter()-started,
              'training_seconds': sum(r['training_seconds'] for r in completed), 'paid_cost_cad': 0}
    write_json(out / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(read_json(args.config), args.resume)


if __name__ == '__main__':
    main()
