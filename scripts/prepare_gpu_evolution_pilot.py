"""Prepare synthetic legal-history pilot openings without consulting any labels.

These are throughput-calibration openings, not a new final-strength pool. Record
and exclude all available supervised positions and reserved study openings.
"""
import argparse
import json
from pathlib import Path
import random
import sys

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from chessengine.features import position_key, record_board, restore
from chessengine.gpu.protocol import atomic_json, identity, sha


def prepare(root, output, config):
    root, output = Path(root), Path(output)
    excluded = set()
    inputs = {}
    datasets = ['data/main-positions.jsonl', 'data/pilot-positions.jsonl',
                'data/main-labeled.jsonl', 'data/pilot-labeled.jsonl',
                'data/expanded-200k-labeled.jsonl', 'data/gpu-value-v1/positions.jsonl']
    for name in datasets:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'Required historical position exclusion: {path}')
        inputs[name] = sha(path)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                excluded.add(position_key(chess.Board(row['fen'])))
        print(f'Excluded {name}; {len(excluded)} unique positions', flush=True)
    reserved = [root/'configs/openings.json']
    for folder in ['runs/evolutionary', 'runs/rl-improvement', 'runs/move-ranking', 'runs/policy-value',
                   'runs/augmented-evaluation-20261122', 'runs/evolutionary-augmented-20261122']:
        reserved.extend((root/folder).rglob('*openings*.json'))

    def collect(value):
        if isinstance(value, dict):
            if 'fen' in value:
                excluded.add(position_key(restore(value)))
            else:
                for v in value.values():
                    collect(v)
        elif isinstance(value, list):
            for v in value:
                collect(v)

    for path in sorted(set(reserved)):
        if path.is_file():
            inputs[str(path.relative_to(root))] = sha(path)
            collect(json.loads(path.read_text()))
    rng = random.Random(config['opening_seed'])
    quotas = dict(coverage=config['coverage_positions'], benchmark=config['benchmark_opening_pairs'],
                  pilot_train=config['pilot_opening_pairs'])
    pools = {k: [] for k in quotas}
    while any(len(pools[k]) < quotas[k] for k in quotas):
        board = chess.Board()
        for _ in range(rng.randrange(16, 49)):
            if board.is_game_over(claim_draw=True):
                break
            moves = sorted(board.legal_moves, key=lambda m: m.uci())
            board.push(rng.choice(moves))
        key = position_key(board)
        if board.is_game_over(claim_draw=True) or key in excluded:
            continue
        group = next(k for k in quotas if len(pools[k]) < quotas[k])
        row = record_board(board)
        row.update(id=identity(row), source_game_id='synthetic-' + identity(row),
                   source='seeded-legal-random-pilot-only')
        pools[group].append(row)
        excluded.add(key)
    atomic_json(output, pools)
    record = dict(schema=1, method='seeded legal random histories; no target values read or used',
        config_sha256=identity(config), input_sha256=inputs, openings_sha256=sha(output),
        counts={k:len(v) for k,v in pools.items()}, seed=config['opening_seed'],
        final_pools_opened_for_play=False, final_position_labels_used=False,
        limitation='Synthetic openings calibrate throughput; they do not establish strength or main-study game lengths.')
    atomic_json(output.with_suffix('.manifest.json'), record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/evolutionary-gpu-native.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.historical_root, args.output, json.loads(Path(args.config).read_text())), indent=2))
