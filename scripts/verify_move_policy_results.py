"""Verify downloaded direct-policy checkpoints against the sealed result protocol."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from chessengine.common import digest, read_json, write_json
from chessengine.move_policy import NumpyMovePolicy
from chessengine.move_policy_data import ShardDataset
from chessengine.move_policy_train import load_export_to_torch


def verify(root):
    root = Path(root)
    protocol = read_json(root / 'test-protocol.json')
    results = read_json(root / 'test-results.json')
    if results['protocol'] != protocol or digest(root / 'dataset.json') != protocol['dataset_sha256']:
        raise ValueError('Sealed result/data protocol mismatch')
    torch.set_num_threads(1)
    report = []
    for seed in protocol['config']['seeds']:
        path = root / f'seed-{seed}/best.npz'
        if digest(path) != protocol['checkpoints'][str(seed)]:
            raise ValueError('Selected checkpoint checksum mismatch')
        policy = NumpyMovePolicy(path)
        model, metadata = load_export_to_torch(path, torch.device('cpu'))
        model.eval()
        maximum_error = 0.0
        count = 0
        for split in ['train', 'validation', 'test']:
            data = ShardDataset(root, split)
            for index in range(min(8, len(data))):
                x, moves, _, _ = data[index]
                with torch.inference_mode():
                    expected = model(torch.from_numpy(x[None]), torch.from_numpy(moves[None]),
                                     torch.ones((1, len(moves)), dtype=torch.bool))[0].numpy()
                actual = policy.logits(x, moves)
                np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=1e-4)
                maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
                count += 1
        selected = read_json(root / f'seed-{seed}/metrics.json')
        if not selected['complete'] or selected['best_sha256'] != policy.checkpoint_id:
            raise ValueError('Incomplete or inconsistent selection')
        report.append({'seed': seed, 'selected_epoch': metadata['epoch'],
                       'checkpoint_sha256': policy.checkpoint_id, 'parity_positions': count,
                       'maximum_logit_error': maximum_error, 'cpu_numpy_export_verified': True})
    return {'verified_unix': time.time(), 'seeds': report, 'all_selected_checkpoints_verified': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = verify(args.root)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
