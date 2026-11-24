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


def compare_logits(actual, expected, strict=True):
    """Relative logit agreement alone can hide large softmax errors."""
    actual_centered = actual - actual.max()
    expected_centered = expected - expected.max()
    actual_probability = np.exp(actual_centered)
    expected_probability = np.exp(expected_centered)
    actual_probability /= actual_probability.sum()
    expected_probability /= expected_probability.sum()
    chosen_matches = int(np.argmax(actual)) == int(np.argmax(expected))
    raw_matches = np.allclose(actual, expected, atol=3e-5, rtol=1e-4)
    probability_matches = np.allclose(actual_probability, expected_probability, atol=3e-5, rtol=1e-4)
    if strict:
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=1e-4)
        np.testing.assert_allclose(actual_probability, expected_probability, atol=3e-5, rtol=1e-4)
        if not chosen_matches:
            raise ValueError('Exported policy chooses a different move')
    return {'raw': float(np.max(np.abs(actual - expected))),
            'centered': float(np.max(np.abs(actual_centered - expected_centered))),
            'probability': float(np.max(np.abs(actual_probability - expected_probability))),
            'chosen_matches': chosen_matches,
            'passed': bool(raw_matches and probability_matches and chosen_matches)}


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
        maximum_error = maximum_centered_error = maximum_probability_error = 0.0
        failures = []
        count = 0
        for split in ['train', 'validation', 'test']:
            data = ShardDataset(root, split)
            for index in range(min(8, len(data))):
                x, moves, _, _ = data[index]
                with torch.inference_mode():
                    expected = model(torch.from_numpy(x[None]), torch.from_numpy(moves[None]),
                                     torch.ones((1, len(moves)), dtype=torch.bool))[0].numpy()
                actual = policy.logits(x, moves)
                errors = compare_logits(actual, expected, strict=False)
                if not errors['passed']:
                    failures.append({'split': split, 'index': index, **errors})
                maximum_error = max(maximum_error, errors['raw'])
                maximum_centered_error = max(maximum_centered_error, errors['centered'])
                maximum_probability_error = max(maximum_probability_error, errors['probability'])
                count += 1
        selected = read_json(root / f'seed-{seed}/metrics.json')
        if not selected['complete'] or selected['best_sha256'] != policy.checkpoint_id:
            raise ValueError('Incomplete or inconsistent selection')
        report.append({'seed': seed, 'selected_epoch': metadata['epoch'],
                       'checkpoint_sha256': policy.checkpoint_id, 'parity_positions': count,
                       'maximum_logit_error': maximum_error, 'cpu_numpy_export_verified': not failures,
                       'failures': failures})
        report[-1].update(maximum_centered_logit_error=maximum_centered_error,
                          maximum_probability_error=maximum_probability_error,
                          chosen_moves_match=all(f['chosen_matches'] for f in failures))
    import chessengine.move_policy as inference
    passed = all(seed['cpu_numpy_export_verified'] for seed in report)
    return {'verified_unix': time.time(), 'seeds': report, 'all_checkpoint_hashes_verified': True,
            'all_selected_checkpoints_verified': passed, 'all_functional_exports_verified': passed,
            'verifier_sha256': digest(__file__), 'numpy_source_sha256': digest(inference.__file__)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = verify(args.root)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result['all_functional_exports_verified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
