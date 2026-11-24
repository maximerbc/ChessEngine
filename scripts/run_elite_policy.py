"""Run only an explicitly approved, bounded GPU job. No infrastructure creation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from chessengine.common import digest, read_json
from chessengine.elite_data import atomic_json, preflight
from chessengine.elite_train import train, evaluate_all, benchmark


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['benchmark','train'])
    p.add_argument('--root', default='runs/elite-policy')
    p.add_argument('--approval', required=True)
    a = p.parse_args()
    root = Path(a.root)
    approval = read_json(a.approval)
    stamp = read_json('source-revision.json')
    config = preflight(root)['config']
    if (approval.get('approved') is not True or approval.get('study') != config['study']
            or a.action not in approval.get('actions', [])
            or approval.get('dataset_sha256') != digest(root / 'dataset.json')
            or approval.get('source_files') != stamp['files']
            or not approval.get('confirmation_reference')
            or approval.get('deadline_unix', 0) <= time.time() + 90
            or approval.get('cumulative_forecast_cad', float('inf')) > 40):
        raise ValueError('Explicit approval, exact inputs/source and bounded budget are required')
    for path, sha in stamp['files'].items():
        if digest(path) != sha:
            raise ValueError('Packaged source changed after approval')
    os.environ['ELITE_DEADLINE_UNIX'] = str(approval['deadline_unix'])
    guard_status = root / 'guard.json'
    if guard_status.exists():
        guard_status.unlink()  # Never accept an old benchmark's armed receipt as a fresh guard proof.
    guard = subprocess.Popen([sys.executable, 'scripts/runpod_policy_guard.py',
        '--deadline-unix', str(approval['deadline_unix']), '--output', str(guard_status),
        '--backup-verified-file', str(root / 'backup-verified.json')], stdout=subprocess.DEVNULL)
    # Require the independent guard to prove it is armed before any gradient update.
    for _ in range(50):
        if guard.poll() is not None:
            raise RuntimeError('Independent deadline guard exited before training')
        if guard_status.exists():
            proof = read_json(guard_status)
            if (proof.get('status') == 'armed' and proof.get('pod_id') == os.environ.get('RUNPOD_POD_ID')
                    and proof.get('deadline_unix') == approval['deadline_unix']):
                break
        time.sleep(.1)
    else:
        raise RuntimeError('Independent shutdown guard failed to arm')
    if a.action == 'benchmark':
        result = benchmark(config, root)
        atomic_json(root / 'benchmark.json', result)
        print(json.dumps(result, indent=2))
        return  # The guard remains independent; operator must stop/approve fitting before its deadline.
    measured = read_json(root / 'benchmark.json')
    # Conservative fit budget includes validation and setup overhead; never silently shrink epochs/seeds.
    forecast = measured['forecast_training_seconds'] * 1.5 + 900
    if time.time() + forecast > approval['deadline_unix'] - 60:
        raise ValueError('Measured fixed study exceeds approved deadline; no fitting started')
    for seed in config['seeds']:
        path = root / f'seed-{seed}' / 'resume.pt'
        if (path.parent / 'failure.json').exists():
            continue
        try:
            result = train(config, root, seed, requested='cuda', resume=path.exists())
        except Exception:
            if (path.parent / 'failure.json').exists():
                continue  # Preserve failed seed, complete independent preregistered attempts.
            raise
        if result['epoch'] < config['training']['epochs']:
            raise RuntimeError('Study paused at budget deadline; artifacts retained')
    evaluate_all(config, root, 'cuda')
    atomic_json(root / 'job-complete.json', dict(status='training-and-evaluation-complete', unix=time.time()))
    # The guard stops at its deadline or when the external operator writes the verified-backup marker.


if __name__ == '__main__':
    main()
