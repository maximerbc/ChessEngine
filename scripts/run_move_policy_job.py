"""Detached full training job; its independent Pod deadline guard runs separately."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('runs/stockfish-move-policy')
    approved = {'train': 800000, 'validation': 100000, 'test': 100000}
    config = json.loads(Path('configs/stockfish-move-policy.json').read_text())
    dataset = json.loads((root / 'dataset.json').read_text())
    if config['positions'] != approved or dataset['counts'] != approved:
        raise RuntimeError('This authorized run requires exactly 800000/100000/100000 positions')
    root.mkdir(parents=True, exist_ok=True)
    receipt = {'started_unix': time.time(), 'status': 'running',
               'deadline_unix': float(os.environ['CHESS_POLICY_DEADLINE_UNIX']),
               'pod_id': os.environ.get('RUNPOD_POD_ID'), 'separate_pilot': False}
    path = root / 'cloud-job.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    result = subprocess.run(['bash', 'scripts/run_move_policy.sh', *sys.argv[1:]])
    receipt.update(status='complete' if result.returncode == 0 else 'stopped',
                   returncode=result.returncode, completed_unix=time.time())
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
