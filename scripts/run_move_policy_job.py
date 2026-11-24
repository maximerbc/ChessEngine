"""Detached full training job; its independent Pod deadline guard runs separately."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('runs/stockfish-move-policy')
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
