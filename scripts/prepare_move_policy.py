"""Run the authorized full local data stages; never rents GPU resources."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output', default='runs/stockfish-move-policy')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    previous = root / 'local-execution.json'
    receipt = json.loads(previous.read_text()) if previous.exists() else {
        'started_unix': time.time(), 'stages': [], 'paid_cost_cad': 0}
    for stage in ['prepare', 'label', 'encode']:
        start = time.time()
        result = subprocess.run([sys.executable, '-u', '-m', 'chessengine.move_policy_train', stage,
                                 '--input-root', args.input_root, '--output', args.output])
        receipt['stages'].append({'stage': stage, 'seconds': time.time() - start,
                                 'returncode': result.returncode})
        (root / 'local-execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if result.returncode:
            raise SystemExit(result.returncode)
    receipt['completed_unix'] = time.time()
    (root / 'local-execution.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
