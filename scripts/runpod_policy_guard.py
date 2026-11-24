"""Independent deadline guard for this training Pod; credentials stay in memory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--backup-verified-file', required=True)
    args = parser.parse_args()
    token = os.environ.get('RUNPOD_POD_API_KEY') or os.environ.get('RUNPOD_API_KEY')
    pod = os.environ.get('RUNPOD_POD_ID')
    if not token or not pod:
        raise RuntimeError('Provider-injected Pod stop credential is unavailable')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'status': 'armed', 'pod_id': pod,
                                 'deadline_unix': args.deadline_unix}) + '\n')
    while time.time() < args.deadline_unix and not Path(args.backup_verified_file).exists():
        time.sleep(min(5, max(0.1, args.deadline_unix - time.time())))
    subprocess.run(['sync'], timeout=10, check=False)
    for attempt in range(12):
        try:
            request = urllib.request.Request(f'https://api.runpod.io/v2/pods/{pod}/action',
                data=json.dumps({'action': 'stop'}).encode(),
                headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(request, timeout=20) as response:
                if response.status not in (200, 202, 204):
                    raise RuntimeError('Stop rejected')
            output.write_text(json.dumps({'status': 'stop-requested', 'pod_id': pod,
                                         'unix': time.time()}) + '\n')
            return
        except Exception as exc:
            output.write_text(json.dumps({'status': 'retry', 'error_type': type(exc).__name__,
                                         'attempt': attempt + 1}) + '\n')
            time.sleep(5)
    raise RuntimeError('Independent stop failed; external control-plane stop required')


if __name__ == '__main__':
    main()
