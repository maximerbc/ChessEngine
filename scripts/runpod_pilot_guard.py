"""Pod-local independent deadline guard; never print injected credentials.

Run with the system Python before installing the training environment. The guard
can stop only RUNPOD_POD_ID using the Pod's injected API credential. The external
operator must also verify the final state through Runpod's control plane.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib
import urllib.request


def pod_credential():
    key = os.environ.get('RUNPOD_POD_API_KEY') or os.environ.get('RUNPOD_API_KEY')
    if not key:
        path = Path.home() / '.runpod/config.toml'
        if path.is_file():
            with path.open('rb') as stream:
                key = tomllib.load(stream).get('apiKey')
    if not key:
        raise RuntimeError('Pod-local stop credential is missing')
    return key


def stop_self():
    pod = os.environ['RUNPOD_POD_ID']
    key = pod_credential()
    request = urllib.request.Request(f'https://api.runpod.io/v2/pods/{pod}/action',
        data=json.dumps({'action':'stop'}).encode(),
        headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status not in [200, 202, 204]:
            raise RuntimeError('Pod-local stop was rejected')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--done-file')
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    credential = bool(pod_credential())
    if not os.environ.get('RUNPOD_POD_ID') or not credential:
        raise RuntimeError('Independent shutdown cannot be armed')
    output.write_text(json.dumps(dict(status='armed', deadline_unix=args.deadline_unix,
                                     pod_id=os.environ['RUNPOD_POD_ID']))+'\n')
    while time.time() < args.deadline_unix:
        if args.done_file and Path(args.done_file).exists():
            break
        time.sleep(min(5, max(.1, args.deadline_unix-time.time())))
    # Best-effort flush; a stuck learner cannot prevent the independent stop.
    subprocess.run(['sync'], timeout=10, check=False)
    for attempt in range(12):
        try:
            stop_self()
            output.write_text(json.dumps(dict(status='stop-requested', attempt=attempt+1,
                pod_id=os.environ['RUNPOD_POD_ID'], unix=time.time()))+'\n')
            return
        except Exception as exc:
            output.write_text(json.dumps(dict(status='stop-retry', error_type=type(exc).__name__,
                attempt=attempt+1, unix=time.time()))+'\n')
            time.sleep(5)
    raise RuntimeError('Automatic stop failed; external control-plane stop required')


if __name__ == '__main__':
    main()
