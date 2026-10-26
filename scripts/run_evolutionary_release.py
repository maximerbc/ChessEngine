"""Durable local execution of the authorized frozen study; no remote operations.

Run once (or resume with the same command). A PID lock prevents duplicate runners.
Scientific stages independently verify the frozen config/code/input signatures.
The workflow finishes publication and local verification without needing a chat
to stay open. No model is published unless the scientific release gate passes.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/evolutionary'
PYTHON = ROOT/'.venv/bin/python'
STATUS = RUN/'execution.json'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        current = read(STATUS) if STATUS.exists() else {'status': 'not started'}
        if current.get('status') == 'running':
            current['active_seconds_live'] = current.get('active_seconds', 0.) + (
                datetime.now(timezone.utc)-datetime.fromisoformat(current['started_utc'])).total_seconds()
            directory = RUN/'pilot' if current.get('stage') == 'pilot' else RUN
            current['completed_generation_matches'] = sum(len(read(p)['games']) for p in directory.glob('**/generation-*/matches.json'))
        print(json.dumps(current, indent=2))
        return
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    lock = RUN/'runner.lock'
    if lock.exists():
        pid = int(lock.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise SystemExit(f'Evolutionary runner {pid} already exists; do not launch another')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(str(os.getpid()))
    previous = read(STATUS) if STATUS.exists() else {}
    state = dict(status='running', stage='starting', pid=os.getpid(), paid_cost_cad=0,
                 active_seconds=previous.get('active_seconds', 0.),
                 started_utc=datetime.now(timezone.utc).isoformat())
    started = time.monotonic()
    child = None
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/evolution-mpl',
               PLAYWRIGHT_BROWSERS_PATH='/private/tmp/chess-evolution-browsers')
    def interrupted(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            child.wait()
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    def command(stage, argv, required=True):
        nonlocal child
        state.update(stage=stage, status='running', active_seconds=previous.get('active_seconds', 0.)+time.monotonic()-started)
        write(STATUS, state)
        print(f'[{datetime.now(timezone.utc).isoformat()}] {stage}', flush=True)
        child = subprocess.Popen([str(a) for a in argv], cwd=ROOT, env=env, start_new_session=True)
        code = child.wait()
        child = None
        if required and code:
            raise RuntimeError(f'{stage} exited with status {code}; no automatic scientific retry')
        return code
    def scientific(stage):
        command(stage, [PYTHON, '-u', '-m', 'chessengine', 'evolutionary', '--stage', stage])
    try:
        # This command publishes the already measured pre-pilot forecast before learning.
        scientific('pilot')
        decision = read(RUN/'pilot-decision.json')
        if decision['passed']:
            # train emits the revised, measured main forecast before computation.
            scientific('train')
            scientific('evaluate')
            scientific('release')
        scientific('report')
        command('research publication', [PYTHON, '-m', 'chessengine.research_catalog'])
        validations = {}
        for name, argv in [
            ('Python tests', [PYTHON, '-m', 'pytest', '-q']),
            ('lint', [PYTHON, '-m', 'ruff', 'check', 'src', 'tests', 'scripts']),
            ('CLI', [PYTHON, '-m', 'chessengine', '--help']),
            ('frontend build', ['npm', '--prefix', 'web', 'run', 'build']),
            ('browser tests', ['npm', '--prefix', 'web', 'run', 'test:e2e']),
            ('diff check', ['git', 'diff', '--check']),
        ]:
            validations[name] = command(name, argv, required=False)
        results = read(RUN/'results.json') if (RUN/'results.json').exists() else None
        state.update(status='complete' if all(code == 0 for code in validations.values()) else 'verification_failed',
                     stage='finished', pilot_passed=decision['passed'], model_promoted=bool(results and results['release_eligible']),
                     validation_exit_codes=validations)
        text = ('# Evolutionary execution\n\n' +
                f'Pilot passed: {decision["passed"]}. Model promoted: {state["model_promoted"]}.\n\n' +
                'A failed pilot closes this study without changing thresholds or launching full training.\n\n' +
                '## Verification\n\n' + '\n'.join(f'- {name}: exit {code}' for name, code in validations.items()) +
                '\n\nNo paid compute, remote push or deployment. Container checks require Docker and are not established by these local checks.\n')
        (ROOT/'docs/evolutionary-execution.md').write_text(text)
        write(STATUS, state)
    except KeyboardInterrupt:
        state.update(status='interrupted', stage='resume from saved state')
    except Exception as error:
        state.update(status='error', error=str(error))
        print(f'Execution stopped: {error}', flush=True)
    finally:
        state['active_seconds'] = previous.get('active_seconds', 0.)+time.monotonic()-started
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write(STATUS, state)
        lock.unlink(missing_ok=True)
        # Reporting never restarts computation, even after a frozen-source mismatch.
        try:
            subprocess.run([str(PYTHON), '-m', 'chessengine.research_catalog'], cwd=ROOT, env=env, check=True)
        except Exception as error:
            print(f'Partial research publication unavailable: {error}', flush=True)
    if state['status'] == 'complete':
        branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
        if branch != 'codex/evolutionary-engine':
            raise RuntimeError('Local branch changed; evidence preserved without committing another branch')
        paths = ['docs/experiments/evolutionary', 'docs/evolutionary-execution.md',
                 'web/public/research/evolutionary', 'web/public/research/studies.json',
                 'artifacts/models/manifest.json', 'web/public/models/manifest.json']
        paths += [p for p in ['artifacts/models/evolutionary.npz', 'web/public/models/evolutionary.npz']
                  if (ROOT/p).exists()]
        subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
        staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).splitlines()
        if any(not any(p == allowed or p.startswith(allowed+'/') for allowed in paths) for p in staged):
            raise RuntimeError('Unrelated files are staged; evidence preserved without committing them')
        if staged:
            subprocess.run(['git', 'commit', '-m', 'Publish evolutionary evidence'], cwd=ROOT, check=True)
        state['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        state.update(status='complete', stage='finished')
        write(STATUS, state)
    print(json.dumps(state, indent=2), flush=True)
    if state['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
