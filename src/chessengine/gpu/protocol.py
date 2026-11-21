"""Frozen study definitions and immutable, atomic continuation state."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import time
import uuid
import atexit
from importlib.metadata import distributions

ARCHITECTURE = [853, 128, 64, 1]
SEEDS = [42, 43, 44]
MODEL_IDS = {'supervised-1m': 'supervised', 'evolutionary-gpu': 'evolutionary'}
LEGACY_SHA = 'ff3436d07d18eb4953e24e6c83b55d25bbb46b44094ed347f99320148ae04c4b'


def nvidia_available():
    import jax
    return any(device.platform == 'gpu' and
               'cuda' in str(getattr(device.client,'platform_version','')).lower()
               for device in jax.devices())


def sha(path):
    h = hashlib.sha256()
    size = 0
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
            size += len(block)
    if size != Path(path).stat().st_size:
        raise OSError(f'Incomplete file read (possibly an iCloud placeholder): {path}')
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def freeze(output, config, inputs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Source hashes reject code changes even when the Git checkout is dirty.
    source_root = Path(__file__).parent.parent
    sources = {str(p.relative_to(source_root)): sha(p) for p in source_root.rglob('*.py')}
    sources['puct_action_tables.json'] = sha(source_root / 'puct_action_tables.json')
    packages = {d.metadata['Name'].lower():d.version for d in distributions() if d.metadata['Name']}
    lock_root = Path(__file__).resolve().parents[3] / 'training'
    locks = {p.name:sha(p) for p in lock_root.glob('*.lock')}
    research_sources = lock_root.parent / 'docs/gpu-sources.json'
    expected = {'config': config, 'inputs': {str(p): sha(p) for p in inputs},
                'sources': sources, 'python': platform.python_version(), 'packages': packages, 'locks':locks,
                'research_sources_sha256':sha(research_sources) if research_sources.exists() else None}
    path = output / 'protocol.json'
    if path.exists():
        record = json.loads(path.read_text())
        if record['identity'] != identity(expected):
            raise ValueError('Protocol, source, runtime or input changed; use a new study directory')
    else:
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
                                             timeout=5).decode().strip()
            dirty = bool(subprocess.check_output(['git','status','--porcelain'],timeout=5))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            commit = 'unavailable-source-export'
            dirty = True
        atomic_json(path, {'identity': identity(expected), 'definition': expected, 'commit': commit, 'dirty':dirty})
    return identity(expected)


def save_state(path, state, protocol_id):
    """Trusted private continuation only; never load pickle supplied by an API user."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps({'protocol_id': protocol_id, 'state': state}, protocol=5)
    envelope = pickle.dumps((hashlib.sha256(payload).hexdigest(), payload), protocol=5)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        stream.write(envelope)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_state(path, protocol_id):
    checksum, payload = pickle.loads(Path(path).read_bytes())
    if hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError('Continuation checksum mismatch')
    record = pickle.loads(payload)
    if record['protocol_id'] != protocol_id:
        raise ValueError('Continuation protocol mismatch')
    return record['state']


def require_execution(config, approval_path, stage='learning'):
    """A local receipt records the human budget decision; no resource is provisioned here."""
    receipt = json.loads(Path(approval_path).read_text())
    if (receipt.get('protocol_sha256') != identity(config) or receipt.get('budget_cad', 0) <= 0
            or not receipt.get('provider') or not receipt.get('authorized_by_user')
            or (stage != 'benchmark' and receipt.get('forecast_sha256') is None)
            or receipt.get('stage') != stage):
        raise ValueError('A provider, positive budget and reviewed forecast must be authorized first')
    if stage == 'learning':
        path = receipt.get('forecast_path')
        if not path or sha(path) != receipt['forecast_sha256']:
            raise ValueError('The reviewed forecast file and checksum must be preserved in the execution receipt')
    return receipt


def begin_execution(config, receipt, command):
    """Local audit only. Actual invoice charges remain external evidence."""
    started = time.perf_counter()
    utc = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    path = Path('runs/gpu-value-v1/executions') / (time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8]+'.json')
    record = dict(command=command,protocol_sha256=identity(config),provider=receipt['provider'],
        approved_budget_cad=receipt['budget_cad'],started_utc=utc,status='started',elapsed_seconds=None,
        actual_cost_cad=None,estimated_cost_cad=None,resources_provisioned_by_runner=False)
    atomic_json(path,record)
    def finish(status='ended-unverified'):
        if record['status'] != 'started':
            return
        elapsed = time.perf_counter()-started
        rate = receipt.get('hourly_rate_cad')
        record.update(status=status,elapsed_seconds=elapsed,
            ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            estimated_cost_cad=elapsed/3600*rate if isinstance(rate,(float,int)) else None)
        atomic_json(path,record)
    atexit.register(finish)
    return finish
