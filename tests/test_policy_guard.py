"""Credential-safe shutdown checks; no real network or cloud resources."""
import importlib.util
import json
from pathlib import Path
import pytest


def guard_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/runpod_policy_guard.py'
    spec = importlib.util.spec_from_file_location('policy_guard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(monkeypatch, output):
    monkeypatch.setattr('sys.argv', ['guard', '--deadline-unix', '0', '--output', str(output),
                                   '--backup-verified-file', str(output.parent / 'backup.ok')])


def test_guard_stops_only_its_injected_pod(tmp_path, monkeypatch):
    guard = guard_module()
    output = tmp_path / 'guard.json'
    arguments(monkeypatch, output)
    monkeypatch.setenv('RUNPOD_POD_ID', 'test-owned-pod')
    monkeypatch.setenv('RUNPOD_API_KEY', 'dummy-test-value')
    monkeypatch.delenv('RUNPOD_POD_API_KEY', raising=False)
    monkeypatch.setattr(guard.subprocess, 'run', lambda *a, **k: None)
    requests = []

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def urlopen(request, **kwargs):
        requests.append(request)
        return Response()

    monkeypatch.setattr(guard.urllib.request, 'urlopen', urlopen)
    guard.main()
    assert len(requests) == 1
    assert requests[0].full_url == 'https://api.runpod.io/v2/pods/test-owned-pod/action'
    assert json.loads(requests[0].data) == {'action': 'stop'}
    assert json.loads(output.read_text())['status'] == 'stop-requested'
    assert 'dummy-test-value' not in output.read_text()


def test_guard_fails_closed_without_injected_credential(tmp_path, monkeypatch):
    guard = guard_module()
    arguments(monkeypatch, tmp_path / 'guard.json')
    monkeypatch.setenv('RUNPOD_POD_ID', 'test-owned-pod')
    monkeypatch.delenv('RUNPOD_API_KEY', raising=False)
    monkeypatch.delenv('RUNPOD_POD_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='credential'):
        guard.main()
    assert not (tmp_path / 'guard.json').exists()
