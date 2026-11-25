"""The public lineup must keep model identities and inference methods explicit."""
import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chessengine.api import create_app


def test_real_release_lineup_and_direct_inference(monkeypatch):
    monkeypatch.setenv('REQUIRE_MODELS', '1')
    monkeypatch.setattr('chessengine.api.Search.run', lambda *_: pytest.fail('CNN invoked search'))
    monkeypatch.setattr('chessengine.api.stockfish_assessment', lambda *_: {'status': 'unavailable'})
    with TestClient(create_app()) as client:
        catalog = client.get('/models').json()
        assert catalog['default'] == 'supervised-1m'
        assert [m['id'] for m in catalog['models']] == ['handcrafted', 'supervised-1m', 'elite-policy']
        for retired in ['supervised', 'rl', 'evolutionary', 'move-guided', 'evolutionary-gpu']:
            assert client.post('/analyze', json={'model_id': retired}).status_code == 410
        result = client.post('/analyze', json={'model_id': 'elite-policy', 'moves': ['e2e4']})
        assert result.status_code == 200
        body = result.json()
        assert body['best_move'] and body['root_rankings']
        assert body['value'] is None and body['nodes'] is None and body['depth'] is None
        assert body['pv'] == []
        comparison = client.post('/evaluate-comparison', json={'model_id': 'elite-policy'}).json()
        assert comparison['search'] is None and comparison['position_value'] is None
        assert comparison['policy']['root_rankings']


@pytest.mark.parametrize('change', ['default', 'missing', 'schema', 'checksum', 'extra'])
def test_release_rejects_changed_declarations(tmp_path, monkeypatch, change):
    import shutil
    monkeypatch.setenv('REQUIRE_MODELS', '1')
    manifest = json.loads(Path('artifacts/models/manifest.json').read_text())
    for path in Path('artifacts/models').iterdir():
        if path.is_file():
            shutil.copy2(path, tmp_path / path.name)
    if change == 'default':
        manifest['default'] = 'handcrafted'
    elif change == 'missing':
        (tmp_path / 'elite-policy.npz').unlink()
    elif change == 'schema':
        manifest['models'][0]['feature_version'] = 1
    elif change == 'checksum':
        manifest['models'][1]['sha256'] = '0' * 64
    else:
        manifest['models'].append(copy.deepcopy(manifest['models'][0]))
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        create_app(tmp_path / 'manifest.json')

