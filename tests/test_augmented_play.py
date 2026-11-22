import copy
import json
from pathlib import Path

import chess
import pytest
from fastapi.testclient import TestClient
from chessengine.api import create_app
from chessengine.gpu.protocol import LEGACY_SHA, sha
from chessengine.gpu.release import validate_entry, EXPERIMENTAL_SUPERVISED_SHA
from chessengine.model import NumpyEvaluator


def test_real_preview_identity_and_production_move(monkeypatch):
    monkeypatch.setenv('REQUIRE_MODELS', '1')
    manifest = json.loads(Path('artifacts/models/manifest.json').read_text())
    assert manifest['default'] == 'supervised'
    assert sha('artifacts/models/supervised.npz') == LEGACY_SHA
    assert sha('web/public/models/supervised.npz') == LEGACY_SHA
    assert sha('artifacts/models/supervised-1m.npz') == EXPERIMENTAL_SUPERVISED_SHA
    assert sha('web/public/models/supervised-1m.npz') == EXPERIMENTAL_SUPERVISED_SHA
    with TestClient(create_app()) as client:
        data = client.get('/models').json()
        assert [r['id'] for r in data['models']] == ['handcrafted', 'supervised', 'supervised-1m']
        preview = data['models'][2]
        assert preview['experimental'] and preview['qualified'] is False
        assert preview['strength_status'] == 'not-established'
        response = client.post('/analyze', json={'model_id':'supervised-1m', 'simulations':4})
        assert response.status_code == 200
        result = response.json()
        assert result['best_move'] in {m.uci() for m in chess.Board().legal_moves}
        assert result['checkpoint_id'] == EXPERIMENTAL_SUPERVISED_SHA
        assert result['simulations'] == 4 and result['nodes'] == 0
        assert result['strategy'] == 'puct-uniform-v1'


def test_access_exception_does_not_fabricate_or_disable_qualification():
    manifest = json.loads(Path('artifacts/models/manifest.json').read_text())
    row = manifest['models'][1]
    evaluator = NumpyEvaluator('artifacts/models/supervised-1m.npz')
    validate_entry(row, evaluator)
    for change in [dict(authorization_id='unapproved'), dict(qualified=True),
                   dict(experimental=False), dict(strength_status='stronger'),
                   dict(id='evolutionary-gpu'), dict(release_status='qualified')]:
        changed = dict(copy.deepcopy(row), **change)
        with pytest.raises(ValueError):
            validate_entry(changed, evaluator)
    # The authorization is checkpoint-specific, never a blanket environment bypass.
    evaluator.checkpoint_id = '0' * 64
    with pytest.raises(ValueError):
        validate_entry(row, evaluator)
