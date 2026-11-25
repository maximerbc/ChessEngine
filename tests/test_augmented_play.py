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
    assert manifest['default'] == 'supervised-1m'
    assert sha('artifacts/models/supervised.npz') == LEGACY_SHA
    assert not Path('web/public/models/supervised.npz').exists()
    assert sha('artifacts/models/supervised-1m.npz') == EXPERIMENTAL_SUPERVISED_SHA
    assert sha('web/public/models/supervised-1m.npz') == EXPERIMENTAL_SUPERVISED_SHA
    public_manifest = json.loads(Path('web/public/models/manifest.json').read_text())
    assert public_manifest == manifest
    with TestClient(create_app()) as client:
        data = client.get('/models').json()
        assert [r['id'] for r in data['models']] == ['handcrafted', *manifest['required_model_ids']]
        preview = next(row for row in data['models'] if row['id'] == 'supervised-1m')
        assert preview['experimental'] and preview['qualified'] is False
        assert preview['strength_status'] == 'not-established'
        assert preview['strategy'] == 'alpha-beta'
        def forbid_puct(*args, **kwargs):
            pytest.fail('Augmented serving must not dispatch to PUCT')
        monkeypatch.setattr('chessengine.api.ValueSearch', forbid_puct)
        response = client.post('/analyze', json={'model_id':'supervised-1m', 'nodes':128, 'time_ms':100, 'simulations':4})
        assert response.status_code == 200
        result = response.json()
        assert result['best_move'] in {m.uci() for m in chess.Board().legal_moves}
        assert result['checkpoint_id'] == EXPERIMENTAL_SUPERVISED_SHA
        assert result.get('simulations') is None and 0 < result['nodes'] <= 128
        assert result['strategy'] == 'alpha-beta'
        monkeypatch.setattr('chessengine.api.stockfish_assessment', lambda *_: {'status':'unavailable'})
        compared = client.post('/evaluate-comparison', json={'model_id':'supervised-1m'})
        assert compared.status_code == 200
        search = compared.json()['search']
        assert search['strategy'] == 'alpha-beta' and search['simulations'] is None
        assert search['limits'] == {'nodes':30000, 'time_ms':3000}
        assert 0 < search['nodes'] <= 30000


def test_access_exception_does_not_fabricate_or_disable_qualification():
    manifest = json.loads(Path('artifacts/models/manifest.json').read_text())
    row = next(row for row in manifest['models'] if row['id'] == 'supervised-1m')
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


def test_alpha_beta_exception_cannot_qualify_or_change_other_models():
    row = next(row for row in json.loads(Path('artifacts/models/manifest.json').read_text())['models'] if row['id'] == 'supervised-1m')
    evaluator = NumpyEvaluator('artifacts/models/supervised-1m.npz')
    for change in [dict(search_authorization_id='unapproved'),
                   dict(search={'strategy':'alpha-beta', 'simulations':128}),
                   dict(qualified=True, qualification_sha256='a'*64, final_sha256='b'*64)]:
        with pytest.raises(ValueError):
            validate_entry(dict(copy.deepcopy(row), **change), evaluator)
    # Historical explicit PUCT previews remain readable; no study is relabeled.
    historical = copy.deepcopy(row)
    historical.pop('search_authorization_id')
    historical['search'] = dict(strategy='puct-uniform-v1', c_puct=1.5,
                                priors='uniform-legal', simulations=128)
    validate_entry(historical, evaluator)


def test_augmented_alpha_beta_recognizes_terminal_mate():
    fen = '7k/8/5KQ1/8/8/8/8/8 w - - 0 1'
    with TestClient(create_app()) as client:
        response = client.post('/analyze', json={'model_id':'supervised-1m',
            'initial_fen':fen, 'nodes':1000, 'time_ms':500})
        assert response.status_code == 200
        result = response.json()
        board = chess.Board(fen)
        board.push_uci(result['best_move'])
        assert board.is_checkmate() and result['terminal'] == '1-0'
        assert result['mate_score'] and result['strategy'] == 'alpha-beta'
