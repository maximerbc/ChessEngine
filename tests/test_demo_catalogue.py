"""Exercise the actual trained catalogue, serving strategies and release boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import chess
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from chessengine.api import create_app
from chessengine.common import digest
from chessengine.demo_models import MODEL_IDS, POLICY_KINDS, load_catalogue, weight_digest
from chessengine.features import encode
from chessengine.model import load_export
from chessengine.move_policy_train import load_export_to_torch
from chessengine.elite_train import torch_from_export
from chessengine.elite_policy import encode_board, MOVE_IDS
from chessengine.ranking import move_features


DIRECTORY = Path('artifacts/models')
MANIFEST = json.loads((DIRECTORY / 'manifest.json').read_text())


@pytest.fixture(scope='module')
def catalogue():
    return load_catalogue(MANIFEST, DIRECTORY)


@pytest.fixture(scope='module')
def client():
    with TestClient(create_app()) as session:
        yield session


def boards():
    return [chess.Board(),
            chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'),
            chess.Board('4k3/P7/8/8/8/8/7p/4K3 w - - 0 1'),
            chess.Board('4k3/P7/8/8/8/8/7p/4K3 b - - 0 1'),
            chess.Board('rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3')]


def test_complete_lineup_and_private_artifacts(client):
    response = client.get('/models').json()
    assert response['default'] == 'supervised'
    assert {r['id'] for r in response['models']} == {'handcrafted', *MODEL_IDS}
    assert len(response['models']) == 20
    for row in MANIFEST['models']:
        assert digest(DIRECTORY / row['file']) == row['sha256']
        assert weight_digest(DIRECTORY / row['file']) == row['weights_sha256']
        if row['id'] not in {'supervised', 'supervised-1m'}:
            assert not (Path('web/public/models') / row['file']).exists()
            assert row['experimental'] is True and row['qualified'] is False
    assert client.post('/analyze', json={'model_id': 'rl'}).status_code == 410
    assert client.post('/analyze', json={'model_id': 'evolutionary'}).status_code == 404
    assert client.post('/analyze', json={'model_id': 'not-installed'}).status_code == 404


@pytest.mark.parametrize('model_id', ['handcrafted', *sorted(MODEL_IDS)])
def test_real_move_and_history_sensitive_draw_for_every_opponent(client, model_id):
    result = client.post('/analyze', json=dict(model_id=model_id, moves=['e2e4'], nodes=32, time_ms=100))
    assert result.status_code == 200, result.text
    result = result.json()
    board = chess.Board()
    board.push_uci('e2e4')
    assert chess.Move.from_uci(result['best_move']) in board.legal_moves
    board.push_uci(result['best_move'])
    assert result['resulting_fen'] == board.fen()
    if model_id in POLICY_KINDS:
        assert result['value'] is result['depth'] is result['nodes'] is None
        assert result['search'] is None and result['pv'] == []
        assert result['strategy'] == 'direct-policy-v1'
        assert sum(r['probability'] for r in result['root_rankings']) == pytest.approx(1)
    else:
        assert result['strategy'] == 'alpha-beta'
        assert result['nodes'] <= 32
    repeated = ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1']
    draw = client.post('/analyze', json=dict(model_id=model_id, moves=repeated, nodes=32)).json()
    assert draw['terminal'] == '1/2-1/2' and draw['best_move'] is None


@pytest.mark.parametrize('model_id', sorted(MODEL_IDS))
def test_actual_export_matches_training_math(catalogue, model_id):
    """Synthetic legal boards only: no training, labels or new final-test selection."""
    torch.set_num_threads(1)
    row = next(r for r in MANIFEST['models'] if r['id'] == model_id)
    runtime = catalogue[model_id]['evaluator']
    if model_id not in POLICY_KINDS:
        model, _ = load_export(DIRECTORY / row['file'])
        model.eval()
        for board in boards():
            with torch.inference_mode():
                expected = model(torch.from_numpy(encode(board, runtime.feature_version))).item()
            assert runtime(board) == pytest.approx(expected, abs=1e-6)
        return
    policy = runtime.policy
    human = model_id == 'elite-policy'
    model, _ = (torch_from_export if human else load_export_to_torch)(DIRECTORY / row['file'], torch.device('cpu'))
    model.eval()
    for board in boards():
        ranked = policy.rank(board)
        assert {r['move'] for r in ranked} == {m.uci() for m in board.legal_moves}
        moves = sorted(board.legal_moves, key=lambda m: m.uci()) if human else list(board.legal_moves)
        with torch.inference_mode():
            if human:
                expected = model(torch.from_numpy(encode_board(board)[None]))[0].numpy()
                expected = expected[[MOVE_IDS[m.uci()] for m in moves]]
                actual = policy.logits(encode_board(board))[[MOVE_IDS[m.uci()] for m in moves]]
            else:
                x, features = encode(board, 3), move_features(board, moves)
                expected = model(torch.from_numpy(x[None]), torch.from_numpy(features[None]),
                                 torch.ones((1, len(moves)), dtype=torch.bool))[0].numpy()
                actual = policy.logits(x, features)
        def probabilities(logits):
            values = np.exp(logits - logits.max())
            return values / values.sum()
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=1e-4)
        np.testing.assert_allclose(probabilities(actual), probabilities(expected), atol=3e-5, rtol=1e-4)
        assert int(np.argmax(actual)) == int(np.argmax(expected))


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'duplicate-weights', 'weights', 'schema', 'kind', 'default', 'gate', 'path', 'symlink'])
def test_catalogue_fails_closed(tmp_path, change):
    directory = tmp_path / 'models'
    shutil.copytree(DIRECTORY, directory)
    manifest = deepcopy(MANIFEST)
    row = manifest['models'][1]
    if change == 'missing':
        (directory / row['file']).unlink()
    elif change == 'duplicate':
        manifest['models'].append(dict(row))
    elif change == 'duplicate-weights':
        row.update({key: manifest['models'][0][key] for key in ['file', 'sha256', 'weights_sha256']})
    elif change == 'weights':
        row['sha256'] = '0' * 64
    elif change == 'schema':
        row['feature_version'] = 99
    elif change == 'kind':
        row['kind'] = 'human-move-policy'
    elif change == 'default':
        manifest['default'] = row['id']
    elif change == 'gate':
        (directory / 'stockfish-policy-gate.json').write_text('{}')
    elif change == 'path':
        row['file'] = '../outside.npz'
    elif change == 'symlink':
        outside = tmp_path / 'outside.npz'
        shutil.copyfile(directory / row['file'], outside)
        (directory / row['file']).unlink()
        (directory / row['file']).symlink_to(outside)
    with pytest.raises(ValueError):
        load_catalogue(manifest, directory)


def test_policy_comparison_returns_only_real_capabilities(client, monkeypatch):
    monkeypatch.setattr('chessengine.api.stockfish_assessment', lambda *args: {'status': 'unavailable'})
    for model_id in POLICY_KINDS:
        result = client.post('/evaluate-comparison', json={'model_id': model_id}).json()
        assert result['position_value'] is result['search'] is None
        assert result['policy']['root_rankings']
        assert result['stockfish']['status'] == 'unavailable'
