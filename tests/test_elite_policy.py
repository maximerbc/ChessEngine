import io
import json
from pathlib import Path
import shutil
import zipfile
import chess
import chess.pgn
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from chessengine.common import digest, read_json
from chessengine.features import record_board
from chessengine.elite_policy import (MOVES, MOVE_IDS, SCHEMA, encode_board, build_policy,
                                     export_policy, NumpyElitePolicy)
from chessengine.elite_data import (game_pairs, prepare, atomic_json, write_shard, ShardDataset)
from chessengine.elite_train import train, parity, MemoryBatches, evaluate_model, bootstrap


@pytest.fixture
def tiny_data(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    config = dict(study='tiny-test', seeds=[42, 43, 44], representative_seed=42,
        positions=dict(train=2, validation=2, test=2),
        training=dict(learning_rate=.0001, batch_size=1, epochs=2, gradient_clip=1., checkpoint_steps=1))
    board = chess.Board()
    rows = []
    for i, move in enumerate(['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6']):
        legal = list(board.legal_moves)
        rows.append(('', dict(record_board(board), game_id=f'{i:064x}', target_move=move),
                     encode_board(board), [MOVE_IDS[m.uci()] for m in legal]))
        board.push_uci(move)
    shards = [write_shard(root, split, 0, rows[i * 2:(i + 1) * 2])
              for i, split in enumerate(config['positions'])]
    atomic_json(root / 'dataset.json', dict(schema=SCHEMA, counts=config['positions'], config=config, shards=shards))
    return config, root


def test_complete_vocabulary_and_reference_plane_orientation():
    assert len(MOVES) == 4208 and tuple(sorted(set(MOVES))) == MOVES
    board = chess.Board()
    x = encode_board(board)
    assert x.shape == (13, 8, 8) and x[0, 1, 4] == 1 and x[6, 6, 4] == 1
    assert x[5, 0, 4] == 1 and x[11, 7, 4] == 1
    expected = np.zeros((8, 8))
    for m in board.legal_moves:
        expected[m.to_square // 8, m.to_square % 8] = 1
    np.testing.assert_array_equal(x[12], expected)
    for fen in ['4k3/P7/8/8/8/8/7p/4K3 w - - 0 1',
                '4k3/P7/8/8/8/8/7p/4K3 b - - 0 1',
                'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1']:
        assert all(m.uci() in MOVE_IDS for m in chess.Board(fen).legal_moves)
    assert all('a7a8' + p in MOVE_IDS and 'h2h1' + p in MOVE_IDS for p in 'qrbn')


def test_board_is_before_played_move_and_draw_convention():
    game = chess.pgn.read_game(io.StringIO('1. e4 e5 2. Nf3 Nc6 *'))
    gid, pairs = game_pairs(game)
    assert pairs[0][1]['fen'] == chess.STARTING_FEN
    assert pairs[0][1]['target_move'] == 'e2e4'
    assert pairs[1][1]['moves'] == ['e2e4'] and pairs[1][1]['target_move'] == 'e7e5'
    assert len(gid) == 64
    game = chess.pgn.read_game(io.StringIO('1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 *'))
    _, pairs = game_pairs(game)
    assert all(not chess.Board(r[1]['fen']).is_checkmate() for r in pairs)
    assert len(pairs) == 7  # The eighth move permits an automatic intended-move repetition claim.
    invalid = chess.pgn.Game()
    invalid.add_main_variation(chess.Move.from_uci('e2e5'))
    assert game_pairs(invalid) is None


def test_export_probability_and_choice_parity_and_invalid_schema(tiny_data, tmp_path):
    _, root = tiny_data
    torch.manual_seed(123)
    path = tmp_path / 'policy.npz'
    model = build_policy()
    export_policy(model, path, dict(seed=42))
    assert parity(path, root)['passed']
    policy = NumpyElitePolicy(path)
    board = chess.Board()
    ranked = policy.rank(board)
    assert all(chess.Move.from_uci(r['move']) in board.legal_moves for r in ranked)
    assert sum(r['probability'] for r in ranked) == pytest.approx(1)
    assert policy.choose(board) == ranked[0]['move']
    for m in ['f2f3', 'e7e5', 'g2g4', 'd8h4']:
        board.push_uci(m)
    assert policy.rank(board) == [] and policy.analyze(board)['terminal'] == '0-1'
    with np.load(path, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    meta = json.loads(str(arrays['metadata']))
    meta['vocabulary_sha256'] = 'invalid'
    arrays['metadata'] = np.array(json.dumps(meta))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='schema/vocabulary'):
        NumpyElitePolicy(path)


def test_tied_logits_have_stable_uci_choice(tmp_path):
    model = build_policy()
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.zero_()
    path = tmp_path / 'tie.npz'
    export_policy(model, path, dict(seed=42))
    ranked = NumpyElitePolicy(path).rank(chess.Board())
    assert ranked[0]['move'] == min(m.uci() for m in chess.Board().legal_moves)
    assert all(r['probability'] == pytest.approx(.05) for r in ranked)


def test_mid_epoch_cpu_resume_is_exact(tiny_data, tmp_path):
    config, root = tiny_data
    other = tmp_path / 'other'
    shutil.copytree(root, other)
    train(config, root, 42, stop_after_steps=1)
    train(config, root, 42, resume=True)
    train(config, other, 42)
    a, b = NumpyElitePolicy(root / 'seed-42/last.npz'), NumpyElitePolicy(other / 'seed-42/last.npz')
    for name in a.arrays:
        np.testing.assert_array_equal(a.arrays[name], b.arrays[name])
    assert read_json(root / 'seed-42/metrics.json')['complete']
    changed = {**config, 'study': 'changed'}
    with pytest.raises(ValueError, match='mismatch'):
        train(changed, root, 42, resume=True)


def test_tiny_overfit_and_bootstrap_pairing(tiny_data):
    _, root = tiny_data
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = build_policy()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    data = MemoryBatches(root, 'train', 2)
    before, _ = evaluate_model(model, data, torch.device('cpu'))
    for _ in range(25):
        for x, y, _, _, _ in data.batches(torch.device('cpu')):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
    after, games = evaluate_model(model, data, torch.device('cpu'))
    assert after['cross_entropy'] < before['cross_entropy'] * .2 and after['top1'] == 1
    assert bootstrap(games, samples=30)['top1_difference'][0] > 0


def test_prepare_fresh_splits_dedup_and_resume(tmp_path):
    archives = tmp_path / 'archives'
    archives.mkdir()
    # Generate enough distinct valid source games to populate each hash-owned split.
    rng = np.random.default_rng(3)
    text = []
    for _ in range(70):
        board = chess.Board()
        for _ in range(8):
            board.push(list(board.legal_moves)[int(rng.integers(board.legal_moves.count()))])
        text.append(str(chess.pgn.Game.from_board(board)))
    with zipfile.ZipFile(archives / 'lichess_elite_2025-11.zip', 'w') as z:
        z.writestr('games.pgn', '\n\n'.join(text))
    old = tmp_path / 'old.jsonl'
    old.write_text(json.dumps(dict(fen=chess.STARTING_FEN, game_id='reserved')) + '\n')
    config = dict(first_month='2025-11', exclusions=['old.jsonl'], shard_size=5,
                  positions=dict(train=20, validation=5, test=5))
    directory = tmp_path / 'out'
    prepare(config, directory, tmp_path, archives, max_games=3)
    result = prepare(config, directory, tmp_path, archives)
    assert result['counts'] == config['positions']
    positions, games = {}, {}
    for split in config['positions']:
        data = ShardDataset(directory, split)
        assert len(data) == config['positions'][split]
        for shard in result['shards']:
            if shard['split'] != split:
                continue
            for line in (directory / shard['records']).read_text().splitlines():
                row = json.loads(line)
                key = ' '.join(row['fen'].split()[:4])
                assert key != ' '.join(chess.STARTING_FEN.split()[:4]) and key not in positions
                positions[key] = split
                assert row['game_id'] not in games or games[row['game_id']] == split
                games[row['game_id']] = split
    assert prepare(config, directory, tmp_path, archives) == result


def install_test_policy(tmp_path):
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / 'models'
    shutil.copytree(root / 'artifacts/models', directory)
    export_policy(build_policy(), directory / 'elite-policy.npz', dict(seed=42, dataset_sha256='d' * 64,
        study='elite-pytorch-baseline-20261124', training_positions=800000))
    checkpoint = digest(directory / 'elite-policy.npz')
    atomic_json(directory / 'gate.json', dict(seed=42, checkpoint_sha256=checkpoint,
        dataset_sha256='d' * 64, passed=True, export_passed=True, validation_top1_difference_ci=[.01, .1]))
    manifest = read_json(directory / 'manifest.json')
    manifest['release_version'] = 5
    manifest['models'].append(dict(id='elite-policy', name='Move Prediction', kind='human-move-policy',
        file='elite-policy.npz', sha256=checkpoint, architecture=SCHEMA,
        dataset_sha256='d' * 64,
        search=dict(strategy='direct-policy-v1'), experimental=True, qualified=False,
        strength_status='not-established', authorization_id='user-20261124-elite-play',
        evidence_id='elite-pytorch-baseline-20261124', gate_file='gate.json', gate_sha256=digest(directory / 'gate.json')))
    manifest['required_model_ids'].append('elite-policy')
    atomic_json(directory / 'manifest.json', manifest)
    return directory


def test_four_model_api_capabilities_and_no_search(tmp_path, monkeypatch):
    from chessengine.api import create_app
    directory = install_test_policy(tmp_path)
    monkeypatch.setenv('REQUIRE_MODELS', '1')
    def forbid_search(*args, **kwargs):
        pytest.fail('The human policy must not dispatch to alpha-beta')
    monkeypatch.setattr('chessengine.api.Search.run', forbid_search)
    monkeypatch.setattr('chessengine.api.stockfish_assessment', lambda *_: {'status': 'unavailable'})
    with TestClient(create_app(directory / 'manifest.json')) as client:
        models = client.get('/models').json()
        assert [m['id'] for m in models['models']] == ['handcrafted', 'supervised', 'supervised-1m', 'elite-policy']
        assert models['default'] == 'supervised'
        assert models['models'][-1]['capabilities'] == dict(position_value=False, search=False, move_probabilities=True)
        result = client.post('/analyze', json=dict(model_id='elite-policy', moves=['e2e4'])).json()
        assert result['value'] is None and result['depth'] is None and result['nodes'] is None
        assert result['strategy'] == 'direct-policy-v1' and result['root_rankings']
        board = chess.Board()
        board.push_uci('e2e4')
        assert chess.Move.from_uci(result['best_move']) in board.legal_moves
        compared = client.post('/evaluate-comparison', json=dict(model_id='elite-policy', moves=['e2e4']))
        assert compared.status_code == 200
        assert compared.json()['position_value'] is None and compared.json()['search'] is None
        assert compared.json()['policy']['root_rankings']
        assert client.post('/analyze', json=dict(model_id='elite-policy', moves=['e2e5'])).status_code == 422
        assert client.post('/analyze', json=dict(model_id='rl')).status_code == 410
    (directory / 'elite-policy.npz').unlink()
    with pytest.raises(FileNotFoundError):
        create_app(directory / 'manifest.json')


def test_no_test_opening_before_all_attempts_and_failed_seeds_are_retained(tiny_data, monkeypatch):
    from chessengine.elite_train import evaluate_all
    config, root = tiny_data

    def forbidden(*args, **kwargs):
        pytest.fail('A test or training loader was opened before sealing attempts')

    monkeypatch.setattr('chessengine.elite_train.MemoryBatches', forbidden)
    with pytest.raises(ValueError, match='All elite seed attempts'):
        evaluate_all(config, root)
    assert not (root / 'selections.json').exists()
    for seed in config['seeds']:
        directory = root / f'seed-{seed}'
        directory.mkdir()
        atomic_json(directory / 'failure.json', dict(error_type='SyntheticFailure', message='retained'))
    result = evaluate_all(config, root)
    assert len(result['seeds']) == 3 and all(r['status'] == 'failed' for r in result['seeds'])
    assert result['seed_variability']['failed_seeds'] == 3
    assert read_json(root / 'gate.json')['passed'] is False


def test_nonpositive_learning_gate_and_changed_dataset_fail_closed(tmp_path):
    from chessengine.elite_policy import validate_release
    directory = install_test_policy(tmp_path)
    manifest = read_json(directory / 'manifest.json')
    row = manifest['models'][-1]
    gate = read_json(directory / 'gate.json')
    gate['validation_top1_difference_ci'] = [-.01, .1]
    atomic_json(directory / 'gate.json', gate)
    row['gate_sha256'] = digest(directory / 'gate.json')
    with pytest.raises(ValueError, match='validation/export gate'):
        validate_release(row, NumpyElitePolicy(directory / row['file']), directory)
    row['dataset_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_release(row, NumpyElitePolicy(directory / row['file']), directory)
