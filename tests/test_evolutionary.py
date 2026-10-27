import copy
import json
from pathlib import Path
import chess
import numpy as np
import pytest
from chessengine.common import read_json, write_json
from chessengine.evolution_model import features, prediction, export, EvolutionEvaluator, load_evaluator
from chessengine.evolution_games import audit_ledger, tasks_for, run_tasks
from chessengine.evolution_study import generation_starts, fitness, evolve, release_gate
from chessengine.features import record_board


def test_feature_definition_symmetry_promotions_and_original_history():
    board = chess.Board()
    x = features(board)
    assert x.shape == (63,) and x.dtype == np.float32
    assert np.array_equal(x[:10], np.ones(10))
    assert np.array_equal(x[47:51], np.ones(4))
    assert x[59] == 1 and x[60] == 0
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
        board.push_uci(move)
    assert features(board)[61] == 1 and features(board)[62] == 0
    bare = chess.Board(board.fen())
    assert features(bare)[61] == 0
    board.push_uci('e2e4')
    assert features(board)[55] == 1  # en-passant e-file in its own group
    assert np.array_equal(features(board), features(board.mirror()))
    promoted = chess.Board('4k3/8/8/8/8/8/QQ6/4K3 w - - 149 1')
    assert features(promoted)[4] == 2
    assert features(promoted)[60] == pytest.approx(149/150)


def test_export_parity_rejects_shapes_and_schemas(tmp_path):
    weights = np.random.default_rng(42).normal(0, .05, 324)
    path = tmp_path/'evo.npz'
    export(weights, path, dict(seed=42))
    model = load_evaluator(path)
    for board in [chess.Board(), chess.Board('4k3/8/8/8/8/8/QQ6/4K3 b - - 0 1')]:
        assert model(board) == float(prediction(weights, features(board)))
        assert -1 <= model(board) <= 1
    assert prediction(np.zeros(324), features(chess.Board())) == 0
    with pytest.raises(ValueError, match='parameters'):
        export(np.zeros(323), path)
    with np.load(path) as z:
        data = {key: z[key].copy() for key in z.files}
    meta = json.loads(str(data['metadata']))
    meta['feature_schema'] = 'guess'
    np.savez(path, parameters=data['parameters'], metadata=json.dumps(meta))
    with pytest.raises(ValueError, match='schema'):
        load_evaluator(path)


def opening(board, identity, kind='endgame'):
    return dict(record_board(board), id=identity, source_game_id=identity, stratum=kind)


def test_match_resume_truthful_caps_and_checksum_guard(tmp_path):
    candidate = tmp_path/'evo.npz'
    export(np.zeros(324), candidate)
    row = opening(chess.Board(), 'start', 'opening')
    tasks = tasks_for(candidate, 'handcrafted', [row], dict(nodes=8, qdepth=1, max_plies=1))
    ledger = tmp_path/'matches.json'
    result = run_tasks(tasks, ledger, workers=1)
    assert len(result['games']) == 2
    assert {g['candidate_color'] for g in result['games']} == {'white', 'black'}
    assert all(g['capped'] and g['score'] == .5 and g['result'] == '*' for g in result['games'])
    assert audit_ledger(result)['matches'] == 2
    tampered = copy.deepcopy(result)
    tampered['games'][0]['score'] = 1.
    with pytest.raises(ValueError, match='outcome'):
        audit_ledger(tampered)
    assert run_tasks(tasks, ledger, workers=1) == result
    export(np.ones(324)*.01, candidate)
    with pytest.raises(ValueError, match='changed'):
        run_tasks(tasks, ledger, workers=1)


def test_curriculum_is_deterministic_and_generation_matched():
    pool = [dict(id=f'{kind}{i}', stratum=kind) for kind in ('endgame', 'middlegame', 'opening') for i in range(8)]
    for generation, expected in [(1, {'endgame': 6}), (11, {'endgame': 3, 'middlegame': 3}),
                                  (21, {'middlegame': 3, 'opening': 3})]:
        rows = generation_starts(pool, generation, 42)
        assert rows == generation_starts(pool, generation, 42)
        assert {k: sum(r['stratum'] == k for r in rows) for k in expected} == expected


def test_fitness_uses_predeclared_opponent_weights():
    games = [dict(candidate_index=i, opponent_group=group, opening_id=f'{group}{p}', candidate_color=color,
                  score=score, capped=False) for i in range(2)
             for group, score in [('classical', 1.), ('supervised', 0.), ('archive', .5)]
             for p in range(2) for color in ('white', 'black')]
    assert np.array_equal(fitness(games, 2), [.625, .625])
    with pytest.raises(ValueError, match='Incomplete'):
        fitness(games[:-1], 2)


def test_exact_optimizer_resume_without_repeated_updates(tmp_path, monkeypatch):
    config = read_json('configs/evolutionary.json')
    config.update(output=str(tmp_path), population=4, workers=1)
    write_json(tmp_path/'protocol.json', {'test': 'frozen'})
    pool = [dict(id=f'e{i}', stratum='endgame') for i in range(8)]
    calls = []
    interrupt = [True]
    def fake(tasks, path, workers):
        calls.append(str(path))
        if 'generation-002' in str(path) and interrupt[0]:
            raise KeyboardInterrupt()
        games = []
        for task in tasks:
            p = EvolutionEvaluator(task['candidate']).parameters
            score = float(np.tanh(p.sum())*.2+.5)
            games.append(dict(task_id=task['task_id'], candidate_index=task['candidate_index'],
                              opponent_group=task['opponent_group'], opening_id=task['opening']['id'],
                              candidate_color=task['color'], score=score, capped=False, result='1/2-1/2'))
        return dict(games=games, active_seconds=0., peak_combined_bytes=1)
    monkeypatch.setattr('chessengine.evolution_study.run_tasks', fake)
    with pytest.raises(KeyboardInterrupt):
        evolve(config, pool, 42, 2, 32, tmp_path/'resumed')
    interrupt[0] = False
    evolve(config, pool, 42, 2, 32, tmp_path/'resumed')
    evolve(config, pool, 42, 2, 32, tmp_path/'whole')
    assert calls.count(str(tmp_path/'resumed/generation-001/matches.json')) == 1
    a = EvolutionEvaluator(tmp_path/'resumed/mean-002.npz').parameters
    b = EvolutionEvaluator(tmp_path/'whole/mean-002.npz').parameters
    assert np.array_equal(a, b)
    assert len(read_json(tmp_path/'resumed/history.json')) == 2


def test_release_gate_separates_seed_variance_and_requires_representative():
    config = read_json('configs/evolutionary.json')
    config['bootstrap_resamples'] = 100
    def ledger(score):
        return dict(games=[dict(opening_id=str(i), candidate_color=c, score=score, capped=False)
                           for i in range(20) for c in ('white', 'black')])
    sets = {name: [ledger(.75) for _ in range(3)] for name in ('classical', 'supervised')}
    assert release_gate(sets, config)['passed']
    sets['classical'][0] = ledger(.5)
    assert not release_gate(sets, config)['passed']
    sets['classical'][0] = ledger(.75)
    sets['supervised'][0] = ledger(.4)
    assert not release_gate(sets, config)['passed']
    assert 'seed_sd' in release_gate(sets, config)['opponents']['classical']['aggregate']


def test_retirement_manifest_and_reports(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from chessengine.api import create_app
    from chessengine.evolution_report import public_protocol, release
    monkeypatch.setenv('REQUIRE_MODELS', '1')
    with TestClient(create_app()) as client:
        data = client.get('/models').json()
        assert data['default'] == 'supervised'
        manifest = read_json('artifacts/models/manifest.json')
        assert {m['id'] for m in data['models']} == {'handcrafted', *[r['id'] for r in manifest['models']]}
        assert client.post('/analyze', json={'model_id': 'rl'}).status_code == 410
        expected = 200 if 'evolutionary' in manifest['required_model_ids'] else 404
        assert client.post('/analyze', json={'model_id': 'evolutionary', 'nodes': 8}).status_code == expected
    assert not Path('web/public/models/rl.npz').exists()
    old = read_json('artifacts/research/original-release.json')
    assert any(r['id'] == 'rl' for r in old['models'])
    assert not release({'output': str(tmp_path)})['published']
    raw = dict(identity=dict(config=dict(output='/private/foo', archive='data/foo', seed=42),
                             packages={}, sources={}, inputs={'data/foo': 'abc'}), provenance={})
    assert 'data/foo' not in json.dumps(public_protocol(raw))


def test_split_validator_rejects_duplicates():
    from chessengine.evolution_data import validate
    row = opening(chess.Board(), 'same')
    with pytest.raises(ValueError, match='leakage'):
        validate(dict(train=[row], final=[copy.deepcopy(row)]))
