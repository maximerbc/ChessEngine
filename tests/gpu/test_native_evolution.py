"""Correctness of the new lane, not throughput or playing-strength evidence."""
# ruff: noqa: E402
import json
import time

import chess
import pytest

jax = pytest.importorskip('jax')
pytest.importorskip('evosax')

from chessengine.features import record_board
from chessengine.gpu import native_games
from chessengine.gpu.games import tasks_for
from chessengine.gpu.network import initialize, export
from chessengine.gpu.native_study import export_candidate, forecast


def test_compact_search_matches_dense_reference():
    import numpy as np
    from chessengine.gpu import rules
    from chessengine.gpu.native_search import make_search
    from chessengine.gpu.search import make_search as dense_search
    params = jax.tree.map(lambda x: np.zeros_like(x)[None], initialize(jax.random.PRNGKey(5)))
    for board in [chess.Board(), chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')]:
        states = jax.tree.map(lambda x: x[None], rules.from_board(board))
        a, b = make_search(32, root_coverage=False)(params, states, np.array([1])), dense_search(32)(params, states, np.array([1]))
        assert int(a['actions'][0]) == int(b['actions'][0])
        assert float(a['value'][0]) == pytest.approx(float(b['value'][0]), abs=1e-6)
        labels = np.asarray(a['labels'][0])
        visits = np.asarray(a['visits'][0])
        restored = np.zeros(4672, np.int32)
        np.add.at(restored, labels, visits)
        assert np.array_equal(restored, np.asarray(b['visits'][0]))


def fixtures(tmp_path):
    target = tmp_path / 'candidate.npz'
    export(initialize(jax.random.PRNGKey(42)), target, 'evolutionary', 'test-only')
    opening = dict(record_board(chess.Board()), id='start', source_game_id='fixture')
    tasks = tasks_for(target, 'handcrafted', [opening], 'classical')
    return target, tasks


def test_native_chunk_resume_padding_and_atomic_ledger(tmp_path):
    _, tasks = fixtures(tmp_path)
    # Two slots with three tasks exercises padding and slot refilling.
    other = dict(record_board(chess.Board()), id='other', source_game_id='other')
    tasks += tasks_for(tasks[0]['candidate'], 'handcrafted', [other], 'classical')[:1]
    native_games.run(tasks, tmp_path/'resumed', 2, 4, 2, 2, stop_after_chunks=1)
    # Simulate loss of the derived ledger after a committed continuation.
    (tmp_path/'resumed/games.json').unlink()
    resumed = native_games.run(tasks, tmp_path/'resumed', 2, 4, 2, 2)
    whole = native_games.run(tasks, tmp_path/'whole', 2, 4, 2, 2)
    assert resumed['games'] == whole['games']
    assert len({g['task_id'] for g in resumed['games']}) == 3
    assert whole['batch_histogram']['1'] > 0
    assert whole['batch_histogram']['2'] > 0
    assert all(g['capped'] and g['result'] == '*' and g['score'] == .5 and g['replay_verified']
               for g in resumed['games'])
    with pytest.raises(ValueError, match='Protocol'):
        native_games.run(tasks, tmp_path/'resumed', 3, 4, 2, 2)


def test_native_deadline_and_input_rejection(tmp_path):
    target, tasks = fixtures(tmp_path)
    with pytest.raises(native_games.DeadlineReached):
        native_games.run(tasks, tmp_path/'partial', 2, 2, 2, 1, deadline=time.time()-1)
    ledger = json.loads((tmp_path/'partial/games.json').read_text())
    assert ledger['games'] == []
    with pytest.raises(ValueError, match='Duplicate'):
        native_games.run(tasks + tasks, tmp_path/'duplicate', 2)
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Candidate changed'):
        native_games.run(tasks, tmp_path/'changed', 2)


def test_native_complete_rules_result_and_inactive_slots(tmp_path):
    target, _ = fixtures(tmp_path)
    # Already terminal at the root: no dummy action is allowed into its history.
    board = chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 100 1')
    opening = dict(record_board(board), id='mate', source_game_id='fixture')
    tasks = tasks_for(target, 'handcrafted', [opening], 'classical')
    ledger = native_games.run(tasks, tmp_path/'mate', 2, 4, 2, 2)
    assert all(g['result'] == '1-0' and not g['capped'] and g['plies'] == 0 for g in ledger['games'])
    assert [g['score'] for g in ledger['games']] == [1, 0]


def test_native_identity_and_forecast_no_invented_rate(tmp_path):
    target = tmp_path/'new.npz'
    export_candidate(initialize(jax.random.PRNGKey(1)), target, 42, 0, 'fixture', 'test-only')
    from chessengine.model import NumpyEvaluator
    model = NumpyEvaluator(target)
    assert model.metadata['study'] == 'evolutionary-gpu-native-v1'
    assert model.metadata['search'] == 'puct-rootcover-v1'
    result = forecast(tmp_path, {}, .89)
    assert result['games_per_second'] is None
    assert all(r['projected_compute_usd'] is None for r in result['scenarios'])
    assert result['full_training_authorized'] is False


def test_rootcover_cpu_gpu_parity(tmp_path):
    import numpy as np
    from chessengine.gpu import rules
    from chessengine.gpu.native_search import make_search
    from chessengine.rootcover import RootCoverageValueSearch
    from chessengine.model import NumpyEvaluator
    target, _ = fixtures(tmp_path)
    evaluator = NumpyEvaluator(target)
    params = jax.tree.map(lambda x: x[None], initialize(jax.random.PRNGKey(42)))
    for board in [chess.Board(), chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')]:
        result = make_search(32)(params, jax.tree.map(lambda x: x[None], rules.from_board(board)), np.array([1]))
        reference = RootCoverageValueSearch(evaluator, 32, None).run(board)
        assert rules.action_map(board)[int(result['actions'][0])].uci() == reference.move
        assert float(result['value'][0]) == pytest.approx(reference.value, abs=1e-5)
        assert float(result['root_coverage'][0]) == pytest.approx(reference.root_coverage)


def test_native_changed_pending_record_is_rejected(tmp_path):
    _, tasks = fixtures(tmp_path)
    native_games.run(tasks, tmp_path/'run', 2, 4, 2, 2, stop_after_chunks=1)
    from chessengine.gpu.protocol import load_state, save_state
    protocol = json.loads((tmp_path/'run/protocol.json').read_text())['identity']
    state = load_state(tmp_path/'run/continuation.pkl', protocol)
    state['next_task'] = 0
    save_state(tmp_path/'run/continuation.pkl', state, protocol)
    with pytest.raises(ValueError, match='Pending slots'):
        native_games.run(tasks, tmp_path/'run', 2, 4, 2, 2)
