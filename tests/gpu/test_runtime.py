"""Small CPU JAX checks; these are not NVIDIA throughput or strength evidence."""
# ruff: noqa: E402
import json
import chess
import numpy as np
import pytest

jax = pytest.importorskip('jax')
pytest.importorskip('evosax')
from chessengine.gpu import rules
from chessengine.gpu.network import initialize, export, apply
from chessengine.gpu.search import make_search
from chessengine.model import NumpyEvaluator
from chessengine.puct_serving import ValueSearch
from chessengine.gpu.protocol import atomic_json


@pytest.mark.parametrize('fen', [chess.STARTING_FEN,
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
    '4k3/P7/8/8/8/8/8/4K3 w - - 0 1',
    '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
    '4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1',
    '8/8/8/8/8/8/6k1/4K2R w - - 99 1',
    '8/8/8/8/8/8/6k1/4K2R w - - 100 1',
    '7k/6Q1/6K1/8/8/8/8/8 b - - 100 1',
    '8/8/8/8/8/8/6k1/4K3 w - - 0 1'])
def test_rules_and_features(fen):
    board = chess.Board(fen)
    assert board.is_valid()
    state = rules.from_board(board)
    from chessengine.engine import Handcrafted
    assert float(rules.handcrafted(state)) == pytest.approx(Handcrafted()(board),abs=1e-6)


def test_original_history_claim_on_intended_move():
    board = chess.Board()
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1']:
        board.push_uci(move)
    state = rules.from_board(board)
    assert bool(rules.terminal(state)[0])
    assert not bool(rules.terminal(rules.from_board(chess.Board(board.fen())))[0])


def test_random_legal_trajectory_and_pinned_actions():
    from chessengine.puct_actions import tables
    from pgx._src.games import chess as pg
    assert np.array_equal(np.array(tables()['to_plane']), np.asarray(pg.TO_PLANE))
    assert np.array_equal(np.array(tables()['from_plane']), np.asarray(pg.FROM_PLANE))
    board = chess.Board()
    state = rules.from_board(board)
    step = jax.jit(rules.step)
    rng = np.random.default_rng(11)
    for _ in range(80):
        if board.is_game_over(claim_draw=True):
            break
        mapping = rules.action_map(board)
        action = int(rng.choice(sorted(mapping)))
        board.push(mapping[action])
        state = step(state, np.int32(action))
        rules.assert_parity(board, state)


def test_numpy_export_and_deterministic_search_parity(tmp_path):
    from chessengine.features import encode
    params = initialize(jax.random.PRNGKey(42))
    target = tmp_path / 'network.npz'
    export(params, target, 'supervised', 'test-only')
    cpu = NumpyEvaluator(target)
    board = chess.Board()
    assert cpu(board) == pytest.approx(float(apply(params, encode(board, 3))), abs=1e-6)
    state = rules.from_board(board)
    p = jax.tree.map(lambda x: x[None], params)
    states = jax.tree.map(lambda x: x[None], state)
    search = make_search(16)
    a = search(p, states, np.array([1]))
    b = search(p, states, np.array([1]))
    assert np.array_equal(np.asarray(a['visits']), np.asarray(b['visits']))
    reference = ValueSearch(cpu, 16, None).run(board)
    assert rules.action_map(board)[int(a['actions'][0])].uci() == reference.move
    assert float(a['value'][0]) == pytest.approx(reference.value, abs=1e-6)


def test_deep_uniform_search_visits_and_mate_sign(tmp_path):
    from chessengine.puct_serving import UniformValue
    from chessengine.puct import PUCT
    params = jax.tree.map(lambda x:np.zeros_like(x),initialize(jax.random.PRNGKey(42)))
    target = tmp_path/'zero.npz'
    export(params,target,'evolutionary','test-only')
    evaluator = NumpyEvaluator(target)
    for board in [chess.Board(),chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')]:
        state = rules.from_board(board)
        result = make_search(64)(jax.tree.map(lambda x:x[None],params),
            jax.tree.map(lambda x:x[None],state),np.array([1]))
        reference = PUCT(UniformValue(evaluator),64,time_ms=None).run(board)
        expected = np.zeros(4672,np.int32)
        mapping = rules.action_map(board)
        inverse = {move:label for label,move in mapping.items()}
        for move,visits in zip(reference.moves,reference.visits):
            expected[inverse[move]] = visits
        assert np.array_equal(np.asarray(result['visits'])[0],expected)
        assert float(result['value'][0]) == pytest.approx(reference.value,abs=1e-6)
        if board.fullmove_number == 1 and len(board.piece_map()) == 3:
            assert reference.value == 1


def test_full_evolution_pending_population_resume(tmp_path,monkeypatch):
    from chessengine.gpu.evolution import evolve
    from chessengine.gpu.protocol import identity
    from chessengine.features import record_board
    opponent = tmp_path/'supervised.npz'
    export(initialize(jax.random.PRNGKey(42)),opponent,'supervised','test-only',dict(seed=42))
    openings = tmp_path/'openings.json'
    atomic_json(openings,{'train':[dict(record_board(chess.Board()),id=str(i),source_game_id=str(i)) for i in range(3)]})
    config = dict(seeds=[42,43,44],pilot_seed=20261121,population=4,games_per_candidate=6,
        generations=2,pilot_generations=10,sigma=.05,simulations=2,max_plies=2,pilot_decision=str(tmp_path/'gate.json'))
    atomic_json(config['pilot_decision'],dict(passed=True,config_sha256=identity(dict(config,games_per_candidate=48))))
    interrupted = [True]
    def fake_run(tasks,directory,*_):
        if directory.name=='generation-002' and interrupted[0]:
            raise KeyboardInterrupt()
        return dict(active_seconds=0.,games=[dict(candidate_index=t['candidate_index'],group=t['group'],
            opening=t['opening'],color=t['color'],score=.5,capped=True,result='*') for t in tasks])
    monkeypatch.setattr('chessengine.gpu.games.run',fake_run)
    with pytest.raises(KeyboardInterrupt):
        evolve(config,openings,str(opponent),tmp_path/'resumed',42)
    interrupted[0] = False
    evolve(config,openings,str(opponent),tmp_path/'resumed',42)
    evolve(config,openings,str(opponent),tmp_path/'whole',42)
    a,b = NumpyEvaluator(tmp_path/'resumed/mean-002.npz'),NumpyEvaluator(tmp_path/'whole/mean-002.npz')
    assert all(np.array_equal(x,y) for pair_a,pair_b in zip(a.layers,b.layers) for x,y in zip(pair_a,pair_b))


def test_supervised_interruption_resume_matches_whole_run(tmp_path):
    from chessengine.gpu.supervised import fit
    from chessengine.gpu.protocol import sha
    data = tmp_path / 'data'
    data.mkdir()
    rng = np.random.default_rng(1)
    for split, n in [('train', 8), ('validation', 4)]:
        np.save(data / f'{split}-x.npy', rng.normal(size=(n, 853)).astype('float32'))
        np.save(data / f'{split}-y.npy', rng.uniform(-1, 1, n).astype('float32'))
    atomic_json(data / 'features.json', {'files': {p.name: sha(p) for p in data.glob('*.npy')}})
    atomic_json(data / 'labels.json', {'test': 'fixture'})
    config = dict(seeds=[42,43,44], counts={'train':8,'validation':4}, learning_rate=.001,
                  weight_decay=.0001,batch_size=4,epochs=2,patience=6)
    fit(config, data, tmp_path / 'resumed', 42, max_batches=1)
    resumed = fit(config, data, tmp_path / 'resumed', 42)
    whole = fit(config, data, tmp_path / 'whole', 42)
    for a, b in zip(jax.tree.leaves(resumed['params']), jax.tree.leaves(whole['params'])):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert resumed['history'] == whole['history']
    assert not json.loads((tmp_path/'resumed/selection.json').read_text())['final_test_opened']


def test_game_resume_keeps_caps_distinct_from_draws(tmp_path):
    from chessengine.gpu.games import tasks_for, run
    from chessengine.features import record_board
    target = tmp_path / 'value.npz'
    export(initialize(jax.random.PRNGKey(42)), target, 'evolutionary', 'test-only')
    opening = dict(record_board(chess.Board()), id='start', source_game_id='fixture')
    tasks = tasks_for(target, 'handcrafted', [opening], 'classical')
    run(tasks, tmp_path/'resumed', 2, max_plies=2, batch_size=2, max_moves=1)
    resumed = run(tasks, tmp_path/'resumed', 2, max_plies=2, batch_size=2)
    whole = run(tasks, tmp_path/'whole', 2, max_plies=2, batch_size=2)
    assert resumed['games'] == whole['games']
    assert all(g['result'] == '*' and g['capped'] and g['score'] == .5 for g in whole['games'])


def test_separable_cma_ask_tell_resume(tmp_path):
    from evosax.algorithms import Sep_CMA_ES
    from chessengine.gpu.protocol import save_state, load_state
    strategy = Sep_CMA_ES(4, initialize(jax.random.PRNGKey(42)))
    config = strategy.default_params.replace(std_init=.05)
    state = strategy.init(jax.random.PRNGKey(3), initialize(jax.random.PRNGKey(4)), config)
    population, state = strategy.ask(jax.random.PRNGKey(5), state, config)
    save_state(tmp_path/'optimizer.pkl', jax.device_get((state,population)), 'fixture')
    restored, pending = load_state(tmp_path/'optimizer.pkl', 'fixture')
    fitness = np.array([1.,2.,3.,4.])
    a, _ = strategy.tell(jax.random.PRNGKey(6), population, fitness, state, config)
    b, _ = strategy.tell(jax.random.PRNGKey(6), pending, fitness, restored, config)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert np.array_equal(np.asarray(x),np.asarray(y))
