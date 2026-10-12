import random
import chess
import numpy as np
import pytest
import torch
from chessengine.common import digest, read_json, write_json
from chessengine.features import encode, feature_size, record_board
from chessengine.model import NumpyEvaluator, build_model, export, load_export
from chessengine.train import fit, trusted_load
from chessengine.fixed_data import canonical_audit, run, combination
from chessengine.fixed_report import generate
from chessengine.rl import play_game, selfplay
from test_learning import dataset, config


@pytest.mark.parametrize('version', [1, 2, 3])
def test_schema_export_resume(tmp_path, version):
    path = tmp_path / 'data'
    dataset(path)
    cfg = {**config(6), 'feature_version': version, 'scheduler': dict(mode='min', factor=.5, patience=0,
           threshold=1, threshold_mode='abs', min_lr=1e-5)}
    fit(path, tmp_path / 'full', cfg)
    fit(path, tmp_path / 'resume', {**cfg, 'epochs': 2})
    fit(path, tmp_path / 'resume', cfg, resume=True)
    full, resumed = [NumpyEvaluator(tmp_path / d / 'last.npz') for d in ['full', 'resume']]
    for p, q in zip(full.layers, resumed.layers):
        for a, b in zip(p, q):
            np.testing.assert_array_equal(a, b)
    a, b = [trusted_load(tmp_path / d / 'resume.pt') for d in ['full', 'resume']]
    assert a['scheduler'] == b['scheduler']
    assert a['scheduler']['_last_lr'][0] < cfg['learning_rate']
    model, _ = load_export(tmp_path / 'full/last.npz')
    model.eval()
    board = chess.Board()
    board.push_uci('e2e4')
    with torch.no_grad():
        expected = model(torch.from_numpy(encode(board, version))).item()
    assert full(board) == pytest.approx(expected, abs=1e-6)
    with pytest.raises(ValueError, match='mismatch'):
        fit(path, tmp_path / 'resume', {**cfg, 'learning_rate': .123}, resume=True)


def test_relative_geometry_history_and_material():
    for fen in [chess.STARTING_FEN, 'r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 20']:
        board = chess.Board(fen)
        np.testing.assert_array_equal(encode(board, 2), encode(board.mirror(), 2))
    board = chess.Board('r3k2r/8/8/8/3Pp3/8/8/R3K2R b Kq d3 0 20')
    x = encode(board, 2)
    np.testing.assert_array_equal(x[769:773], [0, 1, 1, 0])
    assert x[773 + chess.D6] == 1 and x[773 + chess.D3] == 0
    assert x[chess.E5] == 1  # Black pawn becomes the own pawn on e5.
    board = chess.Board()
    for move in ['g1f3','g8f6','f3g1','f6g8'] * 2:
        board.push_uci(move)
    before = record_board(board)
    for version in [2, 3]:
        np.testing.assert_array_equal(encode(board, version)[838:841], encode(board)[838:841])
    assert record_board(board) == before and encode(board, 2)[840] == 1
    initial = encode(chess.Board(), 3)
    np.testing.assert_array_equal(initial[841:851], np.ones(10))
    assert initial[851] == 1 and initial[852] == 0
    board = chess.Board('qq2k3/8/8/8/8/8/8/4K3 b - - 0 1')
    x = encode(board, 3)
    assert x[845] == 2 and x[851] == pytest.approx(8/24) and x[852] == pytest.approx(18/40)
    with pytest.raises(ValueError):
        feature_size(99)


def test_canonical_leakage_and_rl_rejection(tmp_path):
    board = chess.Board()
    rows = [{**record_board(b), 'split': split} for b, split in [(board,'train'),(board.mirror(),'validation')]]
    with pytest.raises(ValueError, match='Canonical'):
        canonical_audit(rows)
    path = tmp_path / 'relative.npz'
    export(build_model([16,8], feature_version=2), path, {'hidden':[16,8]})
    evaluator = NumpyEvaluator(path)
    with pytest.raises(ValueError, match='version 1 only'):
        play_game(evaluator, {}, {}, random.Random(42))
    with pytest.raises(ValueError, match='version 1 only'):
        selfplay(path, 'missing', tmp_path / 'rl', {})


def test_fixed_study_smoke(tmp_path):
    path = tmp_path / 'data'
    dataset(path)
    historic = tmp_path / 'historic'
    runs = []
    for seed in [42,43,44]:
        result = fit(path, historic / f'baseline-all-seed-{seed}', {**config(2),'seed':seed})
        runs.append({'experiment':'baseline','size':'all','seed':seed,'data_sha256':digest(path),
                     'checkpoint_sha256':digest(result['checkpoint']),
                     'validation':{'mse':result['best_validation_mse']}})
    write_json(historic / 'results.json', {'runs':runs})
    cfg = read_json('configs/fixed-data-improvements.json')
    cfg.update(source=str(path), historical=str(historic), output=str(tmp_path / 'study'),
               expected_counts=[6,2], bootstrap_resamples=20, training=config(2),
               minimum_reduction=-1, minimum_seeds=0)
    result = run(cfg)
    assert len(result['runs']) == 18
    report = generate(tmp_path / 'study', tmp_path / 'report')
    assert '6 training positions' in report.read_text()
    assert (report.parent / 'curves.png').stat().st_size > 1000
    assert run(cfg, resume=True)['summary'] == result['summary']
    assert read_json(tmp_path / 'study/reproduction-42.json')['passed']
    original_targets = np.load(tmp_path / 'study/reference-42/predictions.npz')['targets']
    for row in result['runs']:
        stored = np.load(tmp_path / 'study' / f"{row['name']}-{row['seed']}" / 'predictions.npz')
        np.testing.assert_array_equal(stored['targets'], original_targets)
    with pytest.raises(ValueError, match='mismatch'):
        run({**cfg,'minimum_reduction':.1}, resume=True)
    failed = [dict(r, promising=False) for r in result['summary']]
    assert combination(failed, cfg['treatments']) is None
