import json
from pathlib import Path

import chess
import pytest
from chessengine.features import record_board
from chessengine.gpu.evaluation import cpu_matches
from chessengine.gpu.protocol import sha


def test_both_evaluators_use_alpha_beta_and_strategy_change_cannot_resume(tmp_path, monkeypatch):
    import chessengine.gpu.evaluation as evaluation
    calls = []
    original = evaluation.search_for
    def track(evaluator, strategy, nodes, time_ms, simulations):
        calls.append((strategy, nodes, time_ms))
        return original(evaluator, strategy, nodes, time_ms, simulations)
    monkeypatch.setattr(evaluation, 'search_for', track)
    opening = dict(record_board(chess.Board()), id='one', source_game_id='one')
    output = tmp_path/'matches.json'
    ledger = cpu_matches('artifacts/models/supervised-1m.npz', 'artifacts/models/supervised.npz',
        [opening], output, time_ms=20, max_plies=1, simulations=None, candidate_strategy='alpha-beta')
    assert calls == [('alpha-beta', 2**31-1, 20)] * 2
    assert len(ledger['games']) == 2
    assert sum(r['search_metrics']['candidate_nodes'] for r in ledger['games']) > 0
    assert sum(r['search_metrics']['reference_nodes'] for r in ledger['games']) > 0
    assert ledger['config']['simulations'] is None
    assert all(r['search_metrics']['candidate_simulations'] == 0 for r in ledger['games'])
    assert len(cpu_matches('artifacts/models/supervised-1m.npz', 'artifacts/models/supervised.npz',
        [opening], output, time_ms=20, max_plies=1, simulations=None,
        candidate_strategy='alpha-beta')['games']) == 2
    assert len(calls) == 2  # Complete games were not replayed on resume.
    with pytest.raises(ValueError, match='protocol changed'):
        cpu_matches('artifacts/models/supervised-1m.npz', 'artifacts/models/supervised.npz',
            [opening], output, time_ms=20, max_plies=1, simulations=128)


def test_parallel_workers_preserve_candidate_search_strategy(tmp_path):
    starts = [dict(record_board(chess.Board()), id=str(i), source_game_id=str(i)) for i in range(2)]
    ledger = cpu_matches('artifacts/models/supervised-1m.npz', 'handcrafted', starts,
        tmp_path/'parallel.json', time_ms=20, max_plies=1, simulations=None,
        candidate_strategy='alpha-beta', workers=2)
    assert len(ledger['games']) == 4
    assert ledger['config']['strategy'] == 'alpha-beta'
    assert sum(r['search_metrics']['candidate_nodes'] for r in ledger['games']) > 0
    assert all(r['search_metrics']['candidate_simulations'] == 0 for r in ledger['games'])
    from chessengine.strength_screen import summarize
    assert summarize(ledger)['coverage_fraction'] is None


def test_ablation_does_not_change_serving_or_frozen_historical_protocols():
    manifest = json.loads(Path('artifacts/models/manifest.json').read_text())
    augmented = next(r for r in manifest['models'] if r['id'] == 'supervised-1m')
    assert augmented['search']['strategy'] == 'puct-uniform-v1'
    assert manifest['default'] == 'supervised'
    assert sha('artifacts/models/supervised.npz') == 'ff3436d07d18eb4953e24e6c83b55d25bbb46b44094ed347f99320148ae04c4b'
    from chessengine.gpu.evaluation import qualify
    from test_gpu_study import gate_evidence
    evidence = gate_evidence()
    evidence['opponents']['classical'][0]['config']['strategy'] = 'alpha-beta'
    with pytest.raises(ValueError, match='Reference identity'):
        qualify(evidence, 'qualification', 'fixture', 100)
