import json
from pathlib import Path
import threading
import time
import chess
from chessengine.common import digest, write_json, read_json
from chessengine.features import record_board
from chessengine.move_policy_data import prepare, label


def rows_for(moves, split, game):
    board = chess.Board()
    result = []
    for uci in moves:
        board.push_uci(uci)
        result.append({**record_board(board), 'game_id': game, 'split': split, 'target': 0.5})
    return result


def test_reuse_quarantines_whole_games_and_refills_exact_counts(tmp_path):
    rows = (rows_for(['e2e4', 'e7e5'], 'train', 'train-game') +
            rows_for(['d2d4', 'd7d5'], 'validation', 'val-game') +
            rows_for(['c2c4', 'e7e5'], 'test', 'test-game'))
    rows[2] = {**rows[0], 'split': 'validation', 'game_id': 'val-game'}
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    refill = (rows_for(['g2g3', 'g7g6'], 'train', 'new-train') +
              rows_for(['b2b3', 'b7b6'], 'validation', 'new-val'))
    (tmp_path / 'refill.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in refill))
    write_json(tmp_path / 'reserved.json', [])
    config = {'preparation_mode': 'reuse', 'source': 'source.jsonl',
              'reuse_source_sha256': digest(source), 'refill_source': 'refill.jsonl',
              'exclusions': ['reserved.json'], 'positions': {'train': 2, 'validation': 2, 'test': 2}}
    original = source.read_bytes()
    result = prepare(config, tmp_path / 'out', tmp_path)
    assert result['counts'] == config['positions']
    assert result['audit']['refill_counts'] == {'train': 2, 'validation': 2, 'test': 0}
    assert set(result['audit']['quarantined_source_games']) == {'train-game', 'val-game'}
    selected = [json.loads(line) for line in (tmp_path / 'out/positions.jsonl').read_text().splitlines()]
    assert all('target' not in r for r in selected)
    assert source.read_bytes() == original
    assert prepare(config, tmp_path / 'out', tmp_path) == result


def test_two_label_workers_have_independent_one_thread_teachers(tmp_path, monkeypatch):
    rows = (rows_for(['e2e4', 'e7e5'], 'train', 'a') +
            rows_for(['d2d4', 'd7d5'], 'validation', 'b') +
            rows_for(['c2c4', 'e7e5'], 'test', 'c'))
    for row in rows:
        row.pop('target')
    source = tmp_path / 'positions.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    binary = tmp_path / 'teacher'
    binary.write_text('test-only')
    config = {'positions': {'train': 2, 'validation': 2, 'test': 2}, 'shard_size': 2,
              'label_workers': 2, 'teacher': {'binary': 'teacher', 'sha256': digest(binary), 'nodes': 10000}}
    write_json(tmp_path / 'positions.manifest.json', {'identity': {'config': config},
               'sha256': digest(source), 'counts': config['positions']})
    lock = threading.Lock()
    state = {'active': 0, 'peak': 0, 'analyses': 0}

    class Teacher:
        id = {'name': 'test-only'}

        def __enter__(self):
            with lock:
                state['active'] += 1
                state['peak'] = max(state['peak'], state['active'])
            return self

        def __exit__(self, *args):
            with lock:
                state['active'] -= 1

        def configure(self, options):
            assert 'MultiPV' not in options
            if 'Threads' in options:
                assert options['Threads'] == 1 and options['Hash'] == 32

        def analyse(self, board, limit, game):
            assert limit.nodes == 10000
            time.sleep(0.01)
            with lock:
                state['analyses'] += 1
            return {'pv': [list(board.legal_moves)[0]]}

    monkeypatch.setattr(chess.engine.SimpleEngine, 'popen_uci', lambda _: Teacher())
    first = label(config, tmp_path, tmp_path)
    assert state == {'active': 0, 'peak': 2, 'analyses': 6}
    assert [Path(c['path']).name for c in first['chunks']] == ['00000.jsonl', '00001.jsonl', '00002.jsonl']
    assert label(config, tmp_path, tmp_path) == first
    assert state['analyses'] == 6
    assert read_json(tmp_path / 'progress.json')['positions'] == 6
