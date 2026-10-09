import json
from pathlib import Path
import random

import chess
import chess.engine
import chess.pgn
import pytest

from chessengine.common import digest, json_hash, read_json, write_json
from chessengine.data import records, split_game
from chessengine.expand_data import expand, prepare_extension
from chessengine.features import position_key, record_board, restore


def inputs(tmp_path):
    rng = random.Random(99)
    games = []
    for _ in range(16):
        board = chess.Board()
        for _ in range(45):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
        game_id = json_hash({'initial': chess.STARTING_FEN, 'moves': [m.uci() for m in board.move_stack]})
        games.append((board, game_id))
    source = tmp_path / 'games.pgn'
    source.write_text('\n\n'.join(str(chess.pgn.Game.from_board(b)) for b, _ in games))
    base = tmp_path / 'base.jsonl'
    base_rows = [{**record_board(b), 'game_id': g, 'split': split,
                  'target': 0, 'phase': 'middlegame'}
                 for (b, g), split in zip(games[:3], ['train', 'validation', 'test'])]
    base.write_text(''.join(json.dumps(r) + '\n' for r in base_rows))
    reserved = tmp_path / 'reserved.json'
    write_json(reserved, {'development': [{**record_board(games[3][0]), 'source_game_id': games[3][1]}]})
    binary = tmp_path / 'teacher'
    binary.write_text('test executable identity')
    teacher = tmp_path / 'teacher.json'
    write_json(teacher, {'binary_sha256': digest(binary), 'sha256': digest(base),
                        'teacher': {'name': 'fake'}, 'nodes': 10000, 'threads': 1, 'hash_mb': 32})
    return dict(base=str(base), source=str(source), reserved=str(reserved), binary=str(binary),
                teacher_manifest=str(teacher), target_total=15, sampling_seed=2026,
                workers=2, chunk_size=3, output=str(tmp_path / 'out'), combined=str(tmp_path / 'combined'))


def test_extension_excludes_existing_and_reserved_games(tmp_path):
    config = inputs(tmp_path)
    result = prepare_extension(config)
    assert 15 <= result['total_positions'] <= 20
    path = Path(config['output']) / 'positions.jsonl'
    rows = list(records(path))
    base = list(records(config['base']))
    reserved = read_json(config['reserved'])['development']
    excluded = {r['game_id'] for r in base} | {r['source_game_id'] for r in reserved}
    keys = {position_key(restore(r)) for r in base + reserved}
    new_keys = [position_key(restore(r)) for r in rows]
    assert len(new_keys) == len(set(new_keys))
    assert not keys.intersection(new_keys)
    assert all(r['game_id'] not in excluded and r['split'] == split_game(r['game_id']) == 'train' for r in rows)
    assert prepare_extension(config) == result
    with pytest.raises(ValueError, match='mismatch'):
        prepare_extension({**config, 'target_total': 30})


def test_resumable_labels_preserve_base_and_reject_corruption(tmp_path, monkeypatch):
    config = inputs(tmp_path)
    calls = []

    class Teacher:
        id = {'name': 'fake'}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def configure(self, settings):
            pass

        def analyse(self, board, limit, game):
            assert limit.nodes == 10000
            calls.append(board.fen())
            return {'wdl': chess.engine.PovWdl(chess.engine.Wdl(600, 300, 100), board.turn)}

    monkeypatch.setattr(chess.engine.SimpleEngine, 'popen_uci', lambda _: Teacher())
    result = expand(config)
    assert len(calls) == result['additional_positions']
    assert Path(config['combined']).read_bytes().startswith(Path(config['base']).read_bytes())
    assert all(r['target'] == .5 for r in list(records(config['combined']))[3:])
    expand(config)
    assert len(calls) == result['additional_positions']
    chunk = next((Path(config['output']) / 'chunks').glob('*.jsonl'))
    chunk.write_text('corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        expand(config)


def test_teacher_binary_mismatch_rejected(tmp_path):
    config = inputs(tmp_path)
    Path(config['binary']).write_text('different binary')
    with pytest.raises(ValueError, match='teacher binary'):
        expand(config)


def test_expansion_comparison_report(tmp_path, monkeypatch):
    import numpy as np
    from chessengine.diagnostics import run_study
    from chessengine.expansion_report import generate
    from chessengine.model import build_model, export

    config = inputs(tmp_path)

    class Teacher:
        id = {'name': 'fake'}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def configure(self, settings):
            pass

        def analyse(self, board, limit, game):
            return {'wdl': chess.engine.PovWdl(chess.engine.Wdl(400, 300, 300), board.turn)}

    monkeypatch.setattr(chess.engine.SimpleEngine, 'popen_uci', lambda _: Teacher())
    expand(config)
    checkpoint = tmp_path / 'frozen.npz'
    export(build_model([16, 8]), checkpoint, {'hidden': [16, 8]})
    study = read_json('configs/supervised-200k.json')
    study.update(checkpoint=str(checkpoint), bootstrap_resamples=20)
    study['training'].update(hidden=[16, 8], epochs=2)
    previous, expanded = tmp_path / 'previous', tmp_path / 'expanded'
    for source, output in [(config['base'], previous), (config['combined'], expanded)]:
        run_study({**study, 'source': source, 'output': str(output)})
    report = tmp_path / 'report'
    result = generate(previous, expanded, report, config)
    assert result['validation_positions'] == 1
    assert result['new_positions'] > result['old_positions']
    assert (report / 'index.html').exists() and (report / 'data-size.png').exists()
    raw = np.load(expanded / 'comparison-predictions.npz')
    for row in result['seed_comparisons']:
        error = raw[f"new_{row['seed']}"] - raw['targets']
        assert np.mean(error**2) == pytest.approx(row['new']['mse'])
    combined = list(records(config['combined']))
    combined[1]['target'] = .99
    Path(config['combined']).write_text(''.join(json.dumps(r) + '\n' for r in combined))
    with pytest.raises(ValueError, match='Validation set changed'):
        generate(previous, expanded, report, config)
