"""Research-integrity checks independent of accelerator/training dependencies."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile

import chess
import chess.pgn
import numpy as np
import pytest
from fastapi.testclient import TestClient
from chessengine.features import record_board, position_key, restore
from chessengine.common import json_hash
from chessengine.data import split_game, records
from chessengine.gpu.protocol import sha, atomic_json, LEGACY_SHA, require_execution, identity
from chessengine.gpu.dataset import prepare, audit_positions
from chessengine.gpu.forecast import forecast
from chessengine.gpu.backup import verify
from chessengine.gpu.evaluation import qualify, seal_selections
from chessengine.puct_serving import ValueSearch


def test_legacy_assets_and_historical_protocol_are_immutable():
    for path in ['artifacts/models/supervised.npz', 'web/public/models/supervised.npz']:
        assert sha(path) == LEGACY_SHA
    config = json.loads(Path('configs/evolutionary.json').read_text())
    assert config['workers'] == 2 and config['population'] == 16
    from chessengine.evolution_model import PARAMETER_COUNT
    assert PARAMETER_COUNT == 324
    planned = json.loads(Path('configs/gpu-study.json').read_text())
    assert sum(planned['supervised']['counts'].values()) == 1000000
    assert planned['supervised']['counts']['train'] == 800000
    assert planned['evolution']['population'] * planned['evolution']['generations'] * 48 * 3 == 368640
    assert planned['budget_cad'] is None and planned['provider'] is None


def test_exact_source_game_splits_retention_and_relabeling(tmp_path):
    source = tmp_path / 'source.pgn'
    rng = np.random.default_rng(4)
    historical = None
    with source.open('w') as stream:
        for game_index in range(80):
            board = chess.Board()
            moves = []
            for _ in range(25):
                if board.is_game_over(claim_draw=True):
                    break
                move = list(board.legal_moves)[int(rng.integers(board.legal_moves.count()))]
                moves.append(move)
                board.push(move)
            game = chess.pgn.Game()
            node = game
            for move in moves:
                node = node.add_variation(move)
            stream.write(str(game) + '\n\n')
            game_id = json_hash({'initial': chess.STARTING_FEN, 'moves': [m.uci() for m in moves]})
            if historical is None and split_game(game_id) == 'train':
                historical = dict(record_board(board), game_id=game_id, source='fixture', split='train', target=.9)
    old = tmp_path / 'old.jsonl'
    old.write_text(json.dumps(historical) + '\n')
    reserved = tmp_path / 'reserved.json'
    atomic_json(reserved, {'reserved': []})
    sources = tmp_path / 'sources.json'
    atomic_json(sources, [{'url':'https://example.org/source.pgn','file':str(source),'sha256':sha(source)}])
    config = dict(counts={'train':4,'validation':2,'test':2}, retain_training=str(old),
                  historical_datasets=[str(old)],reserved=[str(reserved)],sampling_seed=42)
    result = prepare(config,sources,tmp_path/'prepared')
    assert result['counts'] == config['counts'] and result['retained_training'] == 1
    rows = list(records(tmp_path/'prepared/positions.jsonl'))
    assert len(rows) == 8 and all('target' not in row for row in rows)
    assert rows[0]['fen'] == historical['fen']
    assert len({position_key(chess.Board(r['fen'])) for r in rows}) == 8
    assert audit_positions(tmp_path/'prepared/positions.jsonl',config['counts']) == config['counts']
    assert prepare(config,sources,tmp_path/'prepared') == result
    damaged = tmp_path/'damaged.jsonl'
    damaged.write_text('\n'.join(json.dumps(r) for r in [*rows,rows[0]])+'\n')
    with pytest.raises(ValueError,match='duplicate'):
        audit_positions(damaged,config['counts'])


def test_reserved_puzzles_exclude_trajectories_and_source_site(tmp_path):
    from chessengine.gpu.dataset import puzzle_exclusions,site_id,research_exclusions
    path = tmp_path/'puzzles.csv'
    path.write_text('FEN,Moves,GameUrl\n'+chess.STARTING_FEN+',e2e4 e7e5,https://lichess.org/AbCd1234/black#4\n')
    positions,sites = puzzle_exclusions([path])
    assert len(positions) == 3 and sites == {'AbCd1234'}
    assert site_id('http://en.lichess.org/AbCd1234') == 'AbCd1234'
    board = chess.Board()
    board.push_uci('g1f3')
    history = tmp_path/'research.json'
    atomic_json(history,dict(record_board(board),source_game_id='heldout'))
    keys,games = research_exclusions([str(history)])
    assert len(keys) == 2 and games == {'heldout'}


def test_public_gpu_report_omits_raw_examples(tmp_path):
    from chessengine.gpu.report import publish
    config = json.loads(Path('configs/gpu-study.json').read_text())
    publish(config,dict(status='pilot_failed',summary={'games':[record_board(chess.Board())],
        'cap_fraction':.8,'optimizer':{'parameters':[1,2,3]}},private='/Users/private'),tmp_path/'report')
    text = (tmp_path/'report/evidence.json').read_text()
    assert 'cap_fraction' in text and 'initial_fen' not in text and 'parameters' not in text and '/Users/' not in text


def test_archive_real_restoration_and_checksum_rejection(tmp_path):
    archive = tmp_path/'archive.tar.gz'
    content = b'preserved optimizer and history\x00'
    with tarfile.open(archive,'w:gz') as tar:
        member = tarfile.TarInfo('private/state.bin')
        member.size = len(content)
        tar.addfile(member,io.BytesIO(content))
    manifest = tmp_path/'inventory.json'
    atomic_json(manifest,dict(archive_sha256=sha(archive),files=[dict(path='private/state.bin',bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest())]))
    assert verify(archive,manifest,tmp_path/'restored')['verified_files'] == 1
    assert (tmp_path/'restored/private/state.bin').read_bytes() == content
    with pytest.raises(ValueError,match='unused'):
        verify(archive,manifest,tmp_path/'restored')
    archive.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):
        verify(archive,manifest)


def test_forecast_includes_all_scenarios_pilot_evaluation_and_contingency():
    config = json.loads(Path('configs/gpu-study.json').read_text())['evolution']
    measurement = dict(tournament_games_per_second=1.,label_positions_per_second=100.,
        supervised_seconds_per_epoch=60.,cpu_evaluation_games_per_second=.5,
        compilation_seconds=10.,measured_on_nvidia=False,hardware='synthetic arithmetic fixture',workload_sha256='fixture')
    result = forecast(measurement,.1,10,.1,config)
    assert [r['main_games'] for r in result['scenarios']] == [184320,368640,737280]
    assert all(r['pilot_games'] == 30720 and r['evaluation_games'] > 9600 for r in result['scenarios'])
    assert all(r['cost_with_20_percent_contingency'] > r['quoted_compute_cost'] * 1.19 for r in result['scenarios'])
    assert 'not a GPU quote' in result['status']
    with pytest.raises(ValueError):
        forecast({},.1,10,.1,config)


def test_no_spending_authorized_by_default_or_unreviewed_receipt(tmp_path):
    config = {'test':'protocol'}
    path = tmp_path/'approval.json'
    atomic_json(path,dict(protocol_sha256=identity(config),budget_cad=0,provider='fixture',authorized_by_user=False))
    with pytest.raises(ValueError):
        require_execution(config,path)
    atomic_json(path,dict(protocol_sha256=identity(config),budget_cad=5,provider='fixture',authorized_by_user=True,
                          stage='benchmark'))
    assert require_execution(config,path,'benchmark')['budget_cad'] == 5
    with pytest.raises(ValueError):
        require_execution(config,path)


def gate_opening(index):
    board = chess.Board()
    for bit,square in enumerate([chess.A1,chess.B1,chess.C1,chess.A2,chess.B2,chess.C2,chess.A7,chess.B7]):
        if index & (1 << bit):
            board.remove_piece_at(square)
    board.castling_rights = board.clean_castling_rights()
    assert board.is_valid()
    return dict(record_board(board),id=str(index),source_game_id=str(index))


def winning_game(index, color, checkpoint='fixture'):
    # Both candidate colors win legal games from the same distinct opening.
    board = restore(gate_opening(index))
    moves = ['e2e4','f7f6','d2d4','g7g5','d1h5'] if color == 'white' else ['f2f3','e7e5','g2g4','d8h4']
    for move in moves:
        board.push_uci(move)
    assert board.is_checkmate()
    return dict(record_board(board),opening_id=str(index),source_game_id=str(index),color=color,
                score=1.,capped=False,result=board.result(),checkpoint_sha256=checkpoint)


def gate_evidence():
    ledgers = [dict(seed=seed,stage='qualification',selection_sha256='fixture',
                    games=[winning_game(i,c) for i in range(200) for c in ['white','black']]) for seed in [42,43,44]]
    opponents = {'classical':ledgers,'supervised':copy.deepcopy(ledgers)}
    for name,group in opponents.items():
        for ledger in group:
            ledger['config'] = dict(opponent_sha256='handcrafted-v1' if name == 'classical' else LEGACY_SHA,
                candidate_sha256='fixture',strategy='puct-uniform-v1',opponent_strategy='alpha-beta',time_ms=1500,max_plies=512)
    return dict(seeds=[42,43,44],opening_ids=[str(i) for i in range(200)],opening_records=[gate_opening(i) for i in range(200)],
                opponents=opponents)


def test_qualification_replays_results_requires_pairs_and_all_seeds():
    evidence = gate_evidence()
    assert qualify(evidence,'qualification','fixture',100)['passed']
    damaged = copy.deepcopy(evidence)
    damaged['opponents']['classical'][0]['games'][0]['score'] = 0
    with pytest.raises(ValueError,match='result'):
        qualify(damaged,'qualification','fixture',100)
    damaged = copy.deepcopy(evidence)
    damaged['opponents']['classical'][0]['games'].pop()
    with pytest.raises(ValueError,match='pair'):
        qualify(damaged,'qualification','fixture',100)
    with pytest.raises(ValueError,match='stage'):
        qualify(evidence,'final','fixture',100)


def test_conservative_caps_can_block_promotion():
    evidence = gate_evidence()
    for opponent in evidence['opponents'].values():
        for ledger in opponent:
            for row in ledger['games'][:24]:
                row.update(record_board(restore(gate_opening(int(row['opening_id'])))))
                row.update(capped=True,result='*',score=.5)
    assert not qualify(evidence,'qualification','fixture',100)['passed']


def test_wrong_reference_and_reused_openings_cannot_qualify():
    evidence = gate_evidence()
    evidence['opponents']['supervised'][1]['config']['opponent_sha256'] = 'different-supervised'
    with pytest.raises(ValueError,match='Reference identity'):
        qualify(evidence,'qualification','fixture',100)
    evidence = gate_evidence()
    evidence['opening_records'][1] = dict(evidence['opening_records'][0],id='1',source_game_id='1')
    with pytest.raises(ValueError,match='unique'):
        qualify(evidence,'qualification','fixture',100)


def test_selection_seal_cannot_change_after_final(tmp_path):
    files = []
    for seed in [42,43,44]:
        path = tmp_path/f'{seed}.npz'
        path.write_bytes(str(seed).encode())
        files.append(dict(seed=seed,file=str(path)))
    seal_selections(files,tmp_path/'seal.json')
    Path(files[0]['file']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='already sealed'):
        seal_selections(files,tmp_path/'seal.json')


def gpu_checkpoint(path):
    widths = [853,128,64,1]
    arrays = {f'{name}{i}':np.zeros(shape,dtype=np.float32) for i,(a,b) in enumerate(zip(widths,widths[1:]))
              for name,shape in [('w',(b,a)),('b',(b,))]}
    arrays['metadata'] = np.array(json.dumps(dict(format_version=1,feature_version=3,hidden=[128,64],
        architecture=widths,kind='supervised',evidence='fixture',study='gpu-value-v1',seed=42,
        protocol_id='fixture',training_positions=800000)))
    np.savez_compressed(path,**arrays)


def test_v4_api_puct_and_comparison_dispatch_without_training_imports(tmp_path,monkeypatch):
    from chessengine.api import create_app
    from chessengine.model import NumpyEvaluator
    path = tmp_path/'gpu.npz'
    gpu_checkpoint(path)
    row = dict(id='supervised-1m',file=path.name,name='Augmented Supervised',sha256=sha(path),
        architecture=[853,128,64,1],feature_version=3,evidence_id='fixture',qualified=True,
        qualification_sha256='a'*64,final_sha256='b'*64,
        search=dict(strategy='puct-uniform-v1',c_puct=1.5,priors='uniform-legal',simulations=32))
    manifest = tmp_path/'manifest.json'
    import shutil
    shutil.copy2('artifacts/models/supervised.npz',tmp_path/'supervised.npz')
    old = NumpyEvaluator(tmp_path/'supervised.npz')
    old_row = dict(id='supervised',file='supervised.npz',name='Supervised',sha256=LEGACY_SHA,
        architecture=[841,*old.metadata['hidden'],1],feature_version=1,search={'strategy':'alpha-beta'},evidence_id='original-study')
    atomic_json(manifest,dict(release_version=4,default='supervised',models=[old_row,row]))
    monkeypatch.setenv('REQUIRE_MODELS','0')
    with TestClient(create_app(manifest)) as client:
        response = client.post('/analyze',json=dict(model_id='supervised-1m',simulations=4,time_ms=1500))
        assert response.status_code == 200
        result = response.json()
        assert result['strategy'] == 'puct-uniform-v1' and result['simulations'] == 4 and result['nodes'] == 0
        assert result['best_move'] in {m.uci() for m in chess.Board().legal_moves}
        monkeypatch.setattr('chessengine.api.stockfish_assessment',lambda *_:{'status':'unavailable'})
        comparison = client.post('/evaluate-comparison',json={'model_id':'supervised-1m'}).json()
        assert comparison['search']['strategy'] == 'puct-uniform-v1'
        assert comparison['search']['simulations'] == 32 and 'nodes' not in comparison['search']['limits']
    a = ValueSearch(NumpyEvaluator(path),4,None).run(chess.Board())
    b = ValueSearch(NumpyEvaluator(path),4,None).run(chess.Board())
    assert a.move == b.move and a.value == b.value
    row['qualified'] = False
    atomic_json(manifest,dict(release_version=4,default='supervised',models=[old_row,row]))
    with pytest.raises(ValueError,match='qualification'):
        create_app(manifest)


def test_parallel_cpu_matches_have_the_same_fixed_budget_games(tmp_path):
    from chessengine.gpu.evaluation import cpu_matches
    model = tmp_path/'value.npz'
    gpu_checkpoint(model)
    starts = [dict(record_board(chess.Board()),id=str(i),source_game_id=str(i)) for i in range(2)]
    parallel = cpu_matches(str(model),'handcrafted',starts,tmp_path/'parallel.json',simulations=2,
                           max_plies=1,opponent_strategy='puct-uniform-v1',workers=2)
    serial = cpu_matches(str(model),'handcrafted',starts,tmp_path/'serial.json',simulations=2,
                         max_plies=1,opponent_strategy='puct-uniform-v1')
    assert [(r['fen'],r['score'],r['result']) for r in parallel['games']] == [
        (r['fen'],r['score'],r['result']) for r in serial['games']]
    assert len(cpu_matches(str(model),'handcrafted',starts,tmp_path/'parallel.json',simulations=2,
                          max_plies=1,opponent_strategy='puct-uniform-v1',workers=2)['games']) == 4
