import random
import chess
import chess.pgn
import numpy as np
import pytest
from chessengine.common import read_json, write_json, json_hash
from chessengine.compute_budget import ComputeBudget, BudgetExceeded
from chessengine.features import record_board
from chessengine.model import build_model, export, NumpyEvaluator
from chessengine.rl import play_game, selfplay
from chessengine.rl_improvement import choose_protocol, aggregate_finals, prepare_openings


class Unlimited:
    def check(self):
        pass
    def checkpoint(self):
        pass


def test_budget_persists_and_rejects_change(tmp_path, monkeypatch):
    now=[100.]
    monkeypatch.setattr('chessengine.compute_budget.time.perf_counter',lambda:now[0])
    monkeypatch.setattr('chessengine.compute_budget.time.time',lambda:now[0])
    path=tmp_path/'budget.json'
    with ComputeBudget(path,20) as b:
        now[0]+=7
        b.checkpoint()
        assert b.remaining==13
    with ComputeBudget(path,20) as b:
        now[0]+=13
        with pytest.raises(BudgetExceeded):
            b.check()
    assert read_json(path)['consumed']==20
    with pytest.raises(ValueError,match='mismatch'):
        ComputeBudget(path,30)


def test_v3_pending_resume_and_outcome(tmp_path):
    path=tmp_path/'model.npz'
    export(build_model([16,8],feature_version=3),path,{'hidden':[16,8]})
    cfg=dict(seed=42,learning_rate=.0001,batch_size=8,batches=2,games_per_batch=2,
             positions_per_game=4,nodes=100,max_plies=2,epsilon=0,exploration_plies=0,device='cpu')
    opening={'fen':'7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'}
    result,samples=play_game(NumpyEvaluator(path),opening,cfg,random.Random(42))
    assert samples[0].shape[1]==853 and samples[1].tolist()==[1]
    assert result['searched_moves']==1 and result['depth_one_moves']==1
    opening_path=tmp_path/'openings.json'
    write_json(opening_path,dict(train=[opening],development=[],test=[]))
    selfplay(path,opening_path,tmp_path/'full',cfg)
    class OneGame(Unlimited):
        completed=0
        def checkpoint(self):
            self.completed+=1
        def check(self):
            if self.completed:
                raise BudgetExceeded('test interruption')
    with pytest.raises(BudgetExceeded):
        selfplay(path,opening_path,tmp_path/'resume',cfg,budget=OneGame())
    assert (tmp_path/'resume/pending.pt').exists()
    assert not (tmp_path/'resume/resume.pt').exists()
    selfplay(path,opening_path,tmp_path/'resume',cfg,resume=True)
    for a,b in zip(NumpyEvaluator(tmp_path/'full/last.npz').layers,NumpyEvaluator(tmp_path/'resume/last.npz').layers):
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x,y)
    for batch in [1,2]:
        games=read_json(tmp_path/f'resume/games-{batch:03}.json')
        assert len({g['actor_sha256'] for g in games})==1


def test_pilot_selection_and_paired_aggregate():
    cfg=read_json('configs/rl-improvement.json')
    pilots=[dict(nodes=n,games=20,truncated=0,searched_moves=100,depth_one_moves=95,
                 mean_game_seconds=1,p90_plies=20) for n in cfg['nodes']]
    assert choose_protocol(pilots,cfg,100)['nodes']==512
    pilots[-1]['mean_game_seconds']=100
    assert choose_protocol(pilots,cfg,100)['nodes']==256
    for p in pilots:
        p['depth_one_moves']=89
    assert not choose_protocol(pilots,cfg,100)['approved']
    games=[dict(opening_id=str(i),candidate_color=color,score=score,capped=False)
           for i,score in enumerate([0.,.5,1.]) for color in ['white','black']]
    result=aggregate_finals([games,games,games],100,42)
    assert result['score']==.5 and result['seed_sd']==0 and result['pairs']==3
    assert result==aggregate_finals([games,games,games],100,42)
    with pytest.raises(ValueError):
        aggregate_finals([games,games[:-1]],20,42)


def test_fresh_openings_exclude_corpus_and_old_games(tmp_path):
    games,rows=[],[]
    for seed in range(4):
        board=chess.Board()
        rng=random.Random(seed)
        for _ in range(20):
            board.push(rng.choice(list(board.legal_moves)))
        game=chess.pgn.Game.from_board(board)
        game_id=json_hash({'initial':board.root().fen(),'moves':[m.uci() for m in board.move_stack]})
        while len(board.move_stack)>12:
            board.pop()
        rows.append({**record_board(board),'id':game_id[:12],'source_game_id':game_id})
        games.append(str(game))
    archive=tmp_path/'source.pgn'
    archive.write_text('\n\n'.join(games))
    corpus=tmp_path/'corpus.jsonl'
    import json
    corpus.write_text(json.dumps(dict(fen=rows[0]['fen'],game_id=rows[0]['source_game_id'],target='never consumed'))+'\n')
    old=tmp_path/'old.json'
    write_json(old,dict(train=[],development=[],test=[rows[1]]))
    cfg=dict(old_openings=str(old),corpus=str(corpus),archive=str(archive),final_pairs=2)
    out=tmp_path/'study'
    out.mkdir()
    path=prepare_openings(cfg,out,Unlimited())
    assert {r['source_game_id'] for r in read_json(path)['test']}=={r['source_game_id'] for r in rows[2:]}
    assert prepare_openings(cfg,out,Unlimited())==path
    archive.write_text(archive.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):
        prepare_openings(cfg,out,Unlimited())


def test_complete_study_smoke(tmp_path, monkeypatch):
    from chessengine.common import digest
    from chessengine.rl_improvement import run
    import chessengine.rl_improvement as study
    from test_learning import dataset
    baseline=tmp_path/'model.npz'
    export(build_model([16,8],feature_version=3),baseline,{'hidden':[16,8]})
    averaging=tmp_path/'averaging.json'
    write_json(averaging,dict(rl_initial_checkpoint=str(baseline),rl_initial_sha256=digest(baseline)))
    positions=[
        '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1',
        '7k/5Q2/7K/8/8/8/8/8 w - - 0 1',
        '7k/5Q2/5K2/8/8/8/8/8 w - - 0 1',
    ]
    openings=tmp_path/'openings.json'
    write_json(openings,{split:[dict(fen=fen,id=split,source_game_id=split)]
                         for split,fen in zip(['train','development','test'],positions)})
    data=tmp_path/'data.jsonl'
    dataset(data)
    cfg=read_json('configs/rl-improvement.json')
    cfg.update(averaging_results=str(averaging),output=str(tmp_path/'study'),old_openings=str(openings),
               corpus=str(data),development_data=str(data),archive=str(data),pilot_games=2,nodes=[128],
               game_targets=[10],development_pairs=1,final_pairs=1,max_plies=2,equal_time_ms=1,
               bootstrap_resamples=20)
    monkeypatch.setattr(study,'prepare_openings',lambda *_:openings)
    original=study.run_matches
    def checked_matches(protocol, output, budget):
        if read_json(protocol)['split']=='test':
            assert all((tmp_path/f'study/seed-{seed}/selection.json').exists() for seed in cfg['seeds'])
        return original(protocol,output,budget)
    monkeypatch.setattr(study,'run_matches',checked_matches)
    result=run(cfg)
    assert read_json(tmp_path/'study/status.json')['status']=='complete'
    assert result['aggregate']['games']==6
    from chessengine.rl_improvement_report import generate
    assert generate(tmp_path/'study',tmp_path/'report').exists()
    assert len(result['teacher_agreement'])==4
    assert run(cfg)['aggregate']==result['aggregate']
    with pytest.raises(ValueError,match='mismatch'):
        run({**cfg,'max_plies':3})
