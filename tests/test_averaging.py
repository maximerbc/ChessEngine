import copy
import numpy as np
import pytest
import torch
from chessengine.averaging import ensemble_predictions, paired_mean_member, run
from chessengine.averaging_report import generate
from chessengine.common import read_json, write_json, digest
from chessengine.model import build_model, NumpyEvaluator, load_export
from chessengine.train import update_average, fit, trusted_load
from test_learning import config, dataset


def test_average_arithmetic():
    model = build_model([8,4])
    average = copy.deepcopy(model)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2)
    update_average(average, model, 0)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(4)
    update_average(average, model, 1)
    assert all(torch.all(p == 3) for p in average.parameters())
    assert all(torch.all(p == 4) for p in model.parameters())
    members = np.array([[0., 1.], [1., 0.]])
    pred = ensemble_predictions(members)
    np.testing.assert_array_equal(pred, [.5,.5])
    ci = paired_mean_member(pred, members, np.array([0.,0.]), ['a','b'], 20, 42)
    assert ci['mse'] == [-.25,-.25]


def test_averaging_resume_trajectory_and_parity(tmp_path):
    source = tmp_path/'data'
    dataset(source)
    cfg = {**config(6), 'feature_version':3}
    fit(source, tmp_path/'plain', cfg)
    cfg['average_start_epoch'] = 2
    fit(source, tmp_path/'full', cfg)
    fit(source, tmp_path/'resume', {**cfg,'epochs':3})
    fit(source, tmp_path/'resume', cfg, resume=True)
    for filename in ['best.npz','last.npz','average-best.npz','average-last.npz']:
        a,b = [NumpyEvaluator(tmp_path/d/filename) for d in ['full','resume']]
        for p,q in zip(a.layers,b.layers):
            for x,y in zip(p,q):
                np.testing.assert_array_equal(x,y)
        if not filename.startswith('average'):
            original = NumpyEvaluator(tmp_path/'plain'/filename)
            for p,q in zip(a.layers,original.layers):
                for x,y in zip(p,q):
                    np.testing.assert_array_equal(x,y)
    full, resumed = [trusted_load(tmp_path/d/'resume.pt') for d in ['full','resume']]
    assert full['average_count'] == resumed['average_count'] == 5
    model, _ = load_export(tmp_path/'full/average-best.npz')
    model.eval()
    x = np.ones((2,853),dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy().squeeze(-1)
    np.testing.assert_allclose(expected,NumpyEvaluator(tmp_path/'full/average-best.npz').batch(x),atol=1e-6)


def test_averaging_study_smoke(tmp_path):
    source = tmp_path/'data'
    dataset(source)
    previous = tmp_path/'previous'
    rows=[]
    for seed in [42,43,44]:
        cfg={**config(4),'feature_version':3,'seed':seed}
        fit(source,previous/f'combined-{seed}',cfg)
        metrics=read_json(previous/f'combined-{seed}/metrics.json')
        rows.append(dict(name='combined',seed=seed,config=cfg,
                         checkpoint_sha256=digest(previous/f'combined-{seed}/best.npz'),
                         validation=dict(mse=metrics['best_validation_mse'])))
    write_json(previous/'results.json',dict(runs=rows,protocol=dict(identity=dict(source_sha256=digest(source)))))
    cfg=dict(source=str(source),previous=str(previous),output=str(tmp_path/'study'),expected_counts=[6,2],
             seeds=[42,43,44],average_start_epoch=2,resamples=20,bootstrap_seed=2026,tolerance=1e-6)
    result=run(cfg)
    assert len(result['entries'])==7
    assert generate(tmp_path/'study',tmp_path/'report').exists()
    assert run(cfg,resume=True)['averaged_mean_mse']==result['averaged_mean_mse']
    assert all(r['ordinary_reproduction_delta']==0 for r in result['entries'] if r['name'].startswith('average'))
    assert result['entries'][-1]['mse'] <= result['reference_mean_mse']+1e-12
    with pytest.raises(ValueError,match='mismatch'):
        run({**cfg,'average_start_epoch':3},resume=True)
