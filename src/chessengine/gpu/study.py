"""Development selection precedes qualification and fresh final pools."""
import json
from pathlib import Path
import numpy as np
from .evaluation import cpu_matches, pair_scores, seal_selections, qualify
from .protocol import atomic_json, sha, SEEDS, LEGACY_SHA
from ..model import NumpyEvaluator


def select_evolution(config, directory, openings, legacy_supervised, output, workers=1):
    directory = Path(directory)
    if sha(legacy_supervised) != LEGACY_SHA:
        raise ValueError('Reference must be the preserved 80k Supervised model')
    rows = []
    for seed in SEEDS:
        run = directory / f'seed-{seed}'
        history = json.loads((run / 'history.json').read_text())
        if len(history) != config['generations']:
            raise ValueError('Complete all main generations before selecting')
        checkpoints = [run / 'initial.npz' if i == 0 else run / f'mean-{i:03}.npz'
                       for i in config['development_checkpoints']]
        candidates = []
        for checkpoint in checkpoints:
            scores = []
            for name, opponent in [('classical', 'handcrafted'), ('supervised', legacy_supervised)]:
                ledger = cpu_matches(str(checkpoint), opponent, openings['development'],
                    run / 'development' / checkpoint.stem / f'{name}.json',
                    config['cpu_match_time_ms'], config['simulations'], config['max_plies'], workers=workers)
                scores.append(np.mean(list(pair_scores(ledger['games']).values())))
            candidates.append(dict(file=str(checkpoint), score=float(np.mean(scores))))
        # Earliest checkpoint wins a tie; final matches cannot change this choice.
        selected = max(candidates, key=lambda r: r['score'])
        atomic_json(run / 'development-selection.json', dict(selected=selected, candidates=candidates))
        rows.append(dict(seed=seed, file=selected['file']))
    return seal_selections(rows, output)


def select_supervised(directory, output):
    rows = []
    for seed in SEEDS:
        path = Path(directory) / f'seed-{seed}'
        selected = json.loads((path / 'selection.json').read_text())
        if (selected['seed'] != seed or selected['final_test_opened']
                or sha(path / 'best.npz') != selected['checkpoint_sha256']):
            raise ValueError('Expected a completed validation-only selection')
        rows.append(dict(seed=seed, file=str(path / 'best.npz')))
    return seal_selections(rows, output)


def evaluate(config, selections_path, openings_path, legacy_supervised, stage, directory, workers=1):
    if stage not in {'qualification', 'final'}:
        raise ValueError('Expected qualification or final stage')
    directory = Path(directory)
    selections = json.loads(Path(selections_path).read_text())
    pools = json.loads(Path(openings_path).read_text())
    if sha(legacy_supervised) != LEGACY_SHA or selections['seeds'] != SEEDS:
        raise ValueError('Frozen reference and three selections required')
    kinds = {NumpyEvaluator(row['file']).metadata['kind'] for row in selections['models']}
    if len(kinds) != 1 or not kinds.issubset({'supervised','evolutionary'}):
        raise ValueError('One new study identity is required for a sealed assessment')
    seal_path = Path(openings_path).parent / ('assessment-seal-'+next(iter(kinds))+'.json')
    seal = dict(selection_sha256=sha(selections_path),opening_manifest_sha256=sha(openings_path))
    if seal_path.exists() and json.loads(seal_path.read_text()) != seal:
        raise ValueError('These assessment pools are already sealed to another selection')
    atomic_json(seal_path,seal)
    if stage == 'final':
        qualification = json.loads((directory / 'qualification/evidence.json').read_text())
        if not qualify(qualification, 'qualification', sha(selections_path))['passed']:
            raise ValueError('Failed qualification blocks fresh final evaluation')
        if set(r['id'] for r in pools['final']) & set(r['id'] for r in pools['qualification']):
            raise ValueError('Final and qualification pools overlap')
    result = dict(stage=stage, seeds=SEEDS, opening_ids=[r['id'] for r in pools[stage]],
                  opening_records=pools[stage], opening_manifest_sha256=sha(openings_path), opponents={})
    for name, opponent in [('classical', 'handcrafted'), ('supervised', legacy_supervised)]:
        ledgers = []
        for row in selections['models']:
            if sha(row['file']) != row['sha256']:
                raise ValueError('Frozen checkpoint changed')
            ledger = cpu_matches(row['file'], opponent, pools[stage],
                directory / stage / f'{name}-seed-{row["seed"]}.json',
                config['cpu_match_time_ms'], config['simulations'], config['max_plies'], workers=workers)
            ledgers.append(dict(ledger, seed=row['seed'], stage=stage,
                                selection_sha256=sha(selections_path)))
        result['opponents'][name] = ledgers
    atomic_json(directory / stage / 'evidence.json', result)
    decision = qualify(result, stage, sha(selections_path), config['bootstrap_resamples'])
    atomic_json(directory / stage / 'gate.json', decision)
    return decision
