"""Validate manifests and stage qualified models; allow one explicit playable preview."""
import json
from pathlib import Path
import shutil
from ..model import NumpyEvaluator
from .protocol import LEGACY_SHA, ARCHITECTURE, MODEL_IDS, atomic_json, sha
from .evaluation import qualify
from ..features import position_key, restore

# Explicit user-authorized playable preview; this does not confer qualification.
EXPERIMENTAL_SUPERVISED_SHA = '1d1b3fa0d436d19be062c4ae782af29f455455d3e7a4038279aa45225d2dbbc2'
EXPERIMENTAL_AUTHORIZATION = 'user-20261122-augmented-play'


def validate_entry(row, evaluator):
    if row['id'] not in MODEL_IDS:
        return
    metadata = evaluator.metadata
    if (metadata.get('architecture') != ARCHITECTURE or metadata.get('feature_version') != 3
            or metadata.get('study') != 'gpu-value-v1' or metadata.get('seed') != 42
            or not metadata.get('protocol_id')
            or (row['id'] == 'supervised-1m' and metadata.get('training_positions') != 800000)
            or (row['id'] == 'evolutionary-gpu' and metadata.get('generation',0) <= 0)
            or metadata.get('kind') != MODEL_IDS[row['id']]
            or row.get('architecture') != ARCHITECTURE or row.get('feature_version') != 3
            or row.get('search') != {'strategy': 'puct-uniform-v1', 'c_puct': 1.5,
                                      'priors': 'uniform-legal', 'simulations': row.get('search', {}).get('simulations')}
            or row['search']['simulations'] not in [32, 64, 128]
            or not row.get('evidence_id')):
        raise ValueError('New GPU model requires explicit architecture, search and evidence identities')
    experimental_preview = (
        row['id'] == 'supervised-1m'
        and evaluator.checkpoint_id == EXPERIMENTAL_SUPERVISED_SHA
        and row.get('release_status') == 'experimental-playable'
        and row.get('authorization_id') == EXPERIMENTAL_AUTHORIZATION
        and row.get('experimental') is True and row.get('qualified') is False
        and row.get('strength_status') == 'not-established'
        and row.get('qualification_sha256') is None and row.get('final_sha256') is None
    )
    if not experimental_preview and (row.get('qualified') is not True
            or not row.get('qualification_sha256') or not row.get('final_sha256')):
        raise ValueError('New GPU model requires qualification or the explicitly authorized supervised preview')


def stage(legacy_manifest, candidates, destination):
    legacy_manifest, destination = Path(legacy_manifest), Path(destination)
    original = json.loads(legacy_manifest.read_text())
    if original.get('default') != 'supervised':
        raise ValueError('Current Supervised must remain the default')
    legacy_row = next(r for r in original['models'] if r['id'] == 'supervised')
    if sha(legacy_manifest.parent / legacy_row['file']) != LEGACY_SHA:
        raise ValueError('Released 80k identity changed')
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in original['models']:
        model = NumpyEvaluator(legacy_manifest.parent / row['file'])
        enriched = dict(row, architecture=[853 if model.feature_version == 3 else 841, *model.metadata['hidden'], 1],
                        feature_version=model.feature_version, search={'strategy': 'alpha-beta'},
                        evidence_id='original-study')
        shutil.copy2(legacy_manifest.parent / row['file'], destination / row['file'])
        rows.append(enriched)
    for candidate in candidates:
        model_id = candidate['id']
        if model_id not in MODEL_IDS or any(r['id'] == model_id for r in rows):
            raise ValueError('Unknown or duplicate model identity')
        selection_path = candidate['selections']
        selection = json.loads(Path(selection_path).read_text())
        if selection.get('seeds') != [42,43,44] or selection.get('selection_basis') != 'development-only':
            raise ValueError('All three development selections must be sealed')
        for model in selection['models']:
            if sha(model['file']) != model['sha256']:
                raise ValueError('A frozen seed checkpoint changed')
        selected = next(r for r in selection['models'] if r['seed'] == 42)
        if sha(selected['file']) != selected['sha256']:
            raise ValueError('Selected artifact changed')
        qual_raw = json.loads(Path(candidate['qualification']).read_text())
        final_raw = json.loads(Path(candidate['final']).read_text())
        if set(qual_raw['opening_ids']) & set(final_raw['opening_ids']):
            raise ValueError('Qualification and final opening pools overlap')
        q_games = {r['source_game_id'] for r in qual_raw['opening_records']}
        f_games = {r['source_game_id'] for r in final_raw['opening_records']}
        q_positions = {position_key(restore(r)) for r in qual_raw['opening_records']}
        f_positions = {position_key(restore(r)) for r in final_raw['opening_records']}
        if q_games & f_games or q_positions & f_positions:
            raise ValueError('Qualification and final source games or canonical positions overlap')
        for raw in [qual_raw, final_raw]:
            for ledgers in raw['opponents'].values():
                for seed, ledger in zip([42, 43, 44], ledgers):
                    checkpoint = next(r['sha256'] for r in selection['models'] if r['seed'] == seed)
                    if any(r['checkpoint_sha256'] != checkpoint for r in ledger['games']):
                        raise ValueError('Evidence does not match the frozen checkpoint')
        gates = [qualify(raw, name, sha(selection_path)) for raw, name in
                 [(qual_raw, 'qualification'), (final_raw, 'final')]]
        if not all(g['passed'] for g in gates):
            raise ValueError('Qualification/final gate failed; preserve the old release')
        target = destination / f'{model_id}.npz'
        shutil.copy2(selected['file'], target)
        row = dict(id=model_id, file=target.name, name=candidate['name'], kind=MODEL_IDS[model_id],
            sha256=sha(target), architecture=ARCHITECTURE, feature_version=3,
            search=dict(strategy='puct-uniform-v1', c_puct=1.5, priors='uniform-legal', simulations=candidate['simulations']),
            qualified=True, evidence_id=candidate['evidence_id'],
            qualification_sha256=sha(candidate['qualification']), final_sha256=sha(candidate['final']))
        validate_entry(row, NumpyEvaluator(target))
        rows.append(row)
    manifest = dict(release_version=4, default='supervised', required_model_ids=[r['id'] for r in rows], models=rows)
    atomic_json(destination / 'manifest.json', manifest)
    return manifest
