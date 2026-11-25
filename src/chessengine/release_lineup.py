"""The three approved playable identities, separate from the historical v6 catalogue."""
from .demo_models import local_file
from .elite_policy import NumpyElitePolicy, validate_release
from .evolution_model import load_evaluator
from .gpu.release import validate_entry

VALUE_ID = 'supervised-1m'
POLICY_ID = 'elite-policy'
CHECKPOINTS = {
    VALUE_ID: '1d1b3fa0d436d19be062c4ae782af29f455455d3e7a4038279aa45225d2dbbc2',
    POLICY_ID: '5e6deee52407776949178f846b97ed1b02d5af964328beb632b7361b6d4b7687',
}


def load_three_approaches(manifest, directory):
    ids = [VALUE_ID, POLICY_ID]
    rows = manifest.get('models', [])
    if (manifest.get('release_version') != 7 or manifest.get('default') != VALUE_ID
            or manifest.get('required_model_ids') != ids or [row['id'] for row in rows] != ids):
        raise ValueError('Manifest v7 requires the fixed three-approach release identities')
    models = {}
    for row in rows:
        model_id = row['id']
        path = local_file(directory, row['file'])
        policy = model_id == POLICY_ID
        evaluator = NumpyElitePolicy(path) if policy else load_evaluator(path)
        if evaluator.checkpoint_id != row.get('sha256'):
            raise ValueError('Model manifest checksum mismatch')
        if evaluator.checkpoint_id != CHECKPOINTS[model_id]:
            raise ValueError('Approved checkpoint identity mismatch')
        if policy:
            local_file(directory, row.get('gate_file'))
            validate_release(row, evaluator, directory)
        else:
            validate_entry(row, evaluator)
            if (row.get('kind') != 'supervised' or row.get('search') != {'strategy': 'alpha-beta'}
                    or row.get('architecture') != [853, *evaluator.metadata['hidden'], 1]
                    or row.get('feature_version') != evaluator.feature_version):
                raise ValueError('Manifest architecture, feature, search or evidence identity mismatch')
        models[model_id] = dict(
            name=row['name'], kind='human-move-policy' if policy else 'supervised',
            evidence=row['evidence_id'], checkpoint_id=evaluator.checkpoint_id, evaluator=evaluator,
            strategy=row['search']['strategy'], search=row['search'], experimental=True, qualified=False,
            strength_status='not-established', documentation_path=row.get('documentation_path'),
            components={'policy' if policy else 'value': evaluator.checkpoint_id}, evidence_id=row['evidence_id'])
    return models
