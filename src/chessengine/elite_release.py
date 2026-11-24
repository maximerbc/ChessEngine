"""Stage only the fixed seed-42 human policy after its validation/export gate."""
import argparse
from pathlib import Path
import shutil
from .common import digest, read_json
from .elite_data import atomic_json, preflight
from .elite_policy import NumpyElitePolicy, SCHEMA, STRATEGY, validate_release


def stage(root, manifest_path):
    root, manifest_path = Path(root), Path(manifest_path)
    preflight(root)
    gate = read_json(root / 'gate.json')
    selections = read_json(root / 'selections.json')
    if (gate.get('selections_sha256') != digest(root / 'selections.json')
            or selections.get('representative_seed') != 42
            or [r['seed'] for r in selections['models']] != [42, 43, 44]
            or len(read_json(root / 'results.json')['seeds']) != 3):
        raise ValueError('Complete sealed three-seed evidence is required')
    source = root / 'seed-42/best.npz'
    policy = NumpyElitePolicy(source)
    if gate.get('dataset_sha256') != digest(root / 'dataset.json'):
        raise ValueError('Gate belongs to a different dataset')
    row = dict(id='elite-policy', name='Move Prediction', kind='human-move-policy',
        file='elite-policy.npz', sha256=policy.checkpoint_id, architecture=SCHEMA,
        search=dict(strategy=STRATEGY), experimental=True, qualified=False,
        dataset_sha256=digest(root / 'dataset.json'),
        strength_status='not-established', release_status='experimental-playable',
        authorization_id='user-20261124-elite-play', evidence_id='elite-pytorch-baseline-20261124',
        documentation_path='/research?study=elite-pytorch-baseline-20261124',
        gate_file='elite-policy-gate.json', gate_sha256=digest(root / 'gate.json'))
    # Validate the candidate before changing the existing release.
    candidate = dict(row, gate_file='gate.json')
    validate_release(candidate, policy, root)
    manifest = read_json(manifest_path)
    if manifest.get('default') != 'supervised' or not {'supervised','supervised-1m'}.issubset(
            {r['id'] for r in manifest['models']}):
        raise ValueError('Preserve the current supervised lineup/default')
    selected = next(r for r in selections['models'] if r['seed'] == 42)
    if selected['checkpoint_sha256'] != policy.checkpoint_id:
        raise ValueError('Representative checkpoint differs from sealed selection')
    for existing in manifest['models']:
        if existing['id'] != 'elite-policy' and digest(manifest_path.parent / existing['file']) != existing['sha256']:
            raise ValueError('Existing released checkpoint changed')
    shutil.copy2(source, manifest_path.parent / row['file'])
    shutil.copy2(root / 'gate.json', manifest_path.parent / row['gate_file'])
    manifest['release_version'] = 5
    manifest['models'] = [r for r in manifest['models'] if r['id'] != 'elite-policy'] + [row]
    manifest['required_model_ids'] = [r['id'] for r in manifest['models']]
    atomic_json(manifest_path, manifest)
    public_manifest = manifest_path.resolve().parents[2] / 'web/public/models/manifest.json'
    if public_manifest.parent.is_dir():
        atomic_json(public_manifest, manifest)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--manifest', default='artifacts/models/manifest.json')
    a = p.parse_args()
    print(stage(a.root, a.manifest))


if __name__ == '__main__':
    main()
