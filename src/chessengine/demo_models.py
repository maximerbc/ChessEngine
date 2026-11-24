"""Explicit, inference-only catalogue for the authorized experimental demo."""
from pathlib import Path
import json
import re
import time

import numpy as np

from .common import digest
from .features import feature_size
from .model import NumpyEvaluator
from .elite_policy import NumpyElitePolicy, SCHEMA as HUMAN_SCHEMA, validate_release
from .move_policy import NumpyMovePolicy, SCHEMA as STOCKFISH_SCHEMA
from .gpu.protocol import LEGACY_SHA
from .gpu.release import validate_entry


VALUE_VERSIONS = {
    'supervised': 1, 'supervised-large': 1,
    'supervised-10k': 1, 'supervised-40k': 1, 'supervised-diagnostic': 1,
    'supervised-weaker-decay': 1, 'supervised-stronger-decay': 1,
    'supervised-dropout-01': 1, 'supervised-dropout-02': 1,
    'supervised-180k': 1, 'supervised-smaller-lr': 1, 'supervised-adaptive-lr': 1,
    'supervised-relative': 2, 'supervised-relative-material': 3,
    'supervised-combined': 3, 'supervised-averaged': 3, 'supervised-1m': 3,
}
POLICY_KINDS = {'stockfish-policy': 'research-direct-move-policy', 'elite-policy': 'human-move-policy'}
MODEL_IDS = {*VALUE_VERSIONS, *POLICY_KINDS}
AUTHORIZATION = 'user-20261124-all-supervised-demo'


def weight_digest(path):
    """Identity of trained arrays, independent of metadata/export ZIP timestamps."""
    import hashlib
    result = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in sorted(k for k in data.files if k != 'metadata'):
            result.update(key.encode())
            result.update(data[key].tobytes())
    return result.hexdigest()


def local_file(directory, name):
    if (not isinstance(name, str) or not name or Path(name).name != name
            or (Path(directory) / name).resolve().parent != Path(directory).resolve()):
        raise ValueError('Model file must be inside the manifest directory')
    path = Path(directory) / name
    if not path.is_file():
        raise ValueError(f'Missing declared model artifact: {name}')
    return path


class DirectPolicy:
    """Legal-move inference with no invented value head or search statistics."""
    def __init__(self, policy, label_source):
        self.policy = policy
        self.metadata = policy.metadata
        self.checkpoint_id = policy.checkpoint_id
        self.label_source = label_source

    def analyze(self, board):
        start = time.perf_counter()
        ranked = self.policy.rank(board)
        move = ranked[0]['move'] if ranked else None
        resulting = board.copy(stack=True)
        if move:
            resulting.push_uci(move)
        return dict(best_move=move, move=move, resulting_fen=resulting.fen(),
            terminal=resulting.result(claim_draw=True) if resulting.is_game_over(claim_draw=True) else None,
            value=None, position_value=None, mate_score=False, value_perspective=None,
            depth=None, nodes=None, simulations=None, search=None, pv=[],
            elapsed_ms=(time.perf_counter() - start) * 1000, strategy='direct-policy-v1',
            root_rankings=ranked, explanation=dict(raw_value=None, feature_version=None,
                side_to_move='white' if board.turn else 'black', piece_counts={},
                label_source=self.label_source))


def load_catalogue(manifest, directory):
    rows = manifest.get('models', [])
    ids = [row['id'] for row in rows]
    required = manifest.get('required_model_ids', [])
    if (manifest.get('default') != 'supervised' or set(ids) != MODEL_IDS
            or len(ids) != len(set(ids)) or set(required) != set(ids)
            or len(required) != len(set(required))):
        raise ValueError('Manifest v6 requires the complete explicit supervised catalogue and default')
    models, weights = {}, set()
    for row in rows:
        path = local_file(directory, row['file'])
        if digest(path) != row['sha256']:
            raise ValueError('Model manifest checksum mismatch')
        weight_id = weight_digest(path)
        if weight_id in weights or weight_id != row.get('weights_sha256'):
            raise ValueError('Duplicate or mismatched trained-weight identity')
        weights.add(weight_id)
        if (not re.fullmatch('[0-9a-f]{64}', row.get('source_sha256', ''))
                or not row.get('provenance') or not row.get('description') or not row.get('group')
                or not row.get('evidence_id') or not row.get('documentation_path', '').startswith('/research')
                or row.get('representative_seed') != 42):
            raise ValueError('Missing catalogue provenance or presentation metadata')
        if row['id'] != 'supervised' and (row.get('experimental') is not True
                or row.get('qualified') is not False or row.get('strength_status') != 'not-established'
                or row.get('release_status') != 'experimental-playable'):
            raise ValueError('Experimental catalogue entry must not claim qualification')
        policy_kind = POLICY_KINDS.get(row['id'])
        if policy_kind:
            if row['kind'] != policy_kind or row.get('search') != {'strategy': 'direct-policy-v1'}:
                raise ValueError('Policy kind or inference strategy mismatch')
            human = row['id'] == 'elite-policy'
            policy = NumpyElitePolicy(path) if human else NumpyMovePolicy(path)
            if (row.get('architecture') != (HUMAN_SCHEMA if human else STOCKFISH_SCHEMA)
                    or row.get('checkpoint_schema') != policy.metadata['format']):
                raise ValueError('Policy architecture mismatch')
            if human:
                local_file(directory, row.get('gate_file'))
                validate_release(row, policy, directory)
            else:
                gate_path = local_file(directory, row.get('gate_file'))
                if digest(gate_path) != row.get('gate_sha256'):
                    raise ValueError('Policy gate checksum mismatch')
                gate = json.loads(gate_path.read_text())
                if (gate.get('seed') != 42 or gate.get('checkpoint_sha256') != policy.checkpoint_id
                        or gate.get('cpu_numpy_export_verified') is not True
                        or gate.get('chosen_moves_match') is not True or gate.get('failures') != []
                        or row.get('authorization_id') != AUTHORIZATION):
                    raise ValueError('Stockfish policy export gate failed')
            evaluator = DirectPolicy(policy, 'elite human games' if human else 'Stockfish first-PV moves')
        else:
            evaluator = NumpyEvaluator(path)
            version = VALUE_VERSIONS[row['id']]
            with np.load(path, allow_pickle=False) as arrays:
                expected = {'metadata', *(f'{name}{i}' for i in range(len(evaluator.layers)) for name in ['w', 'b'])}
                if (set(arrays.files) != expected or row.get('checkpoint_schema') != 1
                        or any(arrays[key].dtype != np.float32 for key in expected - {'metadata'})):
                    raise ValueError('Unsupported value checkpoint arrays or schema')
            if (row.get('kind') != 'supervised' or evaluator.metadata.get('kind') != 'supervised'
                    or evaluator.feature_version != version or row.get('feature_version') != version
                    or row.get('architecture') != [feature_size(version), *evaluator.metadata['hidden'], 1]
                    or row.get('search') != {'strategy': 'alpha-beta'}):
                raise ValueError('Catalogue value schema or search mismatch')
            if row['id'] == 'supervised' and evaluator.checkpoint_id != LEGACY_SHA:
                raise ValueError('The released Supervised identity must be preserved')
            if row['id'] == 'supervised-1m':
                validate_entry(row, evaluator)
            elif row['id'] != 'supervised' and row.get('authorization_id') != AUTHORIZATION:
                raise ValueError('Catalogue entry needs experimental-play authorization')
        models[row['id']] = dict(
            **{k: row[k] for k in ['name', 'kind', 'description', 'group', 'evidence_id',
                                  'documentation_path', 'experimental', 'qualified', 'strength_status']},
            checkpoint_id=evaluator.checkpoint_id, evaluator=evaluator,
            evidence=evaluator.metadata.get('evidence', row['evidence_id']),
            strategy=row['search']['strategy'], search=row['search'],
            components={'policy' if policy_kind else 'value': evaluator.checkpoint_id})
    return models
