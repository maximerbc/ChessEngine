"""Shared value architecture; no policy head or inherited supervised parameters."""
import json
from pathlib import Path
import numpy as np
from .protocol import ARCHITECTURE


def initialize(key):
    import jax
    import jax.numpy as jnp
    keys = jax.random.split(key, 3)
    return tuple({'w': jax.random.normal(k, (b, a),dtype=jnp.float32) * jnp.sqrt(2.0 / a),
                  'b': jnp.zeros(b,dtype=jnp.float32)} for k, a, b in zip(keys, ARCHITECTURE, ARCHITECTURE[1:]))


def apply(params, x):
    import jax
    import jax.numpy as jnp
    for i, layer in enumerate(params):
        x = jnp.matmul(x,layer['w'].T,precision=jax.lax.Precision.HIGHEST) + layer['b']
        x = jnp.maximum(x, 0) if i < 2 else jnp.tanh(x)
    return x.squeeze(-1)


def export(params, path, kind, evidence, context=None):
    if kind not in {'supervised', 'evolutionary'}:
        raise ValueError('Unknown value model identity')
    arrays = {}
    for i, (layer, a, b) in enumerate(zip(params, ARCHITECTURE, ARCHITECTURE[1:])):
        for name, shape in [('w', (b, a)), ('b', (b,))]:
            value = np.asarray(layer[name], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid architecture or non-finite network')
            arrays[f'{name}{i}'] = value
    arrays['metadata'] = np.array(json.dumps(dict(format_version=1, feature_version=3,
        hidden=[128, 64], architecture=ARCHITECTURE, kind=kind, evidence=evidence,
        study='gpu-value-v1', search='puct-uniform-v1', **(context or {}))))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def import_numpy(path):
    from ..model import NumpyEvaluator
    model = NumpyEvaluator(path)
    if model.feature_version != 3 or model.metadata.get('architecture') != ARCHITECTURE:
        raise ValueError('GPU study requires explicit v3 853/128/64/1 architecture')
    return tuple({'w': w, 'b': b} for w, b in model.layers)
