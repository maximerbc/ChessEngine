"""Reproduce the frozen demo catalogue from private, checksum-pinned exports."""
import argparse
import html
import json
from pathlib import Path
import shutil
import tempfile

import chess
import numpy as np

from chessengine.common import digest, read_json, write_json
from chessengine.demo_models import load_catalogue, weight_digest
from chessengine.model import NumpyEvaluator


def stage(config, source_root, elite_root, output=Path('.')):
    config = read_json(config)
    roots = {'original': Path(source_root), 'elite': Path(elite_root)}
    output = Path(output)
    destination = output / 'artifacts/models'
    # Build and validate the complete collection before replacing any active file.
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary)
        rows, verification = [], []
        for spec in config['models']:
            source = roots[spec['source_root']] / spec['source']
            if digest(source) != spec['input_sha256']:
                raise ValueError(f"Frozen source changed: {spec['id']}")
            row = {k: v for k, v in spec.items() if k not in {'source', 'source_root', 'gate_source', 'input_sha256', 'export_sha256', 'gate_input_sha256'}}
            target = staged / row['file']
            if spec['kind'] != 'supervised' or spec['id'] in {'supervised', 'supervised-1m'}:
                shutil.copyfile(source, target)
            else:
                with np.load(source, allow_pickle=False) as data:
                    meta = json.loads(str(data['metadata']))
                    clean = {k: meta[k] for k in ['format_version', 'feature_version', 'hidden', 'kind', 'evidence']}
                    clean.update(seed=42, source_sha256=spec['source_sha256'])
                    np.savez_compressed(target, metadata=np.array(json.dumps(clean, sort_keys=True)),
                                        **{k: data[k] for k in data.files if k != 'metadata'})
            row['sha256'] = digest(target)
            if row['sha256'] != spec['export_sha256']:
                raise ValueError('Staged export differs from frozen inference identity')
            row['weights_sha256'] = weight_digest(target)
            if row['weights_sha256'] != weight_digest(source):
                raise ValueError('Staging changed trained weights')
            if 'gate_source' in spec:
                gate = roots[spec['source_root']] / spec['gate_source']
                if digest(gate) != spec['gate_input_sha256']:
                    raise ValueError('Frozen export gate changed')
                if row['id'] == 'stockfish-policy':
                    data = next(r for r in read_json(gate)['seeds'] if r['seed'] == 42)
                    write_json(staged / row['gate_file'], data)
                else:
                    shutil.copyfile(gate, staged / row['gate_file'])
                row['gate_sha256'] = digest(staged / row['gate_file'])
            if row['kind'] == 'supervised':
                a, b = NumpyEvaluator(source), NumpyEvaluator(target)
                boards = [chess.Board(), chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')]
                for board in boards:
                    np.testing.assert_allclose(a(board), b(board), atol=1e-6, rtol=0)
            verification.append(dict(id=row['id'], source_sha256=spec['source_sha256'],
                checkpoint_sha256=row['sha256'], weights_sha256=row['weights_sha256'], unchanged_weights=True))
            rows.append(row)
        manifest = dict(release_version=6, default='supervised',
                        required_model_ids=[r['id'] for r in rows], models=rows)
        load_catalogue(manifest, staged)
        destination.mkdir(parents=True, exist_ok=True)
        for path in staged.iterdir():
            shutil.copyfile(path, destination / path.name)
        write_json(destination / 'manifest.json', manifest)
    write_json(output / 'web/public/models/manifest.json', manifest)
    report = output / 'web/public/research/supervised-variants'
    report.mkdir(parents=True, exist_ok=True)
    evidence = dict(version=1, authorization=config['authorization'], representative_seed=42,
        selection='Previously selected validation checkpoints; no new test-based selection',
        strength='Experimental availability is not strength qualification; historical failures remain valid.',
        models=[{k: v for k, v in r.items() if k not in {'file', 'gate_file', 'provenance'}} for r in rows],
        verification=verification)
    write_json(report / 'evidence.json', evidence)
    body = ''.join('<tr><td>'+html.escape(r['name'])+'</td><td>'+html.escape(r['group'])+
        '</td><td>'+html.escape(r['description'])+'</td><td class="hash">'+r['sha256']+'</td></tr>' for r in rows)
    (report / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Playable supervised variants</title><style>body{max-width:1100px;margin:2rem auto;padding:1rem;font:17px/1.6 system-ui;background:#faf8f4;color:#252922}table{border-collapse:collapse;width:100%}td,th{padding:.6rem;text-align:left;border-bottom:1px solid #ccc;vertical-align:top}.hash{font:11px/1.5 monospace;overflow-wrap:anywhere;max-width:190px}@media(max-width:600px){table,tbody,tr,td{display:block}thead{display:none}tr{margin-bottom:1rem}.hash{max-width:none}}</style><nav><a href="/research">Research</a> · <a href="/research">Research journal</a></nav><h1>Playable supervised variants</h1><p>One fixed seed-42 representative per trained configuration. Classical is an additional untrained baseline. Identical trained weights are listed once. Gameplay does not train models.</p><p>The existing Supervised default and checkpoint remain unchanged. All additional variants are experimental: availability is not qualification or evidence of stronger play. Historical failed release gates and negative results remain in their original reports.</p><p>Value networks use the same bounded alpha-beta search. Stockfish move imitation and human-game prediction choose a legal move directly, without search or a position-value head. Move probabilities are not win probabilities. The human CNN retains its original 13-plane encoder; unlike feature-v1/v2/v3 value encoders, it does not encode all historical state. Rules and draw claims still use the complete history.</p><p>The 180k model missed its promotion gate. The combined model's release comparison also failed. Weight averaging missed the 2% adoption threshold. Stockfish-policy seed 43 collapsed and failed export parity; seed 42 alone represents that configuration here. These outcomes have not been rewritten.</p><p><a href="/research?study=supervised-diagnostics">Diagnostics</a> · <a href="/research?study=supervised-200k">Expanded data</a> · <a href="/research?study=fixed-data-improvements">Representation and optimization</a> · <a href="/research?study=supervised-averaging">Averaging</a> · <a href="/research/stockfish-move-policy/index.html">Stockfish policy evidence</a> · <a href="/research/elite-policy/index.html">Human policy evidence</a> · <a href="evidence.json">Checkpoint evidence</a></p><table><thead><tr><th>Model</th><th>Group</th><th>Method and limits</th><th>Checkpoint SHA-256</th></tr></thead><tbody>'''+body+'</tbody></table></html>')
    archive = output / 'docs/experiments/supervised-variants'
    archive.mkdir(parents=True, exist_ok=True)
    for name in ['index.html', 'evidence.json']:
        shutil.copyfile(report / name, archive / name)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/demo-models.json')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--elite-root', required=True)
    parser.add_argument('--output', default='.')
    args = parser.parse_args()
    manifest = stage(args.config, args.source_root, args.elite_root, args.output)
    print(f"Staged {len(manifest['models'])} supervised variants; default preserved")
