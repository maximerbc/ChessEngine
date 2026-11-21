"""Honest, self-contained research status, independent of promotion."""
import html
import json
from pathlib import Path
from .protocol import atomic_json


def publish(config, evidence, directory):
    from ..pv_reporting import public
    blocked = {'games','records','positions','moves','initial_fen','fen','optimizer','rng','params',
               'parameters','population','tasks','pending','features','targets'}
    def redact(value):
        if isinstance(value,dict):
            return {key:redact(item) for key,item in value.items() if key not in blocked}
        if isinstance(value,list):
            return [redact(item) for item in value]
        return value
    allowed = {'status','conclusion','nvidia_verified','training_executed','paid_spend_cad','summary',
               'seeds','qualification','final','forecast','verification','limitations','hardware'}
    evidence = public(redact({key:value for key,value in evidence.items() if key in allowed}))
    directory = Path(directory)
    record = dict(study='gpu-value-v1', status=evidence.get('status', 'infrastructure-only'),
        supervised_positions=config['supervised']['counts'],
        planned_evolutionary_training_games=368640, pilot_training_games=30720,
        budget='unset pending measured forecast and user decision', evidence=evidence,
        limitations=['No NVIDIA throughput or strength claim without measured execution.',
            'One million total positions include 800,000 training positions.',
            'Teacher agreement and playing strength are separate assessments.',
            '184,320, 368,640 and 737,280 games are scenarios, not proven sufficiency thresholds.',
            'Historical negative and inconclusive results remain unchanged.'])
    atomic_json(directory / 'evidence.json', record)
    content = '<!doctype html><html lang="en"><meta charset="utf-8"><title>GPU learning studies</title>'
    content += '<style>body{max-width:900px;margin:40px auto;padding:20px;font:17px system-ui}pre{white-space:pre-wrap}</style>'
    content += '<h1>GPU learning studies</h1><p>Research infrastructure; new models are unavailable until qualification.</p>'
    content += '<pre>' + html.escape(json.dumps(record, indent=2)) + '</pre></html>'
    (directory / 'index.html').write_text(content)
    return record
