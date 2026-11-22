"""Frozen ten-generation feasibility decision; failure closes this study."""
import json
from pathlib import Path
import numpy as np
from .evaluation import pair_scores
from .protocol import atomic_json, identity


def decide(config, directory, initial_matches, last_matches, coverage, output):
    directory = Path(directory)
    history = json.loads((directory / 'history.json').read_text())
    if len(history) != config['pilot_generations']:
        raise ValueError('Complete the preregistered pilot before deciding')
    ledgers = [json.loads((directory / f'generation-{i:03}/games.json').read_text())
               for i in range(1, config['pilot_generations'] + 1)]
    rows = [r for ledger in ledgers for r in ledger['games']]
    expected = config['population'] * config['pilot_generations'] * config['games_per_candidate']
    if len(rows) != expected or expected != 30720:
        raise ValueError('Pilot must contain exactly 30,720 training games under its separate frozen seed')
    before = pair_scores(initial_matches['games'])
    after = pair_scores(last_matches['games'])
    if len(before) != 40 or set(before) != set(after):
        raise ValueError('Pilot improvement uses the same 40 opening pairs')
    gain = float(np.mean([after[k] - before[k] for k in before]))
    cap_fraction = float(np.mean([r['capped'] for r in rows]))
    decisive_fraction = float(np.mean([r['result'] in ['1-0', '0-1'] for r in rows]))
    move_coverage = sum(r['coverage_hits'] for r in rows) / max(1, sum(r['plies'] for r in rows))
    measured = next((r for r in coverage.get('results', []) if r['simulations'] == config['simulations']), {})
    checks = dict(caps=cap_fraction <= .2, decisiveness=decisive_fraction >= .1,
        improvement=gain >= .05, coverage=measured.get('coverage_qualified') is True and move_coverage >= .9,
        parity=measured.get('rules_parity_passed') is True,
        resume=measured.get('resume_verified') is True)
    result = dict(passed=all(checks.values()), checks=checks, cap_fraction=cap_fraction,
        decisive_fraction=decisive_fraction, conservative_improvement=gain,
        config_sha256=identity(config), status='passed' if all(checks.values()) else 'closed-failed-pilot',
        training_games=expected, representative_seed=42, pilot_seed=config['pilot_seed'],full_pilot_move_coverage=move_coverage)
    path = Path(output)
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Pilot decision is frozen; no automatic retries or threshold changes')
    atomic_json(path, result)
    return result
