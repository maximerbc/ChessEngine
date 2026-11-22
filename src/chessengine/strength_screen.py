"""Small, explicitly exploratory CPU screen; never runs qualification or final pools."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
from .features import position_key, restore
from .model import NumpyEvaluator
from .gpu.protocol import atomic_json, freeze, sha, LEGACY_SHA
from .gpu.evaluation import cpu_matches, paired_summary


def summarize(ledger):
    conservative = paired_summary([ledger])
    games = ledger['games']
    # A second bootstrap retains the tournament's neutral fitness for capped games.
    scores = {}
    for row in games:
        scores.setdefault(row['opening_id'], []).append(row['score'])
    values = np.array([np.mean(scores[k]) for k in sorted(scores)])
    rng = np.random.default_rng(42)
    neutral = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    moves = sum(row['search_metrics']['candidate_moves'] for row in games)
    hits = sum(row['search_metrics']['candidate_coverage_hits'] for row in games)
    return dict(conservative=conservative, neutral_cap_score=float(values.mean()),
        neutral_cap_ci95=np.quantile(neutral, [.025, .975]).tolist(),
        wins=sum(r['score'] == 1 for r in games),
        rules_draws=sum(not r['capped'] and r['score'] == .5 for r in games),
        losses=sum(r['score'] == 0 for r in games), caps=sum(r['capped'] for r in games),
        coverage_fraction=hits/max(moves, 1) if ledger['config']['strategy'] == 'puct-uniform-v1' else None,
        games=len(games))


def run(openings_path, output, candidate, legacy, workers=4):
    if workers < 1:
        raise ValueError('Positive worker count required')
    output = Path(output)
    pools = json.loads(Path(openings_path).read_text())
    # Input order is frozen by the preparation manifest; no outcome-based selection.
    starts = pools['development'][:20]
    if (len(starts) != 20 or len({r['id'] for r in starts}) != 20
            or len({r['source_game_id'] for r in starts}) != 20
            or len({position_key(restore(r)) for r in starts}) != 20):
        raise ValueError('Need twenty unique fresh development opening pairs')
    from .gpu.release import EXPERIMENTAL_SUPERVISED_SHA
    if sha(candidate) != EXPERIMENTAL_SUPERVISED_SHA or sha(legacy) != LEGACY_SHA:
        raise ValueError('The screen is frozen to the selected seed-42 and released 80k identities')
    if NumpyEvaluator(candidate).metadata['seed'] != 42:
        raise ValueError('Seed 42 is the representative, not the best-performing seed')
    definition = dict(stage='development-screen', seed=42, opening_pairs=20, games=80,
        simulations=128, time_ms=1500, max_plies=512, workers=workers,
        candidate_strategy='puct-uniform-v1', reference_strategy='alpha-beta',
        hardware=platform.platform(), qualification=False, final_tests_opened=False)
    protocol = freeze(output, definition, [openings_path, candidate, legacy])
    started = time.perf_counter()
    status_path = output/'status.json'
    atomic_json(status_path, dict(status='running', stage='development-screen', protocol_id=protocol,
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), definition=definition,
        note='Completed games resume; an interrupted unfinished game restarts. No qualification/final run follows.'))
    try:
        results = {}
        for name, opponent in [('classical', 'handcrafted'), ('supervised', legacy)]:
            atomic_json(status_path, dict(status='running', stage='development-screen', opponent=name,
                protocol_id=protocol, completed_opponents=list(results), definition=definition))
            ledger = cpu_matches(str(candidate), str(opponent), starts, output/f'{name}.json',
                                 1500, 128, 512, workers=workers)
            results[name] = summarize(ledger)
            atomic_json(output/'partial-summary.json', results)
        result = dict(status='complete', evidence_level='exploratory development screen', seed=42,
            protocol_id=protocol, opponents=results, elapsed_seconds=time.perf_counter()-started,
            qualification=False, final_tests_opened=False,
            limitations=['Twenty opening pairs, one fixed training seed; close results may be inconclusive.',
                '128 simulations is a provisional CPU screen ceiling, not a resolved evolutionary workload.',
                'No strength qualification, checkpoint selection, Elo or superiority claim is implied.'])
        atomic_json(output/'summary.json', result)
        atomic_json(status_path, result)
        return result
    except BaseException as error:
        atomic_json(status_path, dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            error=str(error), protocol_id=protocol, elapsed_seconds=time.perf_counter()-started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openings', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--candidate', default='artifacts/models/supervised-1m.npz')
    parser.add_argument('--legacy', default='artifacts/models/supervised.npz')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.openings, args.output, args.candidate, args.legacy, args.workers), indent=2))


if __name__ == '__main__':
    main()
