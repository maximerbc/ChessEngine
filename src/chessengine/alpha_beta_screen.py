"""Same-search development ablation; preserves PUCT evidence and sealed final pools."""
import argparse
import json
import platform
import time
from pathlib import Path

from .features import position_key, restore
from .model import NumpyEvaluator
from .gpu.protocol import atomic_json, freeze, sha, LEGACY_SHA
from .gpu.release import EXPERIMENTAL_SUPERVISED_SHA
from .gpu.evaluation import cpu_matches
from .strength_screen import summarize


def run(config_path, openings_path, prior_screen, diagnosis, output, candidate, legacy, workers=4):
    if workers < 1:
        raise ValueError('Positive worker count required')
    config = json.loads(Path(config_path).read_text())
    if (config['candidate_strategy'] != 'alpha-beta' or config['opponent_strategy'] != 'alpha-beta'
            or config['qdepth'] != 4 or config['node_ceiling'] != 2**31-1
            or config['time_ms'] != 1500 or config['max_plies'] != 512
            or config['opening_pool'] != 'development' or config['opening_offset'] != 0
            or config['seed'] != 42 or config['opening_pairs'] != 20 or config['total_games'] != 80
            or config['opponents'] != ['classical', 'supervised']
            or config['qualification'] or config['final_tests_opened']
            or config['automatic_model_promotion'] or config['automatic_larger_runs']):
        raise ValueError('Expected the explicitly authorized eighty-game same-search development protocol')
    if sha(openings_path) != config['opening_manifest_sha256']:
        raise ValueError('Use the unchanged opening manifest from the original screen')
    pools = json.loads(Path(openings_path).read_text())
    starts = pools['development'][:config['opening_pairs']]
    if (len(starts) != 20 or len({r['id'] for r in starts}) != 20
            or len({r['source_game_id'] for r in starts}) != 20
            or len({position_key(restore(r)) for r in starts}) != 20):
        raise ValueError('Twenty distinct source games and canonical openings required')
    if sha(candidate) != EXPERIMENTAL_SUPERVISED_SHA or sha(legacy) != LEGACY_SHA:
        raise ValueError('Use only the frozen seed-42 augmented and released 80k checkpoints')
    if NumpyEvaluator(candidate).metadata['seed'] != 42:
        raise ValueError('The representative seed cannot be selected from match results')
    prior = json.loads(Path(prior_screen).read_text())
    if (prior.get('status') != 'complete' or prior.get('seed') != 42
            or any(prior['opponents'][name]['games'] != 40 for name in config['opponents'])):
        raise ValueError('Preserve the complete preceding PUCT evidence before the ablation')
    output = Path(output)
    definition = dict(config=config, workers=workers, hardware=platform.platform())
    protocol = freeze(output, definition, [config_path, openings_path, prior_screen, diagnosis, candidate, legacy])
    status = dict(status='running', stage='same-search-development-ablation', protocol_id=protocol,
        definition=definition, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        completed_opponents=[], qualification=False, final_tests_opened=False, paid_cost_cad=0)
    atomic_json(output/'status.json', status)
    started = time.perf_counter()
    try:
        results = {}
        for name, opponent in [('classical', 'handcrafted'), ('supervised', str(legacy))]:
            status.update(opponent=name, completed_opponents=list(results))
            atomic_json(output/'status.json', status)
            ledger = cpu_matches(str(candidate), opponent, starts, output/f'{name}.json',
                time_ms=config['time_ms'], simulations=None, max_plies=config['max_plies'],
                workers=workers, candidate_strategy='alpha-beta', opponent_strategy='alpha-beta')
            summary = summarize(ledger)
            metrics = [row['search_metrics'] for row in ledger['games']]
            for side, count_key in [('candidate', 'candidate_moves'), ('reference', 'reference_moves')]:
                count = sum(row[count_key] for row in metrics)
                summary[side+'_search_work'] = dict(moves=count,
                    nodes=sum(row[side+'_nodes'] for row in metrics),
                    search_seconds=sum(row[side+'_search_seconds'] for row in metrics),
                    mean_search_ms=1000*sum(row[side+'_search_seconds'] for row in metrics)/max(count, 1))
            results[name] = summary
            atomic_json(output/'partial-summary.json', results)
        result = dict(status='complete', evidence_level='development-search-ablation', seed=42,
            protocol_id=protocol, opponents=results, session_elapsed_seconds=time.perf_counter()-started,
            reused_development_openings=True, qualification=False, final_tests_opened=False,
            website_changed=False, paid_cost_cad=0,
            limitations=['One fixed training seed and twenty already used opening pairs.',
                'This controls search; it is not independent final confirmation or qualification.',
                'Equal move-time allowances use a nonbinding node ceiling, not the serving node cap.',
                'No model selection, promotion, further training or larger evaluation follows automatically.'])
        atomic_json(output/'summary.json', result)
        atomic_json(output/'status.json', result)
        return result
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            error=str(error), session_elapsed_seconds=time.perf_counter()-started)
        atomic_json(output/'status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/supervised-alpha-beta-screen.json')
    parser.add_argument('--openings', required=True)
    parser.add_argument('--prior-screen', required=True)
    parser.add_argument('--diagnosis', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--candidate', default='artifacts/models/supervised-1m.npz')
    parser.add_argument('--legacy', default='artifacts/models/supervised.npz')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.openings, args.prior_screen, args.diagnosis, args.output,
                         args.candidate, args.legacy, args.workers), indent=2))


if __name__ == '__main__':
    main()
