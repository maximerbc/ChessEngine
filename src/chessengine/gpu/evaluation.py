"""CPU matches, whole-pair uncertainty, frozen selection and fail-closed promotion."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import numpy as np
import chess
from threadpoolctl import threadpool_limits
from ..features import restore, record_board, position_key
from ..engine import Handcrafted
from ..model import NumpyEvaluator
from ..puct_serving import search_for, STRATEGY
from .protocol import atomic_json, identity, sha, SEEDS, LEGACY_SHA


def _pair_job(arguments):
    return cpu_matches(*arguments)


@threadpool_limits.wrap(limits=1, user_api='blas')
def cpu_matches(candidate, opponent, openings, output, time_ms=1500, simulations=64, max_plies=512,
                opponent_strategy='alpha-beta', workers=1):
    if workers < 1:
        raise ValueError('Positive CPU evaluation concurrency required')
    config = dict(candidate_sha256=sha(candidate), opponent_sha256='handcrafted-v1' if opponent == 'handcrafted'
                  else sha(opponent), openings=openings, time_ms=time_ms, simulations=simulations,
                  max_plies=max_plies, strategy=STRATEGY, opponent_strategy=opponent_strategy, search_metrics_version=1)
    protocol = identity(config)
    path = Path(output)
    ledger = json.loads(path.read_text()) if path.exists() else dict(protocol_id=protocol, config=config, games=[],active_seconds=0.)
    if ledger['protocol_id'] != protocol:
        raise ValueError('CPU evaluation protocol changed')
    if workers > 1:
        complete = {(g['opening_id'],g['color']) for g in ledger['games']}
        remaining = [row for row in openings if any((row['id'],color) not in complete for color in ['white','black'])]
        directory = path.parent / (path.stem + '-pairs')
        order = {row['id']:i for i,row in enumerate(openings)}
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(_pair_job,(candidate,opponent,[row],directory/f'{row["id"]}.json',
                       time_ms,simulations,max_plies,opponent_strategy,1)) for row in remaining]
            for future in as_completed(futures):
                result = future.result()
                returned_ids = {row['opening_id'] for row in result['games']}
                ledger['games'] = [r for r in ledger['games'] if r['opening_id'] not in returned_ids] + result['games']
                ledger['games'].sort(key=lambda r:(order[r['opening_id']],r['color'] != 'white'))
                ledger['execution_workers'] = workers
                ledger['active_seconds'] = sum(row.get('active_seconds',0.) for row in ledger['games'])
                atomic_json(path,ledger)
        return ledger
    candidate_model = NumpyEvaluator(candidate)
    reference = Handcrafted() if opponent == 'handcrafted' else NumpyEvaluator(opponent)
    done = {(g['opening_id'], g['color']) for g in ledger['games']}
    for opening in openings:
        for color in ['white', 'black']:
            if (opening['id'], color) in done:
                continue
            board = restore(opening)
            started = time.perf_counter()
            plies = 0
            metrics = dict(candidate_moves=0, candidate_coverage_hits=0, candidate_simulations=0,
                           candidate_search_seconds=0., reference_search_seconds=0.)
            progress_path = path.with_name(path.stem + '-progress.json')
            while not board.is_game_over(claim_draw=True) and plies < max_plies:
                candidate_turn = board.turn == (color == 'white')
                evaluator = candidate_model if candidate_turn else reference
                strategy = STRATEGY if candidate_turn else opponent_strategy
                result = search_for(evaluator, strategy, 2**31 - 1, time_ms, simulations).run(board)
                if result.move is None:
                    raise ValueError('Nonterminal search returned no move')
                if candidate_turn:
                    metrics['candidate_moves'] += 1
                    metrics['candidate_coverage_hits'] += int(result.root_coverage >= .8)
                    metrics['candidate_simulations'] += result.simulations
                    metrics['candidate_search_seconds'] += result.elapsed_ms / 1000
                else:
                    metrics['reference_search_seconds'] += result.elapsed_ms / 1000
                board.push_uci(result.move)
                plies += 1
                if plies == 1 or plies % 10 == 0:
                    atomic_json(progress_path, dict(status='playing', opening_id=opening['id'], color=color,
                        plies=plies, elapsed_seconds=time.perf_counter()-started, metrics=metrics))
            outcome = board.outcome(claim_draw=True)
            capped = outcome is None
            score = .5 if capped or outcome.winner is None else float(outcome.winner == (color == 'white'))
            ledger['games'].append(dict(opening_id=opening['id'], source_game_id=opening['source_game_id'],
                color=color, score=score, capped=capped, result='*' if capped else outcome.result(),
                played_plies=plies, search_metrics=metrics,
                checkpoint_sha256=sha(candidate), active_seconds=time.perf_counter()-started, **record_board(board)))
            ledger['active_seconds'] = sum(row['active_seconds'] for row in ledger['games'])
            atomic_json(path, ledger)
            atomic_json(progress_path, dict(status='game-complete', opening_id=opening['id'], color=color,
                plies=plies, elapsed_seconds=time.perf_counter()-started, metrics=metrics))
    return ledger


def pair_scores(games, conservative=True):
    pairs = {}
    seen = set()
    for row in games:
        key = row['opening_id'], row['color']
        if key in seen or row['color'] not in {'white', 'black'}:
            raise ValueError('Duplicate or invalid color in paired evidence')
        seen.add(key)
        board = chess.Board(row['initial_fen'])
        for move in row.get('moves',[]):
            if board.is_game_over(claim_draw=True):
                raise ValueError('Recorded moves continue after automatic rules termination')
            board.push_uci(move)
        if board.fen() != row['fen']:
            raise ValueError('Recorded game FEN disagrees with legal history')
        outcome = board.outcome(claim_draw=True)
        if row['capped']:
            if outcome is not None or row['result'] != '*' or row['score'] != .5:
                raise ValueError('Fabricated capped-game outcome')
        else:
            score = .5 if outcome and outcome.winner is None else float(outcome.winner == (row['color'] == 'white')) if outcome else None
            if outcome is None or outcome.result() != row['result'] or score != row['score']:
                raise ValueError('Stored result disagrees with replayed rules')
        score = 0 if conservative and row['capped'] else row['score']
        pairs.setdefault(row['opening_id'], []).append(score)
    if any(len(v) != 2 for v in pairs.values()):
        raise ValueError('Incomplete color-swapped pair')
    return {k: float(np.mean(v)) for k, v in pairs.items()}


def paired_summary(ledgers, resamples=10000):
    values = [pair_scores(ledger['games']) for ledger in ledgers]
    if not values or any(set(v) != set(values[0]) for v in values):
        raise ValueError('Seeds must share a complete paired opening pool')
    matrix = np.array([[v[k] for k in sorted(v)] for v in values])
    rng = np.random.default_rng(42)
    samples = np.empty(resamples)
    for i in range(resamples):
        # Same opening indices across all seeds preserve cross-seed correlations.
        samples[i] = matrix[:, rng.integers(0, matrix.shape[1], matrix.shape[1])].mean()
    return dict(score=float(matrix.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
        seed_sd=float(matrix.mean(1).std(ddof=1)) if len(values) > 1 else 0., pairs=matrix.shape[1],
        cap_fraction=sum(r['capped'] for ledger in ledgers for r in ledger['games'])
            / sum(len(ledger['games']) for ledger in ledgers))


def qualify(evidence, stage, selection_sha256, resamples=10000):
    if stage not in {'qualification', 'final'} or evidence.get('seeds') != SEEDS:
        raise ValueError('Three frozen seed selections required')
    expected_pool = evidence.get('opening_ids')
    if not expected_pool or len(expected_pool) != 200 or len(set(expected_pool)) != 200:
        raise ValueError('Exactly 200 separate opening pairs required')
    openings = evidence.get('opening_records',[])
    by_id = {row['id']:row for row in openings}
    if (len(openings) != 200 or set(by_id) != set(expected_pool)
            or len({row['source_game_id'] for row in openings}) != 200
            or len({position_key(restore(row)) for row in openings}) != 200):
        raise ValueError('Qualification requires 200 unique source-game and canonical opening records')
    output, passed = {}, True
    for name, threshold in [('classical', .5), ('supervised', .45)]:
        ledgers = evidence['opponents'][name]
        if len(ledgers) != 3:
            raise ValueError('All three seeds must be evaluated')
        for seed, ledger in zip(SEEDS, ledgers):
            if ledger.get('seed') != seed or ledger.get('stage') != stage:
                raise ValueError('Seed or evaluation stage mismatch')
            if set(pair_scores(ledger['games'])) != set(expected_pool):
                raise ValueError('Unexpected evaluation opening pool')
            if ledger.get('selection_sha256') != selection_sha256:
                raise ValueError('Selected checkpoint set changed')
            settings = ledger.get('config',{})
            reference = 'handcrafted-v1' if name == 'classical' else LEGACY_SHA
            if (settings.get('opponent_sha256') != reference or settings.get('strategy') != STRATEGY
                    or settings.get('opponent_strategy') != 'alpha-beta' or settings.get('time_ms') != 1500
                    or settings.get('max_plies') != 512):
                raise ValueError('Reference identity or equal CPU allowance does not match the frozen qualification protocol')
            for game in ledger['games']:
                opening = by_id[game['opening_id']]
                prefix = opening.get('moves',[])
                if (game['initial_fen'] != opening['initial_fen'] or game.get('moves',[])[:len(prefix)] != prefix
                        or game.get('source_game_id') != opening['source_game_id']
                        or game.get('checkpoint_sha256') != settings.get('candidate_sha256')):
                    raise ValueError('Match history or checkpoint does not match its frozen opening')
        representative = paired_summary([ledgers[0]], resamples)
        aggregate = paired_summary(ledgers, resamples)
        per_seed = [paired_summary([row], resamples) for row in ledgers]
        for result in [representative, aggregate]:
            lower = result['ci95'][0]
            passed &= result['cap_fraction'] <= .05 and (lower > threshold if name == 'classical' else lower >= threshold)
        output[name] = dict(representative=representative, aggregate=aggregate, per_seed=per_seed)
    return dict(passed=bool(passed), stage=stage, opponents=output,
                cap_sensitivity='all capped games counted as candidate losses',
                selection_sha256=selection_sha256, evidence_sha256=identity(evidence))


def seal_selections(models, output):
    if [r['seed'] for r in models] != SEEDS:
        raise ValueError('Freeze all three selections together, representative seed is 42')
    rows = [dict(r, sha256=sha(r['file'])) for r in models]
    result = dict(seeds=SEEDS, representative=42, models=rows, selection_basis='development-only')
    path = Path(output)
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Selections already sealed; final evidence cannot select another candidate')
    atomic_json(path, result)
    return result
