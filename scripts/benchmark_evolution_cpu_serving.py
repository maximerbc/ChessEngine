"""Four CPU serving games for timing, never checkpoint selection or qualification."""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from chessengine.engine import Handcrafted, Search
from chessengine.features import restore, record_board
from chessengine.model import NumpyEvaluator
from chessengine.rootcover import RootCoverageValueSearch
from chessengine.gpu.protocol import freeze, atomic_json, identity, sha


def play(task):
    candidate, supervised, opening, color, opponent, simulations = task
    board = restore(opening)
    actor = RootCoverageValueSearch(NumpyEvaluator(candidate), simulations, 1500)
    reference = Search(Handcrafted() if opponent == 'classical' else NumpyEvaluator(supervised),
                       2**31-1, 1500)
    seconds = dict(candidate=0., reference=0.)
    work = dict(candidate_simulations=0, reference_alpha_beta_nodes=0)
    start = time.perf_counter()
    plies = 0
    for _ in range(512):
        if board.is_game_over(claim_draw=True):
            break
        acting = board.turn == color
        result = (actor if acting else reference).run(board)
        seconds['candidate' if acting else 'reference'] += result.elapsed_ms/1000
        work['candidate_simulations' if acting else 'reference_alpha_beta_nodes'] += (
            result.simulations if acting else result.nodes)
        if result.move is None:
            raise ValueError('Nonterminal serving game returned no move')
        board.push_uci(result.move)
        plies += 1
    outcome = board.outcome(claim_draw=True)
    if 'jax' in sys.modules or 'torch' in sys.modules:
        raise RuntimeError('Training library imported into CPU serving')
    return dict(task_id=identity([str(candidate),opening['id'],color,opponent,simulations]),
        candidate_color='white' if color else 'black', opponent=opponent,
        opening=opening, **record_board(board), result=outcome.result() if outcome else '*',
        capped=outcome is None, plies=plies, search_seconds=seconds, work=work,
        elapsed_seconds=time.perf_counter()-start, training_libraries_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['candidate','supervised','openings','selection','output']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    selection = json.loads(Path(args.selection).read_text())
    if not selection['passed']:
        raise ValueError('A passed search-coverage selection is required')
    opening = json.loads(Path(args.openings).read_text())['benchmark'][0]
    protocol = freeze(output, dict(purpose='CPU serving throughput only', games=4,
        workers=4, time_ms=1500, simulations=selection['simulations'], candidate='fixed-random-initializer'),
        [args.candidate,args.supervised,args.openings,args.selection,__file__])
    tasks = [(args.candidate,args.supervised,opening,color,opponent,selection['simulations'])
             for opponent in ['classical','augmented-supervised'] for color in [True,False]]
    start = time.perf_counter()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in concurrent.futures.as_completed([pool.submit(play,t) for t in tasks]):
            row = future.result()
            atomic_json(output/(row['task_id']+'.json'),row)
            rows.append(row)
    elapsed = time.perf_counter()-start
    atomic_json(output/'measurement.json', dict(protocol_id=protocol, games=rows,
        elapsed_seconds=elapsed, workers=4, games_per_second=len(rows)/elapsed,
        candidate_sha256=sha(args.candidate), reference_sha256=sha(args.supervised),
        purpose='CPU serving cost/time calibration; no model selection or qualification',
        candidate_strategy='puct-rootcover-v1', reference_strategy='alpha-beta',
        time_ms_allowance=1500, reference_node_ceiling=2**31-1,
        limitation='Four already used development games with a fixed random initializer cannot establish playing strength or final-evaluation duration.'))


if __name__ == '__main__':
    main()
