"""Resumable, paired tournament matches shared by fitness and evaluation."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import multiprocessing
from pathlib import Path
import resource
import sys
import time
import io
import chess
import chess.pgn
from threadpoolctl import threadpool_limits
from .common import digest, json_hash, read_json, write_json
from .engine import Handcrafted, Search
from .evolution_model import load_evaluator
from .features import restore


def memory_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


@lru_cache(maxsize=64)
def evaluator(spec):
    return Handcrafted() if spec == 'handcrafted' else load_evaluator(spec)


def identity(spec):
    return 'handcrafted-v1' if spec == 'handcrafted' else digest(spec)


def play(task):
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        candidate, opponent = evaluator(task['candidate']), evaluator(task['opponent'])
        board = restore(task['opening'])
        color = task['color'] == 'white'
        searched = depth_one = depth_two = nodes = 0
        for _ in range(task['max_plies']):
            if board.is_game_over(claim_draw=True):
                break
            result = Search(candidate if board.turn == color else opponent, nodes=task['nodes'],
                            time_ms=task.get('time_ms'), qdepth=task['qdepth']).run(board)
            if result.move is None or chess.Move.from_uci(result.move) not in board.legal_moves:
                raise ValueError('Tournament engine returned an illegal move')
            searched += 1
            depth_one += result.depth >= 1
            depth_two += result.depth >= 2
            nodes += result.nodes
            board.push_uci(result.move)
        outcome = board.outcome(claim_draw=True)
        capped = outcome is None
        score = .5 if capped or outcome.winner is None else float(outcome.winner == color)
        game = chess.pgn.Game.from_board(board)
        game.headers['Event'] = 'Evolutionary tournament'
        game.headers['Termination'] = 'unfinished: neutral tournament adjudication' if capped else outcome.termination.name
        # A cap never changes the rules-based '*' result into a fabricated draw.
        game.headers['Result'] = board.result(claim_draw=True)
        return dict(task_id=task['task_id'], opening_id=task['opening']['id'], candidate_color=task['color'],
                    candidate=task['candidate'], opponent=task['opponent'], opponent_group=task.get('opponent_group'),
                    candidate_index=task.get('candidate_index'), score=score, capped=capped,
                    result=game.headers['Result'], pgn=str(game), searched_moves=searched,
                    depth_one_moves=depth_one, depth_two_moves=depth_two, nodes=nodes,
                    plies=len(board.move_stack)-len(task['opening']['moves']),
                    seconds=time.perf_counter()-started, worker_peak_bytes=memory_bytes())


def tasks_for(candidate, opponent, openings, config, prefix='', **extra):
    return [dict(task_id=f'{prefix}{row["id"]}:{color}', candidate=str(candidate), opponent=str(opponent),
                 opening=row, color=color, nodes=config['nodes'], qdepth=config['qdepth'],
                 max_plies=config['max_plies'], time_ms=config.get('time_ms'), **extra)
            for row in openings for color in ('white', 'black')]


def run_tasks(tasks, path, workers=2):
    """Hash all inputs and weights before accepting a saved match ledger."""
    path = Path(path)
    specs = {t[k] for t in tasks for k in ('candidate', 'opponent')}
    checkpoints = {s: identity(s) for s in sorted(specs)}
    signature = json_hash(dict(tasks=tasks, checkpoints=checkpoints))
    saved = read_json(path) if path.exists() else dict(signature=signature, tasks=tasks, checkpoints=checkpoints, games=[])
    if saved['signature'] != signature:
        raise ValueError('Frozen tournament tasks/checkpoints changed')
    done = {g['task_id'] for g in saved['games']}
    if len(done) != len(saved['games']):
        raise ValueError('Duplicate tournament match IDs')
    pending = [t for t in tasks if t['task_id'] not in done]
    started = time.perf_counter()
    prior = saved.get('active_seconds', 0.)
    def record(result):
        saved['games'].append(result)
        saved['active_seconds'] = prior + time.perf_counter()-started
        peaks = sorted((g['worker_peak_bytes'] for g in saved['games']), reverse=True)
        saved['peak_combined_bytes'] = memory_bytes() + sum(peaks[:workers])
        write_json(path, saved)
    if workers == 1:
        for task in pending:
            record(play(task))
    elif pending:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
        active = []
        remaining = iter(pending)
        try:
            for _ in range(workers):
                task = next(remaining, None)
                if task is not None:
                    active.append(pool.submit(play, task))
            while active:
                record(active.pop(0).result())
                task = next(remaining, None)
                if task is not None:
                    active.append(pool.submit(play, task))
        finally:
            # Never leave an entire queued generation running after interruption.
            for future in active:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
    return saved


def audit_ledger(ledger):
    """Independently replay PGNs and verify rules-based results and identities."""
    if json_hash(dict(tasks=ledger['tasks'], checkpoints=ledger['checkpoints'])) != ledger['signature']:
        raise ValueError('Tournament ledger signature mismatch')
    for spec, sha in ledger['checkpoints'].items():
        if identity(spec) != sha:
            raise ValueError('Audited tournament checkpoint changed')
    tasks = {t['task_id']: t for t in ledger['tasks']}
    if len(ledger['games']) != len(tasks) or len({g['task_id'] for g in ledger['games']}) != len(tasks):
        raise ValueError('Audit requires exactly one result per frozen task')
    for row in ledger['games']:
        task = tasks[row['task_id']]
        game = chess.pgn.read_game(io.StringIO(row['pgn']))
        if game is None or game.errors:
            raise ValueError('Invalid tournament PGN')
        board = game.board()
        moves = list(game.mainline_moves())
        prefix = task['opening']['moves']
        if board.fen() != task['opening']['initial_fen'] or [m.uci() for m in moves[:len(prefix)]] != prefix:
            raise ValueError('Audited opening history mismatch')
        for move in moves:
            if board.is_game_over(claim_draw=True) or move not in board.legal_moves:
                raise ValueError('Audited move after termination or illegal move')
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        capped = outcome is None
        color = task['color'] == 'white'
        score = .5 if capped or outcome.winner is None else float(outcome.winner == color)
        if (row['candidate'] != task['candidate'] or row['opponent'] != task['opponent']
                or row['candidate_color'] != task['color'] or row['opening_id'] != task['opening']['id']
                or row['capped'] != capped or row['score'] != score
                or row['result'] != board.result(claim_draw=True)
                or game.headers['Result'] != row['result']
                or (capped and len(moves)-len(prefix) != task['max_plies'])):
            raise ValueError('Audited tournament outcome/adjudication mismatch')
    return dict(matches=len(tasks), legal_histories=True, outcomes=True, checkpoints=True)
