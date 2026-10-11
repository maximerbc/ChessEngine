"""Extend a frozen corpus with new training games and resumable teacher labels."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import shutil
import time

import chess
import chess.engine
import chess.pgn

from .common import digest, json_hash, provenance, read_json, write_json
from .data import open_pgn, records, split_game
from .features import position_key, record_board, restore


def pin(path, identity):
    if path.exists():
        if read_json(path)['identity'] != identity:
            raise ValueError('Expansion configuration/input mismatch')
    else:
        write_json(path, {'identity': identity, 'provenance': provenance()})


def prepare_extension(config):
    """Keep every accepted sample from the final game (target may overshoot by five)."""
    started = time.perf_counter()
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    identity = {'config': config, 'base_sha256': digest(config['base']),
                'source_sha256': digest(config['source']), 'reserved_sha256': digest(config['reserved']),
                'code_sha256': digest(__file__)}
    pin(out / 'protocol.json', identity)
    destination = out / 'positions.jsonl'
    if (out / 'preparation.json').exists():
        result = read_json(out / 'preparation.json')
        if digest(destination) != result['sha256']:
            raise ValueError('Prepared extension checksum mismatch')
        return result
    base = list(records(config['base']))
    excluded_games = {r['game_id'] for r in base}
    # The old dataset is already validated. FEN keys avoid replaying its histories.
    seen = {position_key(chess.Board(r['fen'])) for r in base}
    for group in read_json(config['reserved']).values():
        for row in group:
            seen.add(position_key(restore(row)))
            if row.get('source_game_id'):
                excluded_games.add(row['source_game_id'])
    needed = config['target_total'] - len(base)
    if needed <= 0:
        raise ValueError('Target must exceed the base corpus')
    added, games, scanned = 0, 0, 0
    temporary = destination.with_suffix('.tmp')
    with open_pgn(config['source']) as stream, temporary.open('w') as output:
        while added < needed:
            game = chess.pgn.read_game(stream)
            if game is None:
                raise ValueError('Source exhausted before the target; use a separately planned source')
            scanned += 1
            if game.errors or game.headers.get('Variant', 'Standard') != 'Standard':
                continue
            board = game.board()
            if board.chess960 or not board.is_valid():
                continue
            moves = list(game.mainline_moves())
            game_id = json_hash({'initial': board.fen(), 'moves': [m.uci() for m in moves]})
            if game_id in excluded_games or split_game(game_id) != 'train':
                continue
            excluded_games.add(game_id)
            rng = random.Random(json_hash([config['sampling_seed'], game_id]))
            selected = set()
            for lo, hi in [(8, 30), (30, 70), (70, 1000)]:
                phase = list(range(lo, min(hi, len(moves))))
                selected.update(rng.sample(phase, min(2, len(phase))))
            before = added
            for i, move in enumerate(moves):
                board.push(move)
                if i not in selected or board.is_game_over(claim_draw=True):
                    continue
                key = position_key(board)
                if key in seen:
                    continue
                seen.add(key)
                row = {**record_board(board), 'game_id': game_id, 'source': config['source'],
                       'split': 'train',
                       'phase': 'opening' if i < 30 else 'middlegame' if i < 70 else 'endgame'}
                output.write(json.dumps(row) + '\n')
                added += 1
            games += added > before
            if games and games % 2000 == 0 and added > before:
                print(f'Prepared {added:,} additional positions from {games:,} new games', flush=True)
    temporary.replace(destination)
    result = {'positions': added, 'games': games, 'source_games_scanned': scanned,
              'base_positions': len(base), 'total_positions': len(base) + added,
              'sha256': digest(destination), 'wall_seconds': time.perf_counter() - started,
              'provenance': provenance()}
    write_json(out / 'preparation.json', result)
    return result


def label_chunk(rows, destination, binary, teacher):
    identity = {'rows_sha256': json_hash(rows), 'binary_sha256': teacher['binary_sha256'],
                'nodes': teacher['nodes'], 'threads': teacher['threads'], 'hash_mb': teacher['hash_mb']}
    metadata = destination.with_suffix('.json')
    if metadata.exists():
        result = read_json(metadata)
        if result['identity'] != identity or digest(destination) != result['sha256']:
            raise ValueError('Label chunk configuration/checksum mismatch')
        return result
    started = time.perf_counter()
    with chess.engine.SimpleEngine.popen_uci(binary) as engine:
        engine.configure({'Threads': teacher['threads'], 'Hash': teacher['hash_mb'], 'UCI_ShowWDL': True})
        if engine.id != teacher['teacher']:
            raise ValueError('Teacher identity mismatch')
        temporary = destination.with_suffix('.tmp')
        with temporary.open('w') as output:
            for row in rows:
                board = restore(row)
                engine.configure({'Clear Hash': None})
                info = engine.analyse(board, chess.engine.Limit(nodes=teacher['nodes']), game=object())
                if 'wdl' not in info:
                    raise ValueError('Teacher did not return WDL')
                wdl = info['wdl'].pov(board.turn)
                output.write(json.dumps({**row, 'target': (wdl.wins - wdl.losses) / wdl.total()}) + '\n')
    temporary.replace(destination)
    result = {'identity': identity, 'positions': len(rows), 'sha256': digest(destination),
              'label_seconds': time.perf_counter() - started}
    write_json(metadata, result)
    return result


def expand(config):
    prepare_extension(config)
    out = Path(config['output'])
    teacher = read_json(config['teacher_manifest'])
    if digest(config['binary']) != teacher['binary_sha256'] or digest(config['base']) != teacher['sha256']:
        raise ValueError('Original teacher binary or base labels changed')
    pin(out / 'label-protocol.json', {'teacher': teacher, 'positions_sha256': digest(out / 'positions.jsonl'),
                                   'workers': config['workers'], 'chunk_size': config['chunk_size']})
    rows = list(records(out / 'positions.jsonl'))
    chunks = out / 'chunks'
    chunks.mkdir(exist_ok=True)
    started = time.perf_counter()
    chunk_results = {}
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        futures = {
            pool.submit(label_chunk, rows[i:i + config['chunk_size']], chunks / f'{i:07d}.jsonl',
                        config['binary'], teacher): i
            for i in range(0, len(rows), config['chunk_size'])
        }
        for future in as_completed(futures):
            chunk_results[futures[future]] = future.result()
            count = sum(r['positions'] for r in chunk_results.values())
            print(f'Labeled {count:,}/{len(rows):,} new positions ({time.perf_counter()-started:.0f}s)', flush=True)
    destination = Path(config['combined'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    with temporary.open('wb') as output:
        with open(config['base'], 'rb') as base:
            shutil.copyfileobj(base, output)
        for offset in sorted(chunk_results):
            with (chunks / f'{offset:07d}.jsonl').open('rb') as chunk:
                shutil.copyfileobj(chunk, output)
    temporary.replace(destination)
    result = {'sha256': digest(destination), 'base_sha256': digest(config['base']),
              'additional_positions': len(rows), 'teacher': teacher,
              'label_worker_seconds': sum(r['label_seconds'] for r in chunk_results.values()),
              'label_invocation_wall_seconds': time.perf_counter()-started,
              'timing_note': 'Worker times sum completed chunks; wall time is this invocation only.',
              'paid_cost_cad': 0, 'provenance': provenance()}
    write_json(str(destination) + '.manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    print(json.dumps(expand(read_json(args.config)), indent=2))


if __name__ == '__main__':
    main()
