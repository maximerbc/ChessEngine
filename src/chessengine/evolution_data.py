"""Freeze tournament starts without consuming teacher or human outcome labels."""
from pathlib import Path
import random
import chess
import chess.pgn
from .common import digest, json_hash, read_json, write_json, provenance
from .data import open_pgn
from .features import record_board, restore, position_key


def canonical(board):
    return position_key(board if board.turn else board.mirror())


def stratum(board):
    count = len(board.piece_map())-2
    return 'endgame' if count <= 10 else 'middlegame' if count <= 22 else 'opening'


def validate(groups):
    games, keys = set(), set()
    for rows in groups.values():
        for row in rows:
            board = restore(row)
            if row['source_game_id'] in games or canonical(board) in keys:
                raise ValueError('Evolutionary source-game/position split leakage')
            if not board.is_valid() or board.is_game_over(claim_draw=True):
                raise ValueError('Invalid or terminal tournament start')
            games.add(row['source_game_id'])
            keys.add(canonical(board))


def prepare(config):
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=True)
    inputs = {p: digest(p) for p in [config['archive'], config['corpus'], *config['reserved']]}
    signature = json_hash(dict(inputs=inputs, pools=config['pools'], seed=config['sampling_seed']))
    path, manifest = root/'openings.json', root/'openings-manifest.json'
    if path.exists():
        saved = read_json(manifest)
        if saved['signature'] != signature or saved['sha256'] != digest(path):
            raise ValueError('Frozen evolutionary opening inputs changed')
        groups = read_json(path)
        validate(groups)
        return groups
    used_games, used_positions = set(), set()
    # These are exclusion metadata only: never read the label field.
    with open(config['corpus']) as stream:
        for line in stream:
            row = __import__('json').loads(line)
            used_games.add(row['game_id'])
            used_positions.add(canonical(chess.Board(row['fen'])))
    for source in config['reserved']:
        for rows in read_json(source).values():
            for row in rows:
                used_games.add(row['source_game_id'])
                used_positions.add(canonical(restore(row)))
    groups = {name: [] for name in config['pools']}
    # Sampling is independent of source results. Fill disjoint round-robin pools.
    rng = random.Random(config['sampling_seed'])
    scanned = 0
    with open_pgn(config['archive']) as stream:
        while any(len(groups[k]) < n for k, n in config['pools'].items()):
            game = chess.pgn.read_game(stream)
            if game is None:
                raise ValueError('Insufficient fresh tournament positions; do not recycle reserved games')
            scanned += 1
            board = game.board()
            moves = list(game.mainline_moves())
            if (game.errors or len(moves) < 16 or board.chess960 or not board.is_valid()
                    or game.headers.get('Variant', 'Standard') != 'Standard'):
                continue
            source_id = json_hash(dict(initial=board.fen(), moves=[m.uci() for m in moves]))
            if source_id in used_games:
                continue
            available = [k for k, n in config['pools'].items() if len(groups[k]) < n]
            rng.shuffle(available)
            snapshots = {}
            for ply, move in enumerate(moves, 1):
                board.push(move)
                if ply < 12 or board.is_game_over(claim_draw=True) or canonical(board) in used_positions:
                    continue
                kind = stratum(board)
                if kind not in snapshots:
                    snapshots[kind] = record_board(board)
                if ply == 12:
                    snapshots['twelve'] = record_board(board)
            for name in available:
                size = config['pools'][name]
                wanted = ('endgame', 'middlegame', 'opening')[len(groups[name]) % 3]
                balanced = name in ('train', 'pilot')
                snapshot = snapshots.get(wanted if balanced else 'twelve')
                if snapshot is None or len(groups[name]) >= size:
                    continue
                selected = restore(snapshot)
                row = dict(snapshot, source_game_id=source_id, id=source_id[:16], stratum=stratum(selected))
                groups[name].append(row)
                used_games.add(source_id)
                used_positions.add(canonical(selected))
                break
    validate(groups)
    write_json(path, groups)
    write_json(manifest, dict(signature=signature, inputs=inputs, sha256=digest(path), scanned_games=scanned,
                             sizes={k: len(v) for k, v in groups.items()}, source_game_overlaps=0,
                             canonical_overlaps=0, corpus_exclusions='all pools (metadata only)', provenance=provenance()))
    return groups
