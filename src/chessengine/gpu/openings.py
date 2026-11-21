"""Fresh whole-game-separated opening pools, frozen before any learning."""
import json
from pathlib import Path
import random
import chess.pgn
from ..common import json_hash
from ..data import open_pgn, records
from ..features import position_key, record_board, restore
from .protocol import freeze, atomic_json, sha
from .dataset import research_inputs, research_exclusions, puzzle_exclusions, site_id


def prepare(config, sources_manifest, dataset, output):
    paths = [dataset, *config['historical_datasets']]
    excluded_games, excluded_positions = set(), set()
    for path in paths:
        for row in records(path):
            excluded_games.add(row['game_id'])
            excluded_positions.add(position_key(chess.Board(row['fen'])))
    for path in config['reserved']:
        for rows in json.loads(Path(path).read_text()).values():
            for row in rows:
                excluded_positions.add(position_key(restore(row)))
                if row.get('source_game_id'):
                    excluded_games.add(row['source_game_id'])
    historical_research = research_inputs(config)
    other_positions, other_games = research_exclusions(historical_research)
    excluded_positions.update(other_positions)
    excluded_games.update(other_games)
    puzzle_positions,puzzle_sites = puzzle_exclusions(config.get('reserved_puzzle_csv',[]))
    excluded_positions.update(puzzle_positions)
    quotas = config['opening_pairs']
    pools = {name: [] for name in quotas}
    protocol = freeze(Path(output).parent / 'opening-protocol', config,
                      [sources_manifest, *paths, *config['reserved'], *historical_research,
                       *config.get('reserved_puzzle_csv',[])])
    sources = json.loads(Path(sources_manifest).read_text())
    for source in sources:
        if sha(source['file']) != source['sha256']:
            raise ValueError('Source archive changed')
        with open_pgn(source['file']) as stream:
            while any(len(pools[k]) < quotas[k] for k in quotas):
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                if game.errors or game.headers.get('Variant', 'Standard') != 'Standard':
                    continue
                if site_id(game.headers.get('Site','')) in puzzle_sites:
                    continue
                board = game.board()
                if board.chess960 or not board.is_valid():
                    continue
                moves = list(game.mainline_moves())
                game_id = json_hash({'initial': board.fen(), 'moves': [m.uci() for m in moves]})
                if game_id in excluded_games or len(moves) < 16:
                    continue
                bucket = int(json_hash(['gpu-openings-v1', game_id])[:8], 16) % 20
                group = ('train' if bucket < 10 else 'pilot_train' if bucket < 14 else
                         ['development','qualification','qualification','final','final','pilot'][bucket - 14])
                if len(pools[group]) >= quotas[group]:
                    continue
                ply = random.Random(game_id).randrange(12, min(len(moves), 81))
                valid_history = True
                for move in moves[:ply]:
                    if board.is_game_over(claim_draw=True):
                        valid_history = False
                        break
                    board.push(move)
                key = position_key(board)
                if not valid_history or key in excluded_positions or board.is_game_over(claim_draw=True):
                    continue
                pools[group].append(dict(record_board(board), id=json_hash([game_id, ply]), source_game_id=game_id))
                excluded_games.add(game_id)
                excluded_positions.add(key)
    if any(len(pools[k]) != quotas[k] for k in quotas):
        raise ValueError('Fresh opening pools exhausted; preserve protocol and extend sources prospectively')
    atomic_json(output, dict(pools, protocol_id=protocol))
    return {k: len(v) for k, v in pools.items()}
