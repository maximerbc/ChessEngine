"""Teacher-free evolutionary evaluator. Every evaluation coefficient is learned."""
from pathlib import Path
import json
import numpy as np
import chess
from .common import digest

FORMAT = 'chessengine-evolution-v1'
FEATURE_SCHEMA = 'evolution-summary-v1'
FEATURE_COUNT = 63
PARAMETER_COUNT = 324
LIMITS = (8, 2, 2, 2, 1)


def centrality(square):
    return 3.5 - (abs(chess.square_file(square) - 3.5) + abs(chess.square_rank(square) - 3.5)) / 2


def features(board):
    """Counts, center, advance, pawns, kings, attacks, phase and original state.

    Within each group own-side entries precede opponent entries. No material
    weights enter this representation. Attack coverage is pseudo-legal, not
    legal mobility; promotions are retained without clipping piece counts.
    """
    sides = (board.turn, not board.turn)
    pieces = [[list(board.pieces(p, c)) for p in range(1, 6)] for c in sides]
    values = [len(squares) / LIMITS[p] for row in pieces for p, squares in enumerate(row)]
    values += [sum(centrality(s) for s in squares) / (3.5 * LIMITS[p])
               for row in pieces for p, squares in enumerate(row)]
    values += [sum(chess.square_rank(s) if c else 7-chess.square_rank(s) for s in row[0]) / 48
               for c, row in zip(sides, pieces)]
    for c, row in zip(sides, pieces):
        pawns = row[0]
        enemy = board.pieces(chess.PAWN, not c)
        files = {chess.square_file(s) for s in pawns}
        doubled = len(pawns)-len(files)
        isolated = passed = defended = 0
        for s in pawns:
            f, r = chess.square_file(s), chess.square_rank(s)
            isolated += not ({f-1, f+1} & files)
            passed += not any(abs(chess.square_file(t)-f) <= 1 and
                              ((chess.square_rank(t)>r) if c else (chess.square_rank(t)<r)) for t in enemy)
            defended += bool(board.attackers(c, s) & board.pieces(chess.PAWN, c))
        values += [doubled/8, isolated/8, passed/8, defended/8]
    for c in sides:
        king = board.king(c)
        if king is None:
            raise ValueError('Evolutionary evaluator requires both kings')
        f, r = chess.square_file(king), chess.square_rank(king)
        ahead = r + (1 if c else -1)
        shield = sum(board.piece_at(chess.square(t, ahead)) == chess.Piece(chess.PAWN, c)
                     for t in range(max(0, f-1), min(8, f+2))) if 0 <= ahead < 8 else 0
        danger = sum(board.is_attacked_by(not c, s) for s in chess.SquareSet(chess.BB_KING_ATTACKS[king]))
        values += [shield/3, danger/8, centrality(king)/3.5]
    for c, row in zip(sides, pieces):
        for squares in row:
            attacks = 0
            for s in squares:
                attacks |= int(board.attacks(s))
            values.append((attacks & ~board.occupied_co[c]).bit_count()/64)
    values += [sum(len(row[p]) for row in pieces for p in range(1, 5))/14]
    values += [float(right) for c in sides for right in
               (board.has_kingside_castling_rights(c), board.has_queenside_castling_rights(c))]
    ep = 8 if board.ep_square is None else chess.square_file(board.ep_square)
    values += [float(i == ep) for i in range(9)]
    values += [min(board.halfmove_clock/150, 1), float(board.is_repetition(2)), float(board.is_repetition(3))]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (FEATURE_COUNT,) or not np.isfinite(result).all():
        raise ValueError('Invalid evolutionary features')
    return result


def prediction(parameters, x):
    p = np.asarray(parameters, dtype=np.float32)
    if p.shape != (PARAMETER_COUNT,) or not np.isfinite(p).all():
        raise ValueError('Invalid evolutionary parameters')
    hidden = np.maximum(np.asarray(x, dtype=np.float32) @ p[:252].reshape(4, 63).T + p[252:256], 0)
    return np.tanh(hidden @ p[256:260] + np.asarray(x, dtype=np.float32) @ p[260:323] + p[323])


def export(parameters, path, metadata=None):
    p = np.asarray(parameters, dtype=np.float32)
    prediction(p, np.zeros(FEATURE_COUNT, dtype=np.float32))
    meta = dict(metadata or {}, format=FORMAT, feature_schema=FEATURE_SCHEMA, kind='evolutionary',
                evidence='evolutionary-tournaments', parameter_count=PARAMETER_COUNT)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open the exact path to avoid numpy appending a suffix to atomic staging names.
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, parameters=p, metadata=np.array(json.dumps(meta, sort_keys=True)))
    temp.replace(path)


class EvolutionEvaluator:
    def __init__(self, path=None, parameters=None):
        if path is not None:
            with np.load(path, allow_pickle=False) as data:
                self.metadata = json.loads(str(data['metadata']))
                if (self.metadata.get('format') != FORMAT or self.metadata.get('feature_schema') != FEATURE_SCHEMA
                        or self.metadata.get('kind') != 'evolutionary'
                        or self.metadata.get('parameter_count') != PARAMETER_COUNT):
                    raise ValueError('Unsupported evolutionary checkpoint schema')
                self.parameters = data['parameters'].copy()
            self.checkpoint_id = digest(path)
        else:
            self.parameters = np.asarray(parameters, dtype=np.float32).copy()
            self.metadata = dict(kind='evolutionary')
        prediction(self.parameters, np.zeros(FEATURE_COUNT, dtype=np.float32))

    def __call__(self, board):
        return float(prediction(self.parameters, features(board)))


def load_evaluator(path):
    """Dispatch by an explicit format, never by guessed parameter shapes."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
    if metadata.get('format') == FORMAT:
        return EvolutionEvaluator(path)
    if metadata.get('kind') == 'evolutionary':
        raise ValueError('Unsupported evolutionary checkpoint schema')
    from .model import NumpyEvaluator
    return NumpyEvaluator(path)
