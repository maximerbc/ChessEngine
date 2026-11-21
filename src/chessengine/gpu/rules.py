"""PGX 2.6 chess with original histories and automatic intended-move draw claims.

Only low-level legal transitions are used. PGX's built-in 512-ply termination is
not a rules draw. All public conversions validate against python-chess.
"""
from typing import NamedTuple
import chess
import numpy as np
import jax
import jax.numpy as jnp
from pgx._src.games import chess as pg

HISTORY = 2048
LABELS = jnp.arange(4672)
FROM = LABELS // 73
TO = pg.FROM_PLANE[FROM, LABELS % 73]
GAME = pg.Game()


class State(NamedTuple):
    game: pg.GameState
    repeated: jax.Array
    overflow: jax.Array


def canonical_hash(game):
    legal_ep = jnp.any(game.legal_action_mask & (game.board[FROM] == 1) & (TO == game.en_passant))
    return pg._zobrist_hash(game._replace(en_passant=jnp.where(legal_ep, game.en_passant, -1)))


def repetitions(game):
    valid = jnp.arange(HISTORY) <= game.step_count
    return jnp.sum(jnp.all(game.hash_history == canonical_hash(game), axis=1) & valid)


def step(state, action):
    game = GAME.step(state.game, action)
    key = canonical_hash(game)
    game = game._replace(hash_history=game.hash_history.at[0].set(key))
    count = repetitions(game)
    repeated = jnp.where(game.halfmove_count == 0, False, state.repeated | (count >= 2))
    return State(game, repeated, state.overflow | (game.step_count >= HISTORY - 1))


def terminal(state):
    game = state.game
    legal = game.legal_action_mask
    no_moves = ~jnp.any(legal)
    mate = no_moves & pg._is_checked(game)
    current_draw = (game.halfmove_count >= 100) | (repetitions(game) >= 3) | pg.has_insufficient_pieces(game)

    def intended_claim(_):
        # All 4672 labels are represented; the loop only evaluates legal actions.
        # No fixed legal-move shortlist can silently drop promotions or evasions.
        actions = jnp.nonzero(legal, size=4672, fill_value=0)[0]
        def body(index, found):
            action = actions[index]
            def check(_):
                child = step(state, action)
                fifty = (child.game.halfmove_count >= 100) & child.game.legal_action_mask.any()
                return fifty | (repetitions(child.game) >= 3)
            found |= jax.lax.cond(legal[action] & ~found, check, lambda _: jnp.bool_(False), None)
            return found
        return jax.lax.fori_loop(0, legal.sum(), body, jnp.bool_(False))

    claim = jax.lax.cond((game.halfmove_count >= 99) | state.repeated,
                         intended_claim, lambda _: jnp.bool_(False), None)
    done = no_moves | current_draw | claim
    value = jnp.where(mate, -1.0, 0.0).astype(jnp.float32)
    return done, value


def features(state):
    game = state.game
    # PGX is file-major and already player-relative; v3 is rank-major.
    board = game.board.reshape(8, 8).T.reshape(64)
    pieces = jnp.concatenate([jnp.arange(1, 7), -jnp.arange(1, 7)])
    planes = (pieces[:, None] == board[None, :]).astype(jnp.float32)
    rights = game.castling_rights[:, ::-1].reshape(4).astype(jnp.float32)
    ep = jnp.where(game.en_passant < 0, 64, (game.en_passant % 8) * 8 + game.en_passant // 8)
    counts = planes.reshape(2, 6, 64).sum(-1)[:, :5]
    normalized = counts / jnp.array([8, 2, 2, 2, 1])
    phase = (counts * jnp.array([0, 1, 1, 2, 4])).sum() / 24
    balance = ((counts[0] - counts[1]) * jnp.array([1, 3.2, 3.3, 5, 9])).sum() / 40
    rep = repetitions(game)
    return jnp.concatenate([planes.reshape(-1), jnp.ones(1), rights,
        jax.nn.one_hot(ep, 65), jnp.array([jnp.minimum(game.halfmove_count / 150, 1), rep >= 2, rep >= 3]),
        normalized.reshape(10), jnp.array([phase, balance])]).astype(jnp.float32)


def handcrafted(state):
    game = state.game
    square = jnp.arange(64)
    rank = square % 8
    file = square // 8
    pieces = jnp.abs(game.board)
    relative_rank = jnp.where(game.board > 0, rank, 7 - rank)
    center = 3.5 - (jnp.abs(file - 3.5) + jnp.abs(relative_rank - 3.5)) / 2
    bonus = jnp.where((pieces == 2) | (pieces == 3), .05 * center, 0)
    bonus += jnp.where(pieces == 1, .04 * relative_rank, 0)
    score = ((jnp.array([0, 1, 3.2, 3.3, 5, 9, 0])[pieces] + bonus) * jnp.sign(game.board)).sum()
    return jnp.tanh(score / 6)


def action_to_move(action, color):
    action = int(action)
    origin, plane = divmod(action, 73)
    target = int(np.asarray(pg.FROM_PLANE)[origin, plane])
    def square(p):
        f, r = divmod(p, 8)
        return chess.square(f, r if color == chess.WHITE else 7 - r)
    promotion = [chess.ROOK, chess.BISHOP, chess.KNIGHT][plane // 3] if plane < 9 else None
    return chess.Move(square(origin), square(target), promotion=promotion)


def action_map(board):
    """Canonical label ordering shared by NumPy PUCT and PGX search."""
    plane = np.asarray(pg.TO_PLANE)
    mapping = {}
    for move in board.legal_moves:
        def relative(s):
            f, r = chess.square_file(s), chess.square_rank(s)
            return f * 8 + (r if board.turn else 7 - r)
        origin, target = relative(move.from_square), relative(move.to_square)
        if move.promotion and move.promotion != chess.QUEEN:
            under = [chess.ROOK, chess.BISHOP, chess.KNIGHT].index(move.promotion)
            # Underpromotion direction follows PGX's lookup, avoiding a second rule table.
            candidates = [p for p in range(under * 3, under * 3 + 3)
                          if int(np.asarray(pg.FROM_PLANE)[origin, p]) == target]
            if len(candidates) != 1:
                raise ValueError('Ambiguous underpromotion mapping')
            label = origin * 73 + candidates[0]
        else:
            label = origin * 73 + int(plane[origin, target])
        if label in mapping:
            raise ValueError('Action mapping collision')
        mapping[label] = move
    return mapping


def from_board(board):
    """Import the root, then replay every original move. Never fabricate repetitions."""
    from pgx.experimental.chess import from_fen
    if len(board.move_stack) >= HISTORY - 129:
        raise ValueError('History exceeds the declared search capacity')
    root = board.root()
    game = from_fen(root.fen(en_passant='fen'))._x
    game = game._replace(step_count=jnp.int32(0), hash_history=jnp.zeros((HISTORY, 2), jnp.uint32))
    game = game._replace(hash_history=game.hash_history.at[0].set(canonical_hash(game)))
    state = State(game, jnp.bool_(False), jnp.bool_(False))
    current = root.copy(stack=True)
    apply_step = jax.jit(step)
    for move in board.move_stack:
        mapping = action_map(current)
        label = next(a for a, m in mapping.items() if m == move)
        state = apply_step(state, jnp.int32(label))
        current.push(move)
    assert_parity(board, state)
    return state


def assert_parity(board, state):
    from ..features import encode
    legal = set(np.flatnonzero(np.asarray(state.game.legal_action_mask)))
    mapping = action_map(board)
    if legal != set(mapping):
        raise ValueError('PGX/python-chess legal action mismatch')
    if not np.allclose(np.asarray(features(state)), encode(board, 3), atol=1e-6, rtol=1e-6):
        raise ValueError('PGX/python-chess feature or history mismatch')
    done, value = jax.jit(terminal)(state)
    outcome = board.outcome(claim_draw=True)
    expected = 0 if outcome is None or outcome.winner is None else (1 if outcome.winner == board.turn else -1)
    if bool(done) != (outcome is not None) or float(value) != expected or bool(state.overflow):
        raise ValueError('PGX/python-chess termination or history mismatch')
