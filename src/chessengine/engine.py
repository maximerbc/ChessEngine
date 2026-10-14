from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from typing import Protocol
import chess

MATE = 10000.0
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3.2, chess.BISHOP: 3.3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class Evaluator(Protocol):
    def __call__(self, board: chess.Board) -> float: ...


class Handcrafted:
    def __call__(self, board: chess.Board) -> float:
        score = 0.0
        for square, piece in board.piece_map().items():
            rank = chess.square_rank(square) if piece.color else 7 - chess.square_rank(square)
            file = chess.square_file(square)
            centrality = 3.5 - (abs(file - 3.5) + abs(rank - 3.5)) / 2
            bonus = 0.05 * centrality if piece.piece_type in (chess.KNIGHT, chess.BISHOP) else 0
            if piece.piece_type == chess.PAWN:
                bonus += rank * 0.04
            score += (VALUES[piece.piece_type] + bonus) * (1 if piece.color else -1)
        return math.tanh(score / 6) * (1 if board.turn else -1)


@dataclass
class SearchResult:
    move: str | None
    value: float
    depth: int
    nodes: int
    elapsed_ms: float
    pv: list[str]
    terminal: str | None = None
    root_rankings: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class Stopped(Exception):
    pass


class Search:
    """A fresh instance/table per request or move. History-exact TT keys favor correctness."""

    def __init__(
        self,
        evaluator: Evaluator,
        nodes: int = 2000,
        time_ms: int | None = None,
        max_depth: int = 64,
        qdepth: int = 4,
        ranker=None,
    ):
        self.evaluate = evaluator
        self.limit = max(1, nodes)
        self.time_ms = time_ms
        self.max_depth = max_depth
        self.qdepth = qdepth
        self.ranker = ranker
        self.root_order = {}
        self.nodes = 0
        self.tt: dict = {}
        self.deadline = float("inf")

    def tick(self):
        if self.nodes >= self.limit or time.perf_counter() >= self.deadline:
            raise Stopped
        self.nodes += 1

    @staticmethod
    def terminal(board: chess.Board, ply: int) -> float | None:
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0.0
        return (MATE - ply) * (1 if outcome.winner == board.turn else -1)

    @staticmethod
    def key(board: chess.Board):
        # Same FEN can have a different claimable repetition; full history is intentional.
        return board.fen(), tuple(board.move_stack)

    @staticmethod
    def ordered(board: chess.Board, preferred=None, tactical=False):
        moves = list(board.legal_moves)
        if tactical and not board.is_check():
            moves = [m for m in moves if board.is_capture(m) or m.promotion]

        def priority(move):
            victim = board.piece_type_at(move.to_square)
            attacker = board.piece_type_at(move.from_square)
            return (
                (100 if move == preferred else 0)
                + (10 * VALUES.get(victim, 0) - VALUES.get(attacker, 0) if board.is_capture(move) else 0)
                + VALUES.get(move.promotion, 0)
            )

        return sorted(moves, key=priority, reverse=True)

    def quiet(self, board, alpha, beta, ply, remaining, counted=False):
        if not counted:
            self.tick()
        terminal = self.terminal(board, ply)
        if terminal is not None:
            return terminal, []
        stand = self.evaluate(board)
        # Hard bound on quiescence; check evasions are searched while budget remains.
        if remaining <= 0:
            return stand, []
        best = -float("inf") if board.is_check() else stand
        best_line = []
        if not board.is_check():
            if best >= beta:
                return best, []
            alpha = max(alpha, best)
        for move in self.ordered(board, tactical=True):
            board.push(move)
            try:
                value, line = self.quiet(board, -beta, -alpha, ply + 1, remaining - 1)
                value = -value
            finally:
                board.pop()
            if value > best:
                best, best_line = value, [move] + line
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best, best_line

    def negamax(self, board, depth, alpha, beta, ply):
        self.tick()
        terminal = self.terminal(board, ply)
        if terminal is not None:
            return terminal, []
        if depth == 0:
            return self.quiet(board, alpha, beta, ply, self.qdepth, counted=True)
        key = self.key(board)
        old_alpha = alpha
        entry = self.tt.get(key)
        preferred = entry[3][0] if entry and entry[3] else None
        if entry and entry[0] >= depth:
            _, value, flag, line = entry
            if flag == "exact" or (flag == "lower" and value >= beta) or (flag == "upper" and value <= alpha):
                return value, line
        best, best_line = -float("inf"), []
        moves = self.ordered(board, preferred)
        if ply == 0 and self.root_order:
            moves.sort(key=lambda move: (move != preferred, self.root_order[move.uci()]))
        for move in moves:
            board.push(move)
            try:
                value, line = self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
                value = -value
            finally:
                board.pop()
            if value > best:
                best, best_line = value, [move] + line
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        flag = "upper" if best <= old_alpha else "lower" if best >= beta else "exact"
        self.tt[key] = (depth, best, flag, best_line)
        return best, best_line

    def run(self, board: chess.Board) -> SearchResult:
        board = board.copy(stack=True)
        start = time.perf_counter()
        self.nodes, self.tt = 0, {}
        self.root_order = {}
        self.deadline = start + self.time_ms / 1000 if self.time_ms is not None else float("inf")
        terminal = self.terminal(board, 0)
        if terminal is not None:
            return SearchResult(None, terminal, 0, 0, 0, [], board.result(claim_draw=True))
        rankings = self.ranker.rank(board) if self.ranker is not None else []
        if rankings:
            legal = {move.uci() for move in board.legal_moves}
            if len(rankings) != len(legal) or {row['move'] for row in rankings} != legal:
                raise ValueError('Ranker must include every legal root move exactly once')
            self.root_order = {row['move']: i for i, row in enumerate(rankings)}
        fallback = chess.Move.from_uci(rankings[0]['move']) if rankings else self.ordered(board)[0]
        best_line, best_value, depth_done = [fallback], self.evaluate(board), 0
        for depth in range(1, self.max_depth + 1):
            try:
                value, line = self.negamax(board, depth, -float("inf"), float("inf"), 0)
                if line:
                    best_line, best_value, depth_done = line, value, depth
                if abs(value) > MATE - 1000:
                    break
            except Stopped:
                break
        return SearchResult(
            best_line[0].uci(),
            best_value,
            depth_done,
            self.nodes,
            (time.perf_counter() - start) * 1000,
            [m.uci() for m in best_line],
            root_rankings=rankings,
        )
