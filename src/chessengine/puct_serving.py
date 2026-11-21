"""CPU-only value PUCT. No training library enters production imports."""
from dataclasses import dataclass
import numpy as np
from .engine import Search, SearchResult
from .puct import PUCT
from .puct_actions import action_map

STRATEGY = 'puct-uniform-v1'


class UniformValue:
    def __init__(self, evaluator):
        self.evaluate = evaluator

    def predict(self, board):
        mapping = action_map(board)
        moves = [mapping[label] for label in sorted(mapping)]
        return moves, np.full(len(moves), 1 / len(moves)), float(self.evaluate(board))


@dataclass
class PUCTResult(SearchResult):
    strategy: str = STRATEGY
    simulations: int = 0
    root_coverage: float = 0.


class ValueSearch:
    def __init__(self, evaluator, simulations=64, time_ms=1500):
        self.search = PUCT(UniformValue(evaluator), simulations=simulations, c_puct=1.5, time_ms=time_ms)

    def run(self, board):
        result = self.search.run(board)
        return PUCTResult(result.move, result.value, result.max_depth, 0, result.elapsed_ms,
            [result.move] if result.move else [],
            board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else None,
            simulations=result.simulations, root_coverage=result.root_coverage)


def search_for(evaluator, strategy='alpha-beta', nodes=10000, time_ms=1500, simulations=64):
    if strategy == 'alpha-beta':
        return Search(evaluator, nodes, time_ms)
    if strategy == STRATEGY:
        return ValueSearch(evaluator, simulations, time_ms)
    raise ValueError('Unknown declared search strategy')
