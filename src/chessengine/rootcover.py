"""CPU counterpart of the GPU-native root-coverage PUCT research strategy."""
from .puct import PUCT
from .puct_serving import UniformValue, PUCTResult

STRATEGY = 'puct-rootcover-v1'


class RootCoverageValueSearch:
    def __init__(self, evaluator, simulations=64, time_ms=1500):
        self.search = PUCT(UniformValue(evaluator), simulations=simulations, c_puct=1.5,
                           time_ms=time_ms, root_coverage=True)

    def run(self, board):
        result = self.search.run(board)
        return PUCTResult(result.move, result.value, result.max_depth, 0, result.elapsed_ms,
            [result.move] if result.move else [],
            board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else None,
            strategy=STRATEGY, simulations=result.simulations, root_coverage=result.root_coverage)
