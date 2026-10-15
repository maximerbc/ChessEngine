"""History-preserving research PUCT. Edge Q is from the selecting player's perspective."""

from dataclasses import dataclass
import time
import numpy as np


class Node:
    def __init__(self, moves, priors):
        self.moves = moves
        self.priors = np.asarray(priors, dtype=np.float64)
        if (
            len(moves) != len(priors)
            or not np.isfinite(self.priors).all()
            or np.any(self.priors < 0)
            or not np.isclose(self.priors.sum(), 1)
        ):
            raise ValueError("Invalid legal priors")
        self.n = np.zeros(len(moves), dtype=np.int64)
        self.w = np.zeros(len(moves), dtype=np.float64)
        self.children = {}

    def select(self, c):
        q = np.divide(self.w, self.n, out=np.zeros_like(self.w), where=self.n > 0)
        # max(1,N) makes the first simulation prior-guided; ties use legal generator order.
        return int(np.argmax(q + c * self.priors * np.sqrt(max(1, self.n.sum())) / (1 + self.n)))


@dataclass
class Result:
    move: str | None
    moves: list
    visits: np.ndarray
    policy: np.ndarray
    value: float
    simulations: int
    network_calls: int
    expanded_leaves: int
    terminal_hits: int
    traversed_edges: int
    max_depth: int
    elapsed_ms: float
    root_coverage: float
    root_max_share: float


class PUCT:
    def __init__(self, model, simulations=128, c_puct=1.5, time_ms=None, noise_fraction=0.0, noise_mass=10.0):
        if simulations < 1 or c_puct <= 0 or not 0 <= noise_fraction <= 1 or noise_mass <= 0:
            raise ValueError("Invalid PUCT settings")
        if time_ms is not None and time_ms <= 0:
            raise ValueError("time_ms must be positive")
        self.model, self.simulations, self.c = model, simulations, c_puct
        self.time_ms, self.noise_fraction, self.noise_mass = time_ms, noise_fraction, noise_mass

    def run(self, board, rng=None):
        started = time.perf_counter()
        original = board.copy(stack=True)
        outcome = original.outcome(claim_draw=True)
        if outcome is not None:
            v = 0.0 if outcome.winner is None else (1.0 if outcome.winner == original.turn else -1.0)
            return Result(
                None,
                [],
                np.array([], dtype=np.int64),
                np.array([]),
                v,
                0,
                0,
                0,
                1,
                0,
                0,
                (time.perf_counter() - started) * 1000,
                0.0,
                0.0,
            )
        moves, priors, _ = self.model.predict(original)
        root = Node(moves, priors)
        if self.noise_fraction:
            if rng is None:
                raise ValueError("Training noise requires an explicit RNG")
            noise = rng.dirichlet(np.full(len(moves), self.noise_mass / len(moves)))
            root.priors = (1 - self.noise_fraction) * root.priors + self.noise_fraction * noise
        calls, expanded, terminal, edges, max_depth, completed = 1, 1, 0, 0, 0, 0
        for _ in range(self.simulations):
            if (
                completed
                and self.time_ms is not None
                and (time.perf_counter() - started) * 1000 >= self.time_ms
            ):
                break
            position, node, path = original.copy(stack=True), root, []
            while True:
                i = node.select(self.c)
                path.append((node, i))
                position.push(node.moves[i])
                edges += 1
                outcome = position.outcome(claim_draw=True)
                if outcome is not None:
                    value = (
                        0.0 if outcome.winner is None else (1.0 if outcome.winner == position.turn else -1.0)
                    )
                    terminal += 1
                    break
                if i not in node.children:
                    legal, p, value = self.model.predict(position)
                    node.children[i] = Node(legal, p)
                    calls += 1
                    expanded += 1
                    break
                node = node.children[i]
            max_depth = max(max_depth, len(path))
            for parent, i in reversed(path):
                value = -value
                parent.n[i] += 1
                parent.w[i] += value
            completed += 1
        policy = root.n / root.n.sum()
        best = int(np.argmax(root.n))
        v = float(root.w[best] / root.n[best])
        return Result(
            moves[best].uci(),
            moves,
            root.n.copy(),
            policy,
            v,
            completed,
            calls,
            expanded,
            terminal,
            edges,
            max_depth,
            (time.perf_counter() - started) * 1000,
            float(np.mean(root.n > 0)),
            float(policy.max()),
        )
