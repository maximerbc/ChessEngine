"""Search-player adapters and auditable self-play trajectories for policy/value research."""

import time
from pathlib import Path
import chess
import chess.pgn
import numpy as np
from .common import read_json, write_json, digest, json_hash
from .features import restore
from .puct import PUCT
from .policy_value import PolicyValue
from .engine import Search
from .evaluate import evaluator_for, identity, paired_summary


class Player:
    def __init__(self, spec, config):
        self.spec, self.config = spec, config
        self.is_puct = spec.startswith("puct:")
        self.model = PolicyValue(spec[5:]) if self.is_puct else evaluator_for(spec)
        self.identity = self.model.checkpoint_id if self.is_puct else identity(spec)

    def search(self, board):
        c = self.config
        if self.is_puct:
            return PUCT(self.model, c["simulations"], c_puct=c["c_puct"], time_ms=c.get("time_ms")).run(board)
        return Search(
            self.model, nodes=c.get("reference_nodes", 1000000000), time_ms=c.get("time_ms"), qdepth=2
        ).run(board)


def generate(actor, opening, config, seed):
    rng = np.random.default_rng(seed)
    board = restore(opening)
    played, targets, metrics = [], [], []
    started = time.perf_counter()
    for ply in range(config["max_plies"]):
        if board.is_game_over(claim_draw=True):
            break
        result = PUCT(
            actor,
            config["simulations"],
            c_puct=config["c_puct"],
            noise_fraction=config["noise_fraction"],
            noise_mass=config["noise_mass"],
        ).run(board, rng)
        chosen = (
            int(rng.choice(len(result.moves), p=result.policy))
            if ply < config["exploration_plies"]
            else int(np.argmax(result.visits))
        )
        targets.append(result.visits.tolist())
        metrics.append(
            dict(
                simulations=result.simulations,
                network_calls=result.network_calls,
                expanded_leaves=result.expanded_leaves,
                terminal_hits=result.terminal_hits,
                traversed_edges=result.traversed_edges,
                max_depth=result.max_depth,
                ms=result.elapsed_ms,
                legal=len(result.moves),
                root_coverage=result.root_coverage,
                root_max_share=result.root_max_share,
            )
        )
        move = result.moves[chosen]
        board.push(move)
        played.append(move.uci())
    outcome = board.outcome(claim_draw=True)
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Policy/value research self-play"
    game.headers["Result"] = board.result(claim_draw=True)
    game.headers["Termination"] = "truncated" if outcome is None else outcome.termination.name
    return dict(
        opening=opening,
        actor_sha256=actor.checkpoint_id,
        seed=seed,
        played=played,
        visits=targets,
        metrics=metrics,
        truncated=outcome is None,
        winner=None
        if outcome is None or outcome.winner is None
        else ("white" if outcome.winner else "black"),
        pgn=str(game),
        termination=game.headers["Termination"],
        seconds=time.perf_counter() - started,
    )


def matches(candidate, baseline, openings, config, output, split):
    if split not in ("development", "final") or not openings:
        raise ValueError("Matches need named, nonempty evaluation openings")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ca, ba = Player(candidate, config), Player(baseline, config)
    protocol = dict(
        candidate=candidate,
        baseline=baseline,
        candidate_sha256=ca.identity,
        baseline_sha256=ba.identity,
        config=config,
        openings=openings,
        split=split,
    )
    signature = json_hash(protocol)
    path = output / "matches.json"
    if path.exists():
        previous = read_json(path)
        if previous["signature"] != signature:
            raise ValueError("Research match protocol changed")
        games = previous["games"]
    else:
        games = []
    done = {(g["opening_id"], g["candidate_color"]) for g in games}
    for opening in openings:
        for color in (chess.WHITE, chess.BLACK):
            color_name = "white" if color else "black"
            if (opening["id"], color_name) in done:
                continue
            board = restore(opening)
            started = time.perf_counter()
            work = {"candidate": [], "baseline": []}
            for _ in range(config["max_plies"]):
                if board.is_game_over(claim_draw=True):
                    break
                side = "candidate" if board.turn == color else "baseline"
                player = ca if side == "candidate" else ba
                result = player.search(board)
                work[side].append(
                    dict(
                        ms=result.elapsed_ms,
                        **(
                            {
                                "simulations": result.simulations,
                                "network_calls": result.network_calls,
                                "traversed_edges": result.traversed_edges,
                            }
                            if player.is_puct
                            else {"alpha_beta_nodes": result.nodes}
                        ),
                    )
                )
                board.push_uci(result.move)
            outcome = board.outcome(claim_draw=True)
            capped = outcome is None
            score = 0.5 if capped or outcome.winner is None else float(outcome.winner == color)
            game = chess.pgn.Game.from_board(board)
            game.headers["Event"] = f"Policy/value {split} matches"
            game.headers["Result"] = "*" if capped else board.result(claim_draw=True)
            game.headers["Termination"] = (
                "move cap; provisional half-point in statistics" if capped else outcome.termination.name
            )
            games.append(
                dict(
                    opening_id=opening["id"],
                    candidate_color=color_name,
                    score=score,
                    capped=capped,
                    pgn=str(game),
                    seconds=time.perf_counter() - started,
                    work=work,
                )
            )
            write_json(path, dict(signature=signature, protocol=protocol, games=games))
    summary = paired_summary(
        games, config.get("bootstrap_seed", 2026), config.get("bootstrap_resamples", 10000)
    )
    n = len(games)
    cap = sum(g["capped"] for g in games)
    summary["cap_score_bounds"] = [summary["score"] - 0.5 * cap / n, summary["score"] + 0.5 * cap / n]
    summary["matches_sha256"] = digest(path)
    write_json(output / "summary.json", summary)
    return summary
