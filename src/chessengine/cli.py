from __future__ import annotations

import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from .common import read_json


def main():
    parser = argparse.ArgumentParser(prog="chesslab", description="ChessEngine research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reserved", default="configs/openings.json")
    p = sub.add_parser("label")
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--stockfish", required=True)
    p.add_argument("--nodes", type=int, default=10000)
    for name, default in [("train", "configs/supervised.json"), ("selfplay", "configs/rl.json")]:
        p = sub.add_parser(name)
        p.add_argument("source")
        p.add_argument("output")
        p.add_argument("--config", default=default)
        p.add_argument("--resume", action="store_true")
        if name == "selfplay":
            p.add_argument("--openings", default="configs/openings.json")
    p = sub.add_parser("freeze")
    p.add_argument("candidate")
    p.add_argument("baseline")
    p.add_argument("output")
    p.add_argument("--config", default="configs/evaluation.json")
    p.add_argument("--openings", default="configs/openings.json")
    p.add_argument("--split", choices=["development", "test"], default="test")
    p = sub.add_parser("evaluate")
    p.add_argument("protocol")
    p.add_argument("output")
    p = sub.add_parser("select")
    p.add_argument("output")
    p.add_argument("directories", nargs="+")
    p = sub.add_parser("measure")
    p.add_argument("checkpoint")
    p.add_argument("data")
    p.add_argument("output")
    p = sub.add_parser("puzzles")
    p.add_argument("checkpoint")
    p.add_argument("data")
    p.add_argument("output")
    p.add_argument("--nodes", type=int, default=2000)
    p.add_argument("--limit", type=int, default=100)
    p = sub.add_parser("report")
    p.add_argument("manifest")
    p.add_argument("output")
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p = sub.add_parser("export")
    p.add_argument("checkpoint")
    p.add_argument("output")
    p = sub.add_parser("policy-value", help="Research-only policy/value pilot, study and report")
    p.add_argument("--config", default="configs/policy-value.json")
    p.add_argument("--stage", choices=["benchmark", "prepare", "pilot", "all", "report"], default="pilot")
    p = sub.add_parser("evolutionary", help="Teacher-free evolutionary study and research archive")
    p.add_argument("--config", default="configs/evolutionary.json")
    p.add_argument("--stage", choices=["prepare", "benchmark", "pilot", "train", "evaluate", "report", "release", "all"], default="all")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == "evolutionary":
            from .evolution_study import run
            result = run(read_json(args.config), args.stage)
        elif args.command == "policy-value":
            from .pv_study import benchmark, run, report
            cfg = read_json(args.config)
            result = benchmark(cfg) if args.stage == "benchmark" else report(cfg) if args.stage == "report" else run(cfg, args.stage)
        elif args.command == "prepare":
            from .data import prepare

            result = prepare(args.source, args.output, args.limit, args.seed, args.reserved)
        elif args.command == "label":
            from .data import label

            result = label(args.source, args.output, args.stockfish, args.nodes)
        elif args.command == "train":
            from .train import fit

            result = fit(args.source, args.output, read_json(args.config), args.resume)
        elif args.command == "selfplay":
            from .rl import selfplay

            result = selfplay(args.source, args.openings, args.output, read_json(args.config), args.resume)
        elif args.command == "freeze":
            from .evaluate import freeze_protocol

            result = freeze_protocol(
                args.candidate, args.baseline, args.openings, read_json(args.config), args.output, args.split
            )
        elif args.command == "evaluate":
            from .evaluate import run_matches

            result = run_matches(args.protocol, args.output)
        elif args.command == "select":
            from .evaluate import select_checkpoint

            result = select_checkpoint(args.directories, args.output)
        elif args.command == "measure":
            from .evaluate import measure

            result = measure(args.checkpoint, args.data, args.output)
        elif args.command == "puzzles":
            from .evaluate import puzzles

            result = puzzles(args.checkpoint, args.data, args.output, args.nodes, args.limit)
        elif args.command == "report":
            from .report import generate

            result = generate(args.manifest, args.output)
        elif args.command == "export":
            from .model import NumpyEvaluator
            import shutil

            NumpyEvaluator(args.checkpoint)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.checkpoint, args.output)
            result = {"export": args.output}
        else:
            import uvicorn

            uvicorn.run("chessengine.api:app", host=args.host, port=args.port)
            return
    print(json.dumps(result, indent=2))
