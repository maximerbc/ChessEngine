"""Staged GPU study operations; resource provisioning is always external."""
import argparse
import json
from pathlib import Path
import sys
from .protocol import atomic_json, require_execution, begin_execution, identity, sha, nvidia_available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/gpu-study.json')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('backup')
    p.add_argument('--root', default='.')
    p.add_argument('--extra-root', action='append', default=[], metavar='NAME=PATH')
    p.add_argument('--output',required=True)
    for name in ['verify-backup', 'restore-backup']:
        p = sub.add_parser(name)
        p.add_argument('--archive', required=True)
        p.add_argument('--inventory', required=True)
        if name == 'restore-backup':
            p.add_argument('--destination', required=True)
    p = sub.add_parser('doctor')
    p.add_argument('--require-nvidia', action='store_true')
    p = sub.add_parser('download')
    p.add_argument('--directory', default='data/gpu-value-v1/sources')
    for name in ['prepare-data', 'openings']:
        p = sub.add_parser(name)
        p.add_argument('--sources', default='data/gpu-value-v1/sources/sources.json')
        p.add_argument('--directory', default='data/gpu-value-v1')
    p = sub.add_parser('pin-teacher')
    p.add_argument('--binary', required=True)
    p.add_argument('--eval-file', required=True)
    p.add_argument('--eval-file-small', help='Required only when Stockfish declares EvalFileSmall')
    p.add_argument('--output', default='data/gpu-value-v1/teacher.json')
    p = sub.add_parser('label')
    p.add_argument('--directory', default='data/gpu-value-v1')
    p.add_argument('--binary', required=True)
    p.add_argument('--teacher', default='data/gpu-value-v1/teacher.json')
    p.add_argument('--workers', required=True, type=int)
    p.add_argument('--approval', required=True)
    p = sub.add_parser('features')
    p.add_argument('--directory', default='data/gpu-value-v1')
    p = sub.add_parser('benchmark')
    p.add_argument('--openings', default='data/gpu-value-v1/openings.json')
    p.add_argument('--output', default='runs/gpu-value-v1/benchmark')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--supervised')
    p.add_argument('--approval')
    p = sub.add_parser('resolve-workload')
    p.add_argument('--benchmark', required=True)
    p.add_argument('--games-per-candidate', type=int, choices=[24,48,96], default=48)
    p.add_argument('--output', default='runs/gpu-value-v1/resolved.json')
    p = sub.add_parser('benchmark-components')
    p.add_argument('--tournament',required=True)
    p.add_argument('--positions',default='data/gpu-value-v1/positions.jsonl')
    p.add_argument('--teacher',default='data/gpu-value-v1/teacher.json')
    p.add_argument('--binary',required=True)
    p.add_argument('--openings',default='data/gpu-value-v1/openings.json')
    p.add_argument('--legacy-supervised',default='artifacts/models/supervised.npz')
    p.add_argument('--output',default='runs/gpu-value-v1/components')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--approval',required=True)
    p = sub.add_parser('forecast')
    p.add_argument('--measurement', required=True)
    p.add_argument('--hourly-rate', type=float, required=True)
    p.add_argument('--storage-gb', type=float, required=True)
    p.add_argument('--storage-monthly-rate', type=float, required=True)
    p.add_argument('--output', default='runs/gpu-value-v1/forecast.json')
    p = sub.add_parser('train-supervised')
    p.add_argument('--seed', type=int, choices=[42,43,44], required=True)
    p.add_argument('--data', default='data/gpu-value-v1')
    p.add_argument('--output', default='runs/gpu-value-v1/supervised')
    p.add_argument('--approval', required=True)
    for name in ['pilot', 'evolve']:
        p = sub.add_parser(name)
        p.add_argument('--seed', type=int, choices=[20261121] if name=='pilot' else [42,43,44],
                       default=20261121 if name=='pilot' else 42)
        p.add_argument('--supervised', default='runs/gpu-value-v1/supervised/seed-42/best.npz')
        p.add_argument('--openings', default='data/gpu-value-v1/openings.json')
        p.add_argument('--output', default='runs/gpu-value-v1')
        p.add_argument('--batch-size', type=int, default=32)
        p.add_argument('--approval', required=True)
    p = sub.add_parser('pilot-decision')
    p.add_argument('--directory', default='runs/gpu-value-v1/pilot')
    p.add_argument('--initial-matches', required=True)
    p.add_argument('--last-matches', required=True)
    p.add_argument('--coverage', required=True)
    for name in ['select-supervised', 'select-evolution']:
        p = sub.add_parser(name)
        p.add_argument('--directory', required=True)
        p.add_argument('--output', required=True)
        if name == 'select-evolution':
            p.add_argument('--workers', type=int, default=1)
            p.add_argument('--openings', default='data/gpu-value-v1/openings.json')
            p.add_argument('--legacy-supervised', default='artifacts/models/supervised.npz')
            p.add_argument('--approval', required=True)
    p = sub.add_parser('final-position-test')
    p.add_argument('--data', default='data/gpu-value-v1')
    p.add_argument('--selections', required=True)
    p.add_argument('--output', required=True)
    for name in ['evaluate', 'pilot-check']:
        p = sub.add_parser(name)
        p.add_argument('--openings', default='data/gpu-value-v1/openings.json')
        p.add_argument('--output', required=True)
        p.add_argument('--approval', required=True)
        p.add_argument('--workers', type=int, default=1)
        if name == 'evaluate':
            p.add_argument('--selections', required=True)
            p.add_argument('--stage', choices=['qualification','final'], required=True)
            p.add_argument('--legacy-supervised', default='artifacts/models/supervised.npz')
        else:
            p.add_argument('--candidate', required=True)
    p = sub.add_parser('report')
    p.add_argument('--evidence', required=True)
    p.add_argument('--output', default='docs/experiments/gpu-value')
    p = sub.add_parser('stage-release')
    p.add_argument('--candidates', required=True)
    p.add_argument('--legacy-manifest', default='artifacts/models/manifest.json')
    p.add_argument('--output', default='artifacts/staging/gpu-value-v1')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    command = args.command
    end_audit = None
    paid_commands = {'label','train-supervised','pilot','evolve','pilot-check','select-evolution','evaluate','benchmark-components'}
    if command in paid_commands or (command == 'benchmark' and not args.smoke):
        if not args.approval:
            raise SystemExit('A user-approved provider/budget receipt is required before this compute stage')
        stage = 'benchmark' if command in {'benchmark','benchmark-components'} else 'learning'
        receipt = require_execution(config,args.approval,stage)
        end_audit = begin_execution(config,receipt,command)
    def load(path):
        return json.loads(Path(path).read_text())
    if command == 'backup':
        from .backup import create
        roots = {'working-repository':args.root}
        for entry in args.extra_root:
            name,path = entry.split('=',1)
            roots[name] = path
        result = create(roots,args.output)
    elif command in ['verify-backup', 'restore-backup']:
        from .backup import verify
        result = verify(args.archive, args.inventory, getattr(args, 'destination', None))
    elif command == 'doctor':
        import jax
        from importlib.metadata import version
        if sys.version_info[:2] != (3,12):
            raise SystemExit('Use the separately locked Python 3.12 training environment')
        result = dict(python=sys.version, devices=[str(d) for d in jax.devices()],
            packages={p:version(p) for p in ['jax','jaxlib','pgx','mctx','evosax','optax']})
        if args.require_nvidia and not nvidia_available():
            raise SystemExit('NVIDIA backend unavailable; no GPU verification is claimed')
    elif command == 'download':
        from .dataset import download_sources
        result = download_sources(config['supervised'], args.directory)
    elif command == 'prepare-data':
        from .dataset import prepare
        result = prepare(config['supervised'], args.sources, args.directory)
    elif command == 'pin-teacher':
        from .dataset import pin_teacher
        result = pin_teacher(args.binary, [args.eval_file] + ([args.eval_file_small] if args.eval_file_small else []), args.output)
    elif command == 'label':
        require_execution(config,args.approval)
        from .dataset import label
        result = label(config['supervised'],args.directory,args.binary,args.teacher,args.workers)
    elif command == 'features':
        from .dataset import cache_features
        result = cache_features(args.directory,config['supervised']['counts'])
    elif command == 'openings':
        from .openings import prepare
        specification = {**config['supervised'], 'opening_pairs':config['evolution']['opening_pairs']}
        result = prepare(specification,args.sources,Path(args.directory)/'positions.jsonl',Path(args.directory)/'openings.json')
    elif command == 'benchmark':
        if not args.smoke:
            if not args.approval:
                raise SystemExit('Select and authorize a benchmark budget before complete-game benchmarking')
            require_execution(config,args.approval,'benchmark')
        from .benchmark import benchmark
        result = benchmark(config,args.openings,args.output,args.batch_size,args.smoke,args.supervised)
    elif command == 'resolve-workload':
        measured = load(args.benchmark)
        if measured.get('workload_sha256') != identity(config):
            raise SystemExit('Benchmark workload differs from the proposed configuration')
        selected = measured.get('selected_simulations')
        if measured.get('smoke') or not measured.get('measured_on_nvidia') or selected not in [32,64,128]:
            raise SystemExit('A complete qualifying NVIDIA benchmark is required before freezing the workload')
        config['search']['selected_simulations'] = selected
        config['evolution']['simulations'] = selected
        config['evolution']['games_per_candidate'] = args.games_per_candidate
        # Pilot remains the separately frozen 48-game workload in every scenario.
        config['benchmark_sha256'] = sha(args.benchmark)
        if Path(args.output).exists() and load(args.output) != config:
            raise SystemExit('Workload already frozen; create a new prospective study instead')
        atomic_json(args.output,config)
        result = config
    elif command == 'benchmark-components':
        require_execution(config,args.approval,'benchmark')
        from .components import measure
        result = measure(config,args.tournament,args.positions,args.teacher,args.binary,
                         args.openings,args.legacy_supervised,args.output,args.workers)
    elif command == 'forecast':
        from .forecast import forecast
        measured = load(args.measurement)
        if measured.get('workload_sha256') != identity(config):
            raise SystemExit('Forecast measurement belongs to another workload')
        result = forecast(measured,args.hourly_rate,args.storage_gb,args.storage_monthly_rate,config['evolution'])
        atomic_json(args.output,result)
    elif command == 'train-supervised':
        require_execution(config,args.approval)
        from .supervised import fit
        state = fit(config['supervised'],args.data,Path(args.output)/f'seed-{args.seed}',args.seed)
        result = dict(epoch=state['epoch'],best_validation_mse=state['best'],complete=state['complete'])
    elif command in ['pilot','evolve']:
        require_execution(config,args.approval)
        if config['evolution']['simulations'] not in [32,64,128]:
            raise SystemExit('Freeze a qualifying simulation budget before learning')
        from .evolution import evolve
        ec = dict(config['evolution'])
        if command == 'pilot':
            ec['games_per_candidate'] = 48
        result = evolve(ec,args.openings,args.supervised,
            Path(args.output)/'pilot' if command == 'pilot' else Path(args.output)/'evolution'/f'seed-{args.seed}',
            args.seed,command=='pilot',args.batch_size)
    elif command == 'pilot-decision':
        from .pilot import decide
        ec = dict(config['evolution'], games_per_candidate=48)
        result = decide(ec,args.directory,load(args.initial_matches),load(args.last_matches),load(args.coverage),
                        Path(args.directory)/'decision.json')
    elif command == 'select-supervised':
        from .study import select_supervised
        result = select_supervised(args.directory,args.output)
    elif command == 'select-evolution':
        require_execution(config,args.approval)
        from .study import select_evolution
        result = select_evolution(config['evolution'],args.directory,load(args.openings),args.legacy_supervised,args.output,args.workers)
    elif command == 'final-position-test':
        from .supervised import final_test
        result = final_test(config['supervised'],args.data,args.selections,args.output)
    elif command == 'pilot-check':
        require_execution(config,args.approval)
        from .evaluation import cpu_matches
        result = cpu_matches(args.candidate,'handcrafted',load(args.openings)['pilot'],args.output,
            config['evolution']['cpu_match_time_ms'],config['evolution']['simulations'],config['evolution']['max_plies'],
            opponent_strategy='puct-uniform-v1',workers=args.workers)
    elif command == 'evaluate':
        require_execution(config,args.approval)
        from .study import evaluate
        result = evaluate(config['evolution'],args.selections,args.openings,args.legacy_supervised,args.stage,args.output,args.workers)
    elif command == 'report':
        from .report import publish
        result = publish(config,load(args.evidence),args.output)
    elif command == 'stage-release':
        from .release import stage
        result = stage(args.legacy_manifest,load(args.candidates),args.output)
    print(json.dumps(result,indent=2,default=str))
    if end_audit:
        end_audit('completed')


if __name__ == '__main__':
    main()
