"""Build a private allowlisted training bundle; never uploads, provisions or trains."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile
from chessengine.common import digest
from chessengine.elite_data import atomic_json, preflight


def package(dataset, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    repository = Path(__file__).resolve().parents[1]
    meta = preflight(dataset)
    allowed = [Path('pyproject.toml'), Path('requirements-elite-policy.txt'),
               Path('scripts/run_elite_policy.py'), Path('scripts/runpod_policy_guard.py'),
               Path('docs/elite-policy.md')]
    allowed += [p.relative_to(repository) for p in sorted((repository / 'src/chessengine').glob('*.py'))]
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repository, timeout=10).decode().strip()
    except subprocess.SubprocessError:
        commit = 'uncommitted'
    stamp = dict(commit=commit, files={str(p): digest(repository / p) for p in allowed},
                 dataset_sha256=digest(dataset / 'dataset.json'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output.with_suffix('.tmp'), 'w:gz') as archive:
        for path in allowed:
            archive.add(repository / path, arcname=str(path), recursive=False)
        for name, value in [('source-revision.json', stamp), ('configs/elite-policy.json', meta['config'])]:
            content = json.dumps(value, indent=2).encode()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        archive.add(dataset / 'dataset.json', arcname='runs/elite-policy/dataset.json')
        for shard in meta['shards']:
            for field in ['path', 'records']:
                path = (dataset / shard[field]).resolve()
                if not path.is_relative_to(dataset):
                    raise ValueError('Unsafe shard path')
                archive.add(path, arcname='runs/elite-policy/' + str(path.relative_to(dataset)), recursive=False)
    output.with_suffix('.tmp').replace(output)
    receipt = dict(path=str(output), sha256=digest(output), bytes=output.stat().st_size, **stamp)
    atomic_json(output.with_suffix('.manifest.json'), receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.dataset, args.output), indent=2))
