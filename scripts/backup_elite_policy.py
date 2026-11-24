"""Snapshot private run outputs, then verify every archived byte before extraction."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root, destination, extras=(), allow_incomplete=False):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    complete = (root / 'job-complete.json').exists()
    if not complete and not allow_incomplete:
        raise ValueError('Run is not complete; do not snapshot mutable checkpoints')
    files = {}
    for path in sorted(root.glob('*.json')):
        if path.name != 'backup-verified.json':
            files[path.name] = path
    for directory in sorted(root.glob('seed-*')):
        for path in sorted(directory.iterdir()):
            if path.suffix in ['.json', '.npz', '.pt'] and not path.is_symlink():
                files[str(path.relative_to(root))] = path
    for item in extras:
        name, path = item.split('=', 1)
        if name in files:
            raise ValueError('Duplicate archive path')
        files[name] = Path(path)
    manifest = dict(schema='elite-private-backup-v1', created_unix=time.time(), complete=complete,
        files={name: dict(sha256=sha(path), bytes=path.stat().st_size) for name, path in files.items()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for name, path in files.items():
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                raise ValueError('Unsafe archive path')
            archive.add(path, arcname=name, recursive=False)
        content = json.dumps(manifest, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo('backup-manifest.json')
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    temporary.replace(destination)
    receipt = dict(sha256=sha(destination), bytes=destination.stat().st_size,
                   members=len(files), complete=complete)
    destination.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def verify(path, expected_sha256, extract=None):
    path = Path(path)
    if sha(path) != expected_sha256:
        raise ValueError('Backup archive checksum mismatch')
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or 'backup-manifest.json' not in names:
            raise ValueError('Missing manifest or duplicate archive entries')
        manifest = json.load(archive.extractfile('backup-manifest.json'))
        if manifest.get('schema') != 'elite-private-backup-v1':
            raise ValueError('Unknown backup schema')
        if set(names) != {'backup-manifest.json', *manifest['files']}:
            raise ValueError('Backup member set differs from manifest')
        for member in members:
            parts = PurePosixPath(member.name)
            if not member.isfile() or parts.is_absolute() or '..' in parts.parts:
                raise ValueError('Unsafe archive member')
            if member.name == 'backup-manifest.json':
                continue
            digest = hashlib.sha256()
            size = 0
            with archive.extractfile(member) as source:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    size += len(block)
            expected = manifest['files'][member.name]
            if digest.hexdigest() != expected['sha256'] or size != expected['bytes']:
                raise ValueError('Backup member checksum mismatch: '+member.name)
        if extract:
            destination = Path(extract).resolve()
            destination.mkdir(parents=True, exist_ok=True)
            # All entries and hashes were checked before the first write.
            for member in members:
                target = destination / member.name
                if not target.resolve().is_relative_to(destination):
                    raise ValueError('Extraction escapes destination')
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
    return dict(verified=True, sha256=expected_sha256, members=len(manifest['files']),
                complete=manifest['complete'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('snapshot')
    build.add_argument('--root', required=True)
    build.add_argument('--output', required=True)
    build.add_argument('--extra', action='append', default=[])
    build.add_argument('--allow-incomplete', action='store_true')
    check = sub.add_parser('verify')
    check.add_argument('archive')
    check.add_argument('--sha256', required=True)
    check.add_argument('--extract')
    args = parser.parse_args()
    result = (snapshot(args.root, args.output, args.extra, args.allow_incomplete) if args.action == 'snapshot'
              else verify(args.archive, args.sha256, args.extract))
    print(json.dumps(result, indent=2))
