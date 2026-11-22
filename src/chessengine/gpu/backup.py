"""Verify and restore a private preservation archive without trusting archive paths."""
import hashlib
import json
from pathlib import Path
import tarfile
import io
import subprocess
import time


def create(roots, directory):
    """Archive current private experiment inputs plus tracked/untracked local edits."""
    from .protocol import atomic_json, sha
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / 'results.tar.gz'
    if archive.exists():
        raise ValueError('Use a fresh preservation directory; never overwrite a prior archive')
    rows = []
    temporary = directory / 'results.partial'
    with tarfile.open(temporary, 'w:gz', compresslevel=3) as tar:
        for name, raw_root in roots.items():
            root = Path(raw_root).resolve()
            if not name or '/' in name or name in {'.','..'}:
                raise ValueError('Invalid archive namespace')
            paths = {p for folder in ['data','runs','configs','artifacts','docs']
                     for p in (root / folder).rglob('*') if p.is_file() and directory not in p.resolve().parents}
            paths.update(root / f for f in ['AGENTS.md','README.md','pyproject.toml'] if (root / f).is_file())
            untracked = subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=root)
            paths.update(root / p for p in untracked.decode().split('\0') if p and (root/p).is_file()
                         and directory not in (root/p).resolve().parents)
            patch = subprocess.check_output(['git','diff','--binary','HEAD'],cwd=root)
            payloads = [('uncommitted.patch',patch)]
            for path in sorted(paths):
                before = path.stat()
                data = path.read_bytes()
                after = path.stat()
                if len(data) != after.st_size or (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                    raise ValueError(f'File unavailable or changed during preservation: {path}')
                payloads.append((str(path.relative_to(root)),data))
                # Keep memory bounded to one file, including large datasets.
                relative,content = payloads.pop()
                member = tarfile.TarInfo(name+'/'+relative)
                member.size = len(content)
                tar.addfile(member,io.BytesIO(content))
                rows.append(dict(path=member.name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
            for relative,data in payloads:
                member = tarfile.TarInfo(name+'/'+relative)
                member.size = len(data)
                tar.addfile(member,io.BytesIO(data))
                rows.append(dict(path=member.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    temporary.replace(archive)
    inventory = directory / 'inventory.json'
    record = dict(schema=1,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        archive=str(archive),archive_sha256=sha(archive),files=rows,restore_verified=False)
    atomic_json(inventory,record)
    verify(archive,inventory)
    record['restore_verified'] = True
    record['verification'] = 'Every compressed member reread and SHA-256 matched'
    atomic_json(inventory,record)
    return dict(archive=str(archive),inventory=str(inventory),files=len(rows),verified=True)


def verify(archive, inventory, destination=None):
    from .protocol import sha
    manifest = json.loads(Path(inventory).read_text())
    if sha(archive) != manifest['archive_sha256']:
        raise ValueError('Archive checksum mismatch')
    expected = {r['path']: r for r in manifest['files']}
    root = Path(destination).resolve() if destination else None
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            path = Path(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or member.name not in expected):
                raise ValueError('Unsafe or unexpected archive member')
            row = expected.pop(member.name)
            data = tar.extractfile(member).read()
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('Preserved member checksum mismatch')
            if root:
                target = root / path
                if target.exists() or not target.resolve().is_relative_to(root):
                    raise ValueError('Restoration requires unused paths inside the destination')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    if expected:
        raise ValueError('Missing preserved members')
    return {'verified_files': len(manifest['files']), 'restored': bool(root)}
