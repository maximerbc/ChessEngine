"""A verified backup is required before the only remote training copy is removed."""
import hashlib
import io
import json
from pathlib import Path
import runpy
import tarfile

import pytest


BACKUP = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/backup_elite_policy.py'))


def test_backup_requires_completed_run_and_verifies_every_member(tmp_path):
    root = tmp_path / 'run'
    (root / 'seed-42').mkdir(parents=True)
    (root / 'seed-42' / 'resume.pt').write_bytes(b'private optimizer and RNG state')
    archive = tmp_path / 'private.tar.gz'
    with pytest.raises(ValueError, match='not complete'):
        BACKUP['snapshot'](root, archive)
    (root / 'job-complete.json').write_text('{}')
    receipt = BACKUP['snapshot'](root, archive)
    extracted = tmp_path / 'restored'
    assert BACKUP['verify'](archive, receipt['sha256'], extracted)['verified']
    assert (extracted / 'seed-42/resume.pt').read_bytes() == (root / 'seed-42/resume.pt').read_bytes()
    with pytest.raises(ValueError, match='archive checksum'):
        BACKUP['verify'](archive, '0' * 64)
    tampered = tmp_path / 'tampered.tar.gz'
    with tarfile.open(archive) as source, tarfile.open(tampered, 'w:gz') as target:
        for member in source:
            data = source.extractfile(member).read()
            if member.name == 'seed-42/resume.pt':
                data = b'corrupted'
            info = tarfile.TarInfo(member.name)
            info.size = len(data)
            target.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match='member checksum'):
        BACKUP['verify'](tampered, hashlib.sha256(tampered.read_bytes()).hexdigest(), tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
    assert json.loads((extracted / 'backup-manifest.json').read_text())['complete']
