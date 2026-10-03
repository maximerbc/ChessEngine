from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def provenance() -> dict:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "uncommitted", True
    return {
        "commit": commit,
        "dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
