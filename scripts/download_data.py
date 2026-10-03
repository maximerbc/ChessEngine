"""Bounded source download with integrity recording. No archive extraction."""

import argparse
import ssl
import urllib.request
from pathlib import Path
import certifi
from chessengine.common import digest, write_json

p = argparse.ArgumentParser()
p.add_argument("--output", default="data/lichess-2013-01.pgn.zst")
a = p.parse_args()
url = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
path = Path(a.output)
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    raise SystemExit("File exists; verify its manifest instead of overwriting it")
size = 0
with (
    urllib.request.urlopen(
        url, timeout=60, context=ssl.create_default_context(cafile=certifi.where())
    ) as response,
    open(path.with_suffix(".part"), "wb") as out,
):
    while chunk := response.read(1024 * 1024):
        size += len(chunk)
        if size > 64 * 1024 * 1024:
            raise ValueError("Download exceeded 64 MiB safety limit")
        out.write(chunk)
path.with_suffix(".part").replace(path)
write_json(str(path) + ".source.json", {"url": url, "sha256": digest(path), "bytes": size, "license": "CC0"})
