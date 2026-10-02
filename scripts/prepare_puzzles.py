"""Take a bounded public puzzle sample, excluding every supervised position."""

import argparse
import csv
import io
import ssl
import urllib.request
from pathlib import Path
import certifi
import chess
import zstandard
from chessengine.common import write_json, digest
from chessengine.data import records
from chessengine.features import position_key

p = argparse.ArgumentParser()
p.add_argument("dataset")
p.add_argument("output")
p.add_argument("--limit", type=int, default=50)
a = p.parse_args()
excluded = {" ".join(r["fen"].split()[:4]) for r in records(a.dataset)}
url = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
count = 0
scanned = 0
with urllib.request.urlopen(
    url, timeout=60, context=ssl.create_default_context(cafile=certifi.where())
) as raw:
    with zstandard.ZstdDecompressor().stream_reader(raw) as decompressed, open(a.output, "w") as out:
        reader = csv.DictReader(io.TextIOWrapper(decompressed))
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            scanned += 1
            if scanned > 10000:
                break
            if int(row["Rating"]) > 1600:
                continue
            board = chess.Board(row["FEN"])
            board.push_uci(row["Moves"].split()[0])
            key = position_key(board)
            if key in excluded:
                continue
            excluded.add(key)
            writer.writerow(row)
            count += 1
            if count >= a.limit:
                break
write_json(
    a.output + ".manifest.json",
    {
        "source": url,
        "positions": count,
        "rows_scanned": scanned,
        "rating_ceiling": 1600,
        "sha256": digest(a.output),
        "excluded_dataset_sha256": digest(a.dataset),
        "limitation": "Exact-position disjointness only; common tactical motifs may overlap.",
    },
)
print("Prepared", count, "held-out diagnostic puzzles")
