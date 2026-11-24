"""Independent complete split/position/label audit, with sampled full-history replay."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time
import chess
import numpy as np
from chessengine.common import digest
from chessengine.elite_data import atomic_json, preflight
from chessengine.elite_policy import encode_board, MOVE_IDS
from chessengine.features import restore
from chessengine.data import split_game


def audit(root, history_stride=997):
    root = Path(root)
    meta = preflight(root)
    dbpath = root / 'audit.sqlite'
    if dbpath.exists():
        dbpath.unlink()
    db = sqlite3.connect(dbpath)
    db.execute('CREATE TABLE positions(key TEXT PRIMARY KEY) WITHOUT ROWID')
    db.execute('CREATE TABLE games(key TEXT PRIMARY KEY, split TEXT NOT NULL) WITHOUT ROWID')
    db.execute('ATTACH DATABASE ? AS excluded', (str(root / 'exclusions.sqlite'),))
    counts, histories, move_counts = Counter(), 0, Counter()
    start = time.perf_counter()
    index = 0
    for shard in meta['shards']:
        with np.load(root / shard['path'], allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        with (root / shard['records']).open() as raw:
            rows = 0
            for i, line in enumerate(raw):
                row = json.loads(line)
                split, gid = row['split'], row['game_id']
                if split != shard['split'] or split != split_game(gid) or str(data['games'][i]) != gid:
                    raise ValueError('Whole-game split/encoded identity mismatch')
                owned = db.execute('SELECT split FROM games WHERE key=?', (gid,)).fetchone()
                if owned and owned[0] != split:
                    raise ValueError('Source-game split leakage')
                db.execute('INSERT OR IGNORE INTO games VALUES (?,?)', (gid, split))
                key = ' '.join(row['fen'].split()[:4])
                if db.execute('SELECT 1 FROM excluded.positions WHERE key=?', (key,)).fetchone():
                    raise ValueError('Prior or reserved position leaked')
                if db.execute('SELECT 1 FROM excluded.games WHERE key=?', (gid,)).fetchone():
                    raise ValueError('Prior or reserved source game leaked')
                db.execute('INSERT INTO positions VALUES (?)', (key,))  # Reject duplicates across and within splits.
                y = MOVE_IDS[row['target_move']]
                if int(data['y'][i]) != y or y not in data['legal'][data['pointers'][i]:data['pointers'][i+1]]:
                    raise ValueError('Played label is not aligned to encoded legal moves')
                if index % history_stride == 0:
                    board = restore(row)
                    if board.is_game_over(claim_draw=True) or chess.Move.from_uci(row['target_move']) not in board.legal_moves:
                        raise ValueError('Illegal or terminal sampled history')
                    encoded = np.packbits(encode_board(board).reshape(-1).astype(np.uint8))
                    np.testing.assert_array_equal(encoded, data['x'][i])
                    histories += 1
                counts[split] += 1
                move_counts[row['target_move']] += 1
                rows, index = rows + 1, index + 1
            if rows != shard['count']:
                raise ValueError('Raw/encoded count mismatch')
        db.commit()
        print(f'Audited {index} positions', flush=True)
    if dict(counts) != meta['counts']:
        raise ValueError('Audit quotas differ')
    game_counts = dict(db.execute('SELECT split,count(*) FROM games GROUP BY split'))
    db.close()
    result = dict(passed=True, dataset_sha256=digest(root / 'dataset.json'), positions=dict(counts),
        source_games=game_counts, unique_positions=index, unique_played_moves=len(move_counts),
        full_history_replays=histories, history_stride=history_stride,
        complete_checks=['shard hashes','raw/encoded labels','game ownership','position deduplication',
                         'old source games','prior/reserved positions'],
        sampled_checks=['full history reconstruction','automatic draws','13-plane encoding','legality'],
        seconds=time.perf_counter() - start)
    atomic_json(root / 'audit.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.root), indent=2))
