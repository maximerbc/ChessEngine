"""Copy a frozen position corpus, quarantine canonical leakage, refill exact quotas."""
from pathlib import Path
import time
import chess
from .common import digest, read_json, write_json, provenance
from .data import records
from .move_policy_data import canonical, exclusion_rows, input_identity, SPLITS


def prepare_reused(config, output, input_root):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    base = Path(input_root)
    identity = {'config': config, 'inputs': input_identity(config, input_root)}
    source = base / config['source']
    if identity['inputs'][config['source']] != config['reuse_source_sha256']:
        raise ValueError('Frozen million-position source checksum changed')
    path = root / 'positions.manifest.json'
    if path.exists():
        old = read_json(path)
        if old['identity'] != identity or old['sha256'] != digest(root / 'positions.jsonl'):
            raise ValueError('Reused policy preparation changed')
        return old
    targets = config['positions']
    reserved_games, reserved_positions = set(), set()
    for name in config['exclusions']:
        for row in exclusion_rows(base / name):
            reserved_games.update(row[k] for k in ['game_id', 'source_game_id'] if row.get(k))
            reserved_positions.add(canonical(chess.Board(row['fen'])))
    games, owners, groups, cross_keys = {}, {}, {}, set()
    bad = set(reserved_games)
    original_counts = dict.fromkeys(SPLITS, 0)
    started = time.perf_counter()
    for number, row in enumerate(records(source), 1):
        split, gid = row['split'], row['game_id']
        if split not in targets:
            raise ValueError('Unknown reused split')
        original_counts[split] += 1
        previous = games.setdefault(gid, (split, gid))
        if previous[0] != split:
            raise ValueError('Reused source-game split conflict')
        gid = previous[1]
        key = canonical(chess.Board(row['fen']))
        if key in reserved_positions:
            bad.add(gid)
        if key in owners:
            old_split, old_gid = owners[key]
            involved = groups.setdefault(key, {old_gid})
            involved.add(gid)
            if old_split != split or key in cross_keys:
                cross_keys.add(key)
                bad.update(involved)
        else:
            owners[key] = (split, gid)
        if number % 100000 == 0:
            write_json(root / 'progress.json', {'stage': 'reuse-audit', 'positions': number,
                       'seconds': time.perf_counter() - started})
            print(f'Audited {number:,} reused positions', flush=True)
    if original_counts != targets:
        raise ValueError('Reused source does not have the exact approved counts')
    counts = dict.fromkeys(SPLITS, 0)
    inherited = dict.fromkeys(SPLITS, 0)
    refill_counts = dict.fromkeys(SPLITS, 0)
    selected = set()
    refill_games = {}
    fields = ('initial_fen', 'moves', 'fen', 'game_id', 'source', 'split')
    import json
    temp = root / 'positions.jsonl.tmp'
    with temp.open('w') as stream:
        for row in records(source):
            split, gid = row['split'], row['game_id']
            if gid in bad or counts[split] >= targets[split]:
                continue
            key = canonical(chess.Board(row['fen']))
            if key in selected:
                continue
            selected.add(key)
            stream.write(json.dumps({k: row[k] for k in fields if k in row}) + '\n')
            counts[split] += 1
            inherited[split] += 1
        for row in records(base / config['refill_source']):
            split, gid = row['split'], row['game_id']
            if split not in targets or gid in games or gid in reserved_games or counts[split] >= targets[split]:
                continue
            if gid in refill_games and refill_games[gid] != split:
                raise ValueError('Refill source-game split conflict')
            refill_games[gid] = split
            key = canonical(chess.Board(row['fen']))
            if key in owners or key in selected or key in reserved_positions:
                continue
            selected.add(key)
            stream.write(json.dumps({k: row[k] for k in fields if k in row}) + '\n')
            counts[split] += 1
            refill_counts[split] += 1
            if counts == targets:
                break
    if counts != targets:
        raise ValueError(f'Insufficient audited quota refill: {counts}')
    temp.replace(root / 'positions.jsonl')
    audit = {'original_counts': original_counts, 'inherited_counts': inherited,
             'refill_counts': refill_counts, 'quarantined_source_games': sorted(bad.intersection(games)),
             'canonical_cross_split_keys': len(cross_keys), 'original_unchanged': digest(source) == config['reuse_source_sha256']}
    write_json(root / 'reuse-audit.json', audit)
    result = {'identity': identity, 'counts': counts, 'sha256': digest(root / 'positions.jsonl'),
              'preparation_mode': 'reused-corpus-quarantine-refill', 'audit': audit,
              'seconds': time.perf_counter() - started, 'provenance': provenance()}
    write_json(path, result)
    return result
