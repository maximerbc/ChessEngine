"""Exact, whole-game-separated corpus. Historical labels are never reused."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import shutil
import urllib.request
import ssl
import certifi
import csv
from urllib.parse import urlsplit

import chess
import chess.engine
import chess.pgn
from ..data import open_pgn, records, split_game
from ..features import position_key, record_board, restore
from ..common import json_hash
from ..expand_data import label_chunk
from .protocol import atomic_json, freeze, sha


def download_sources(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for url in config['sources']:
        path = directory / url.rsplit('/', 1)[-1]
        if not path.exists():
            tmp = path.with_suffix('.partial')
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(url, timeout=60, context=context) as source, tmp.open('wb') as target:
                shutil.copyfileobj(source, target)
            tmp.replace(path)
        rows.append({'url': url, 'file': str(path), 'sha256': sha(path)})
    atomic_json(directory / 'sources.json', rows)
    return rows


def site_id(url):
    parsed = urlsplit(url)
    if parsed.hostname == 'lichess.org' or (parsed.hostname or '').endswith('.lichess.org'):
        return parsed.path.strip('/').split('/')[0][:8] or None
    return None


def puzzle_exclusions(paths):
    positions, sites = set(), set()
    for path in paths:
        with open(path,newline='') as stream:
            for row in csv.DictReader(stream):
                board = chess.Board(row['FEN'])
                positions.add(position_key(board))
                for move in row['Moves'].split():
                    board.push_uci(move)
                    positions.add(position_key(board))
                identifier = site_id(row.get('GameUrl',''))
                if identifier:
                    sites.add(identifier)
    return positions, sites


def research_inputs(config):
    paths = set()
    for root in config.get('historical_research_roots', []):
        root = Path(root)
        if not root.exists():
            raise ValueError(f'Restore required historical research input before preparing fresh pools: {root}')
        candidates = [root] if root.is_file() else root.rglob('*')
        paths.update(str(p) for p in candidates if p.is_file() and p.suffix in {'.json','.jsonl'})
    return sorted(paths)


def research_exclusions(paths):
    positions, games = set(), set()
    def visit(value):
        if isinstance(value, dict):
            for name in ['game_id', 'source_game_id']:
                if isinstance(value.get(name), str):
                    games.add(value[name])
            if isinstance(value.get('fen'), str):
                positions.add(position_key(chess.Board(value['fen'])))
            moves = value.get('moves')
            if ('initial_fen' in value and isinstance(moves, list)
                    and all(isinstance(move, str) for move in moves)):
                board = chess.Board(value['initial_fen'])
                positions.add(position_key(board))
                for move in moves:
                    board.push_uci(move)
                    positions.add(position_key(board))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for path in paths:
        if Path(path).suffix == '.jsonl':
            for row in records(path):
                visit(row)
        else:
            visit(json.loads(Path(path).read_text()))
    return positions, games


def exclusions(paths, reserved_paths, research_paths=()):
    positions, games = set(), set()
    for path in paths:
        for row in records(path):
            positions.add(position_key(chess.Board(row['fen'])))
            games.add(row['game_id'])
    reserved_positions, reserved_games = set(), set()
    for path in reserved_paths:
        groups = json.loads(Path(path).read_text())
        for rows in groups.values():
            for row in rows:
                reserved_positions.add(position_key(restore(row)))
                if row.get('source_game_id'):
                    reserved_games.add(row['source_game_id'])
    research_positions, research_games = research_exclusions(research_paths)
    # Old training records may remain training; historical research only enlarges
    # the exclusion set for newly sampled positions and all fresh held-out pools.
    return positions | research_positions, games | research_games, reserved_positions, reserved_games


def prepare(config, source_manifest, output):
    output = Path(output)
    sources = json.loads(Path(source_manifest).read_text())
    historical_research = research_inputs(config)
    inputs = [source_manifest, config['retain_training'], *config['historical_datasets'],
              *config['reserved'], *historical_research, *config.get('reserved_puzzle_csv',[])]
    for row in sources:
        if sha(row['file']) != row['sha256']:
            raise ValueError('Downloaded source checksum changed')
    protocol = freeze(output, config, inputs)
    destination = output / 'positions.jsonl'
    if (output / 'preparation.json').exists():
        result = json.loads((output / 'preparation.json').read_text())
        if sha(destination) != result['sha256']:
            raise ValueError('Prepared corpus changed')
        return result
    historical, historical_games, reserved, reserved_games = exclusions(
        config['historical_datasets'], config['reserved'], historical_research)
    puzzle_positions,puzzle_sites = puzzle_exclusions(config.get('reserved_puzzle_csv',[]))
    reserved.update(puzzle_positions)
    seen, used_games = set(reserved), set()
    counts = dict.fromkeys(config['counts'], 0)
    quotas = config['counts']
    temp = destination.with_suffix('.tmp')
    with temp.open('w') as stream:
        def accept(row):
            split = row['split']
            key = position_key(chess.Board(row['fen']))
            if counts[split] >= quotas[split] or key in seen or row['game_id'] in reserved_games:
                return False
            if split != 'train' and (key in historical or row['game_id'] in historical_games):
                return False
            seen.add(key)
            used_games.add(row['game_id'])
            counts[split] += 1
            # Do not retain any previous teacher's target or prediction.
            row = {k: row[k] for k in ('initial_fen', 'moves', 'fen', 'game_id', 'source', 'split')}
            stream.write(json.dumps(row) + '\n')
            return True

        for row in records(config['retain_training']):
            if row['split'] == 'train' and split_game(row['game_id']) == 'train':
                accept(row)
        retained = counts['train']
        # Previously used held-out positions cannot become new training examples.
        seen.update(historical)
        for source in sources:
            with open_pgn(source['file']) as games:
                while counts != quotas:
                    game = chess.pgn.read_game(games)
                    if game is None:
                        break
                    if game.errors or game.headers.get('Variant', 'Standard') != 'Standard':
                        continue
                    if site_id(game.headers.get('Site','')) in puzzle_sites:
                        continue
                    board = game.board()
                    if board.chess960 or not board.is_valid():
                        continue
                    moves = list(game.mainline_moves())
                    game_id = json_hash({'initial': board.fen(), 'moves': [m.uci() for m in moves]})
                    split = split_game(game_id)
                    if (game_id in used_games or game_id in historical_games or game_id in reserved_games
                            or counts[split] >= quotas[split]):
                        continue
                    rng = random.Random(json_hash([config['sampling_seed'], game_id]))
                    selected = set()
                    for lo, hi in [(8, 30), (30, 70), (70, 1000)]:
                        phase = list(range(lo, min(hi, len(moves))))
                        selected.update(rng.sample(phase, min(2, len(phase))))
                    for i, move in enumerate(moves):
                        board.push(move)
                        if i in selected and not board.is_game_over(claim_draw=True):
                            accept(dict(record_board(board), game_id=game_id, split=split, source=source['url']))
                    used_games.add(game_id)
    if counts != quotas:
        temp.unlink()
        raise ValueError(f'Sources exhausted before exact split quotas: {counts}; do not alter the protocol silently')
    temp.replace(destination)
    audit_positions(destination, quotas)
    result = dict(counts=counts, total=sum(counts.values()), retained_training=retained,
                  sha256=sha(destination), protocol_id=protocol, sources=sources)
    atomic_json(output / 'preparation.json', result)
    return result


def audit_positions(path, quotas):
    games, seen = {}, set()
    counts = dict.fromkeys(quotas, 0)
    for row in records(path):
        split = row['split']
        if split not in quotas:
            raise ValueError('Unknown split')
        key = position_key(chess.Board(row['fen']))
        if key in seen or games.get(row['game_id'], split) != split:
            raise ValueError('Canonical duplicate or source-game leakage')
        seen.add(key)
        games[row['game_id']] = split
        counts[split] += 1
    if counts != quotas:
        raise ValueError('Exact position counts required')
    return counts


def pin_teacher(binary, nnue_files, output):
    """Explicit NNUE identities; embedded-default inference is not guessed."""
    if len(nnue_files) != 2:
        raise ValueError('Pin both explicit Stockfish EvalFile and EvalFileSmall NNUE files')
    with chess.engine.SimpleEngine.popen_uci(binary) as engine:
        if not engine.id.get('name','').startswith('Stockfish '):
            raise ValueError('The pinned teacher must identify itself as Stockfish')
        teacher = {'binary_sha256': sha(binary), 'teacher': engine.id, 'nodes': 10000,
                   'threads': 1, 'hash_mb': 32, 'nnue': {str(Path(p).resolve()): sha(p) for p in nnue_files},
                   'eval_files': dict(zip(['EvalFile', 'EvalFileSmall'], [str(Path(p).resolve()) for p in nnue_files]))}
        engine.configure(teacher['eval_files'])
        for option in ('EvalFile', 'EvalFileSmall'):
            if option in engine.options:
                teacher[option] = engine.options[option].default
    atomic_json(output, teacher)
    return teacher


def label(config, directory, binary, teacher_path, workers):
    directory = Path(directory)
    teacher = json.loads(Path(teacher_path).read_text())
    if sha(binary) != teacher['binary_sha256'] or any(sha(p) != h for p, h in teacher['nnue'].items()):
        raise ValueError('Pinned teacher or NNUE changed')
    if workers < 1:
        raise ValueError('Label workers must be positive')
    freeze(directory / 'labeling', {'teacher': teacher, 'chunk_size': config['label_chunk_size']},
           [directory / 'positions.jsonl', teacher_path])
    chunks = directory / 'labeling/chunks'
    chunks.mkdir(parents=True, exist_ok=True)
    pending, completed = [], []
    # Bound queued data to workers; one million histories need not fit in RAM.
    with ThreadPoolExecutor(workers) as pool:
        batch, offset = [], 0
        for row in records(directory / 'positions.jsonl'):
            batch.append(row)
            if len(batch) == config['label_chunk_size']:
                pending.append((offset, pool.submit(label_chunk, batch, chunks / f'{offset:07}.jsonl', binary, teacher)))
                offset += len(batch)
                batch = []
                if len(pending) >= workers:
                    index, future = pending.pop(0)
                    completed.append((index, future.result()))
        if batch:
            pending.append((offset, pool.submit(label_chunk, batch, chunks / f'{offset:07}.jsonl', binary, teacher)))
        completed.extend((i, future.result()) for i, future in pending)
    temp = directory / 'labeled.tmp'
    with temp.open('wb') as stream:
        for offset, _ in sorted(completed):
            with (chunks / f'{offset:07}.jsonl').open('rb') as chunk:
                shutil.copyfileobj(chunk, stream)
    temp.replace(directory / 'labeled.jsonl')
    result = dict(sha256=sha(directory / 'labeled.jsonl'), teacher=teacher,
                  label_worker_seconds=sum(r['label_seconds'] for _, r in completed))
    atomic_json(directory / 'labels.json', result)
    return result


def cache_features(directory, counts):
    import numpy as np
    from ..features import encode
    directory = Path(directory)
    source = directory / 'labeled.jsonl'
    metadata = directory / 'features.json'
    if metadata.exists():
        record = json.loads(metadata.read_text())
        if record['source_sha256'] != sha(source) or record['counts'] != counts:
            raise ValueError('Feature cache source changed')
        for name, checksum in record['files'].items():
            if sha(directory / name) != checksum:
                raise ValueError('Feature cache checksum mismatch')
        return record
    arrays = {s: (np.lib.format.open_memmap(directory / f'{s}-x.npy', mode='w+', dtype='float32', shape=(n, 853)),
                  np.lib.format.open_memmap(directory / f'{s}-y.npy', mode='w+', dtype='float32', shape=(n,)))
              for s, n in counts.items()}
    offsets = dict.fromkeys(counts, 0)
    for row in records(source):
        split = row['split']
        x, y = arrays[split]
        i = offsets[split]
        if i >= counts[split] or not np.isfinite(row['target']) or not -1 <= row['target'] <= 1:
            raise ValueError('Unexpected split count or target')
        x[i], y[i] = encode(restore(row), 3), row['target']
        offsets[split] += 1
    if offsets != counts:
        raise ValueError('Exact split counts required')
    for x, y in arrays.values():
        x.flush()
        y.flush()
    result = dict(source_sha256=sha(source), counts=counts,
                  files={p.name: sha(p) for p in directory.glob('*-?.npy')})
    atomic_json(metadata, result)
    return result
