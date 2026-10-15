"""CPU training of a masked legal-move scorer with exact epoch resume."""

import argparse
import time
from pathlib import Path
import numpy as np
import torch
from .common import read_json, write_json, digest, json_hash, provenance
from .features import encode, restore
from .ranking import move_features, build_ranker, export_ranker
from .train import seed_all, save_state, trusted_load, rng_state, restore_rng


def dataset(config):
    root = Path(config["output"])
    progress = read_json(root / "label-progress.json")
    if not progress["complete"]:
        raise ValueError("Move labeling is incomplete")
    files = sorted((root / "labels").glob("*.json"))
    files = [p for p in files if ".manifest." not in p.name]
    identity = {str(p): digest(p) for p in files}
    for p in files:
        if read_json(p.with_suffix(".manifest.json"))["sha256"] != identity[str(p)]:
            raise ValueError("Corrupt move labels")
    signature = json_hash(identity)
    path = root / "encoded.npz"
    meta = root / "encoded.manifest.json"
    if path.exists() and meta.exists():
        old = read_json(meta)
        if old["source"] != signature or old["sha256"] != digest(path):
            raise ValueError("Ranking cache mismatch")
    else:
        rows = [r for p in files for r in read_json(p)]
        arrays = {}
        for split in ["train", "validation"]:
            xs = []
            ms = []
            targets = []
            offsets = [0]
            for row in rows:
                if row["split"] != split:
                    continue
                board = restore(row)
                legal = list(board.legal_moves)
                ucis = [m.uci() for m in legal]
                if row["teacher_move"] not in ucis:
                    raise ValueError("Illegal move training target")
                xs.append(encode(board, 3))
                ms.append(move_features(board, legal))
                targets.append(ucis.index(row["teacher_move"]))
                offsets.append(offsets[-1] + len(legal))
            if not xs:
                raise ValueError("Missing ranker split")
            arrays.update(
                {
                    split + "_x": np.stack(xs),
                    split + "_moves": np.concatenate(ms),
                    split + "_target": np.array(targets, dtype=np.int64),
                    split + "_offsets": np.array(offsets, dtype=np.int64),
                }
            )
        np.savez_compressed(path, **arrays)
        write_json(meta, dict(source=signature, sha256=digest(path)))
    with np.load(path) as saved:
        data = {
            s: {
                key: torch.from_numpy(saved[s + "_" + key].copy())
                for key in ["x", "moves", "target", "offsets"]
            }
            for s in ["train", "validation"]
        }
    return data, digest(path)


def batch(data, ids):
    offsets = data["offsets"]
    lengths = offsets[ids + 1] - offsets[ids]
    size = int(lengths.max())
    moves = torch.zeros((len(ids), size, 3), dtype=torch.long)
    mask = torch.zeros((len(ids), size), dtype=torch.bool)
    for j, i in enumerate(ids.tolist()):
        start, end = int(offsets[i]), int(offsets[i + 1])
        moves[j, : end - start] = data["moves"][start:end]
        mask[j, : end - start] = True
    return data["x"][ids], moves, mask, data["target"][ids]


def evaluate(model, data, batch_size=256):
    model.eval()
    loss = top1 = top3 = ranks = 0.0
    n = len(data["x"])
    with torch.no_grad():
        for ids in torch.arange(n).split(batch_size):
            x, m, mask, y = batch(data, ids)
            logits = model(x, m, mask)
            loss += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
            ranking = logits.argsort(dim=1, descending=True, stable=True)
            rank = (ranking == y[:, None]).nonzero()[:, 1] + 1
            top1 += int((rank == 1).sum())
            top3 += int((rank <= 3).sum())
            ranks += int(rank.sum())
    return dict(cross_entropy=loss / n, top1=top1 / n, top3=top3 / n, mean_rank=ranks / n, positions=n)


def train(config, seed, data, data_sha, stop_after=None):
    settings = config["training"]
    root = Path(config["output"]) / f"seed-{seed}"
    root.mkdir(parents=True, exist_ok=True)
    signature = json_hash(
        dict(
            config=config,
            seed=seed,
            data=data_sha,
            baseline=digest(config["baseline"]),
            code={
                n: digest(Path(__file__).with_name(n + ".py"))
                for n in ["ranking", "ranking_train", "features"]
            },
        )
    )
    seed_all(seed)
    torch.set_num_threads(1)
    model = build_ranker(config["baseline"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    history = []
    best = float("inf")
    stale = 0
    start = 0
    state_path = root / "resume.pt"
    if state_path.exists():
        state = trusted_load(state_path)
        if state["signature"] != signature:
            raise ValueError("Ranking resume mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        history = state["history"]
        best = state["best"]
        stale = state["stale"]
        start = state["epoch"]
    for epoch in range(start, settings["epochs"]):
        if stale >= settings["patience"]:
            break
        began = time.perf_counter()
        model.train()
        total = 0.0
        for ids in torch.randperm(len(data["train"]["x"])).split(settings["batch_size"]):
            x, m, mask, y = batch(data["train"], ids)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x, m, mask), y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        val = evaluate(model, data["validation"], settings["batch_size"])
        metadata = dict(
            kind="supervised-move-ranker",
            seed=seed,
            epoch=epoch + 1,
            config=config,
            value_initializer_sha256=digest(config["baseline"]),
            data_sha256=data_sha,
            provenance=provenance(),
        )
        if val["cross_entropy"] < best:
            best = val["cross_entropy"]
            stale = 0
            export_ranker(model, root / "best.npz", metadata)
        else:
            stale += 1
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=total / len(data["train"]["x"]),
                validation=val,
                seconds=time.perf_counter() - began,
                optimizer_steps=int(np.ceil(len(data["train"]["x"]) / settings["batch_size"])),
            )
        )
        export_ranker(model, root / "last.npz", metadata)
        save_state(
            state_path,
            dict(
                signature=signature,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(),
                history=history,
                best=best,
                stale=stale,
                epoch=epoch + 1,
            ),
        )
        write_json(
            root / "metrics.json", dict(history=history, best_validation_loss=best, signature=signature)
        )
        print(
            f"Seed {seed} epoch {epoch + 1}: CE {val['cross_entropy']:.4f}, top1 {val['top1']:.3f}, {history[-1]['seconds']:.1f}s",
            flush=True,
        )
        if stop_after is not None and epoch + 1 >= stop_after:
            break
    return dict(
        seed=seed, checkpoint=str(root / "best.npz"), sha256=digest(root / "best.npz"), history=history
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()
    config = read_json(args.config)
    data, sha = dataset(config)
    results = []
    for seed in config["seeds"][:1] if args.benchmark else config["seeds"]:
        results.append(train(config, seed, data, sha, stop_after=1 if args.benchmark else None))
    write_json(
        Path(config["output"]) / ("training-benchmark.json" if args.benchmark else "training-results.json"),
        results,
    )
