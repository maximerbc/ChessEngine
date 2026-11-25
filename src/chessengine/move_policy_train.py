"""Research CLI for direct-move CNN training; never provisions cloud resources."""
import argparse
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from .common import digest, json_hash, read_json, write_json
from .move_policy import build_policy, export_policy, NumpyMovePolicy
from .move_policy_data import ShardDataset, collate
from .train import seed_all, rng_state, restore_rng, trusted_load, save_state


class ShuffledShards(Sampler):
    """Shuffle shards and their rows, retaining bounded decompression work."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for part in torch.randperm(len(self.dataset.parts)).tolist():
            start = self.dataset.ends[part - 1] if part else 0
            for offset in torch.randperm(self.dataset.parts[part]["count"]).tolist():
                yield start + offset


def device_for(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in ("cpu", "cuda"):
        raise ValueError("Policy training supports cpu/cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; refusing CPU fallback")
    return torch.device(requested)


def loader(root, split, settings, device, training=False):
    dataset = ShardDataset(root, split)
    # Zero workers keeps replay exact and avoids worker-local shard cache duplication.
    return DataLoader(dataset, batch_size=settings["batch_size"], collate_fn=collate,
                      sampler=ShuffledShards(dataset) if training else None,
                      num_workers=0, pin_memory=device.type == "cuda")


def amp_context(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()


def to_device(batch, device):
    return [x.to(device, non_blocking=device.type == "cuda") for x in batch[:4]]


def evaluate_model(model, batches, device):
    model.eval()
    totals = np.zeros(5, dtype=np.float64)
    games = {}
    with torch.inference_mode():
        for batch in batches:
            x, m, mask, y = to_device(batch, device)
            logits = model(x, m, mask).float()
            loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
            order = logits.argsort(dim=1, descending=True, stable=True)
            rank = (order == y[:, None]).to(torch.int64).argmax(1) + 1
            values = torch.stack((loss, (rank == 1).float(), (rank <= 3).float(), rank.float(),
                                  torch.ones_like(loss)), dim=1).cpu().numpy()
            for gid, value in zip(batch[4], values):
                games.setdefault(gid, np.zeros(5, dtype=np.float64))[:] += value
            totals += values.sum(0, dtype=np.float64)
    return summarize(totals), games


def summarize(total):
    return {"cross_entropy": float(total[0] / total[4]), "top1": float(total[1] / total[4]),
            "top3": float(total[2] / total[4]), "mean_rank": float(total[3] / total[4]),
            "positions": int(total[4])}


def bootstrap(games, seed=20261122, samples=1000):
    values = np.stack([games[g] for g in sorted(games)])
    rng = np.random.default_rng(seed)
    draws = np.stack([values[rng.integers(0, len(values), len(values))].sum(0) for _ in range(samples)])
    return {name: np.quantile(draws[:, i] / draws[:, 4], [0.025, 0.975]).tolist()
            for i, name in enumerate(("cross_entropy", "top1", "top3", "mean_rank"))}


def identity(config, root, seed, device):
    meta = read_json(Path(root) / "dataset.json")
    if meta["config"] != config:
        raise ValueError("Dataset/config mismatch")
    if meta["counts"] != config["positions"]:
        raise ValueError("Dataset count mismatch: exact approved split counts are required")
    return json_hash({"config": config, "dataset": digest(Path(root) / "dataset.json"),
                      "seed": seed, "device": str(device), "runtime": runtime_versions(),
                      "science": scientific_hashes()})


def scientific_hashes():
    return {n: digest(Path(__file__).with_name(n + ".py"))
            for n in ("move_policy", "move_policy_data", "move_policy_reuse", "move_policy_train",
                      "features", "ranking", "train")}


def runtime_versions():
    import chess
    return {"torch": torch.__version__, "numpy": np.__version__, "chess": chess.__version__,
            "python": sys.version.split()[0]}


def commit_id():
    repository = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository,
                                       stderr=subprocess.DEVNULL, timeout=10).decode().strip()
    except (OSError, subprocess.SubprocessError):
        stamp = repository / "source-revision.json"
        return read_json(stamp)["commit"] if stamp.exists() else "uncommitted"


def train(config, root, seed, requested="cpu", resume=False, stop_after=None):
    if seed not in config["seeds"]:
        raise ValueError("Seed is outside frozen configuration")
    settings = config["training"]
    device = device_for(requested)
    signature = identity(config, root, seed, device)
    seed_all(seed)
    model = build_policy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    directory = Path(root) / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "resume.pt"
    history, best, stale, start, elapsed = [], float("inf"), 0, 0, 0.0
    if resume:
        state = trusted_load(state_path)
        if state["signature"] != signature:
            raise ValueError("Policy resume identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best, stale, start, elapsed = [state[k] for k in ("history", "best", "stale", "epoch", "elapsed")]
        restore_rng(state["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    elif state_path.exists():
        raise ValueError("Existing policy run; use --resume")
    training = loader(root, "train", settings, device, True)
    validation = loader(root, "validation", settings, device)
    if (directory / "metrics.json").exists() and read_json(directory / "metrics.json")["complete"]:
        return read_json(directory / "metrics.json")
    for epoch in range(start, settings["epochs"]):
        if stale >= settings["patience"]:
            break
        began = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for batch in training:
            x, moves, mask, y = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, dtype):
                loss = torch.nn.functional.cross_entropy(model(x, moves, mask), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite policy loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(y)
            count += len(y)
        val, _ = evaluate_model(model, validation, device)
        seconds = time.perf_counter() - began
        elapsed += seconds
        history.append({"epoch": epoch + 1, "online_train_loss": total / count,
                        "validation": val, "seconds": seconds})
        metadata = {"seed": seed, "epoch": epoch + 1, "signature": signature,
                    "dataset_sha256": digest(Path(root) / "dataset.json"), "config": config,
                    "evidence": "bounded-supervised-policy-study", "device": str(device),
                    "scientific_source": scientific_hashes(), "runtime": runtime_versions(),
                    "commit": commit_id()}
        if val["cross_entropy"] < best:
            best, stale = val["cross_entropy"], 0
            export_policy(model, directory / "best.npz", metadata)
        else:
            stale += 1
        export_policy(model, directory / "last.npz", metadata)
        save_state(state_path, {"signature": signature, "model": model.state_dict(),
                               "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                               "rng": rng_state(),
                               "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                               "history": history, "best": best, "stale": stale,
                               "epoch": epoch + 1, "elapsed": elapsed})
        complete = epoch + 1 >= settings["epochs"] or stale >= settings["patience"]
        result = {"signature": signature, "seed": seed, "device": str(device), "history": history,
                  "best_validation_cross_entropy": best, "complete": complete,
                  "elapsed_seconds": elapsed, "best_sha256": digest(directory / "best.npz")}
        remaining_epochs = (settings["epochs"] - epoch - 1 + settings["epochs"] *
                            (len(config["seeds"]) - config["seeds"].index(seed) - 1))
        result["maximum_remaining_seconds_forecast"] = seconds * remaining_epochs * 1.2
        deadline = float(os.environ.get("CHESS_POLICY_DEADLINE_UNIX", "inf"))
        if not complete and time.time() + result["maximum_remaining_seconds_forecast"] > deadline - 60:
            result["pause_reason"] = "remaining_fixed_protocol_exceeds_budget_deadline"
        write_json(directory / "metrics.json", result)
        print(f"seed {seed} epoch {epoch + 1}: validation CE {val['cross_entropy']:.5f}", flush=True)
        if "pause_reason" in result:
            break
        if stop_after and epoch + 1 >= stop_after:
            break
    return read_json(directory / "metrics.json")


def load_export_to_torch(path, device):
    policy = NumpyMovePolicy(path)  # Strict shape/schema/dtype validation before tensor loading.
    model = build_policy().to(device)
    with torch.no_grad():
        for prefix, layer in [("c0", model.c0), ("c1", model.c1), ("f", model.context),
                              ("h", model.head), ("o", model.output)]:
            layer.weight.copy_(torch.from_numpy(policy.arrays[prefix + "w"]).to(device))
            layer.bias.copy_(torch.from_numpy(policy.arrays[prefix + "b"]).to(device))
    return model, policy.metadata


def evaluate_all(config, root, requested="cpu"):
    root = Path(root)
    device = device_for(requested)
    frozen = {"config": config, "dataset_sha256": digest(root / "dataset.json"), "checkpoints": {}}
    # Validate and freeze ALL selected checkpoints before constructing the test dataset.
    for seed in config["seeds"]:
        metrics = read_json(root / f"seed-{seed}/metrics.json")
        path = root / f"seed-{seed}/best.npz"
        policy = NumpyMovePolicy(path)
        if (not metrics["complete"] or metrics["best_sha256"] != policy.checkpoint_id
                or policy.metadata["seed"] != seed
                or policy.metadata["scientific_source"] != scientific_hashes()
                or policy.metadata["signature"] != metrics["signature"]
                or policy.metadata["dataset_sha256"] != frozen["dataset_sha256"]
                or policy.metadata["config"] != config):
            raise ValueError("Every seed must finish selection before testing")
        frozen["checkpoints"][str(seed)] = policy.checkpoint_id
    protocol = root / "test-protocol.json"
    if protocol.exists() and read_json(protocol) != frozen:
        raise ValueError("Frozen test protocol changed")
    write_json(protocol, frozen)
    test = loader(root, "test", config["training"], device)
    results = []
    for seed in config["seeds"]:
        model, _ = load_export_to_torch(root / f"seed-{seed}/best.npz", device)
        metrics, games = evaluate_model(model, test, device)
        training, _ = evaluate_model(model, loader(root, "train", config["training"], device), device)
        policy = NumpyMovePolicy(root / f"seed-{seed}/best.npz")
        timing_data = test.dataset
        timings = []
        for i in range(min(32, len(timing_data))):
            x, moves, _, _ = timing_data[i]
            began = time.perf_counter()
            policy.logits(x, moves)
            timings.append(time.perf_counter() - began)
        results.append({"seed": seed, "test": metrics, "frozen_checkpoint_train": training,
                        "numpy_encoded_inference_median_ms": float(np.median(timings) * 1000),
                        "test_whole_game_95_interval": bootstrap(games), "test_games": len(games)})
    result = {"protocol": frozen, "results": results,
              "seed_top1_standard_deviation": float(np.std([r["test"]["top1"] for r in results], ddof=1))
              if len(results) > 1 else None,
              "interpretation": "Teacher imitation only; no playing-strength or promotion evidence."}
    write_json(root / "test-results.json", result)
    return result


def benchmark(config, root, requested="cpu", steps=20):
    if steps < 1:
        raise ValueError("Positive benchmark steps required")
    seed_all(config["seeds"][0])
    device = device_for(requested)
    batches = loader(root, "train", config["training"], device, True)
    model = build_policy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    iterator = iter(batches)
    measured = 0
    durations = []
    for step in range(steps + 3):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            batch = next(iterator)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        x, moves, mask, y = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x, moves, mask), y)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= 3:
            durations.append(time.perf_counter() - start)
            measured += len(y)
    throughput = measured / sum(durations)
    maximum = (len(batches.dataset) / throughput * config["training"]["epochs"] * len(config["seeds"]))
    result = {"device": str(device), "positions_per_second": throughput, "measured_steps": steps,
              "all_seeds_training_only_forecast_seconds": maximum,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "excludes": ["validation", "checkpoint writes", "data loading", "labeling", "setup"],
              "paid_training_authorized": False}
    write_json(Path(root) / "benchmark.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "label", "encode", "benchmark", "train", "evaluate", "predict"])
    parser.add_argument("--config", default="configs/stockfish-move-policy.json")
    parser.add_argument("--output", default="runs/stockfish-move-policy")
    parser.add_argument("--input-root", default=".")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--position", help="JSON record with initial_fen, complete moves, and fen")
    args = parser.parse_args()
    config = read_json(args.config)
    from .move_policy_data import prepare, label, encode_shards
    if args.stage == "prepare":
        result = prepare(config, args.output, args.input_root)
    elif args.stage == "label":
        result = label(config, args.output, args.input_root)
    elif args.stage == "encode":
        result = encode_shards(config, args.output)
    elif args.stage == "benchmark":
        result = benchmark(config, args.output, args.device)
    elif args.stage == "train":
        seeds = [args.seed] if args.seed is not None else config["seeds"]
        result = []
        for seed in seeds:
            fitted = train(config, args.output, seed, args.device,
                           args.resume and (Path(args.output) / f"seed-{seed}/resume.pt").exists())
            result.append(fitted)
            if not fitted["complete"]:
                break
    elif args.stage == "evaluate":
        result = evaluate_all(config, args.output, args.device)
    else:
        from .features import restore
        if not args.checkpoint or not args.position:
            parser.error("predict requires --checkpoint and --position")
        result = NumpyMovePolicy(args.checkpoint).rank(restore(read_json(args.position)))
    import json
    print(json.dumps(result, indent=2))
    if args.stage == "train" and any(not row["complete"] for row in result):
        raise SystemExit(75)


if __name__ == "__main__":
    main()
