"""FP32 reference-recipe fitting, exact CPU continuation, sealed held-out evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import numpy as np
import torch
from .common import digest, json_hash, read_json
from .elite_data import atomic_json, preflight, ShardDataset
from .elite_policy import build_policy, export_policy, NumpyElitePolicy, MOVES
from .train import seed_all, rng_state, restore_rng, save_state, trusted_load


def runtime():
    import chess
    return dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, chess=chess.__version__)


def scientific_hashes():
    return {name: digest(Path(__file__).with_name(name + ".py"))
            for name in ["elite_policy", "elite_data", "elite_train", "train"]}


def source_commit():
    repository = Path(__file__).resolve().parents[2]
    stamp = repository / 'source-revision.json'
    if stamp.exists():
        return read_json(stamp)['commit']
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository,
                                   stderr=subprocess.DEVNULL, timeout=5).decode().strip()


class MemoryBatches:
    """Keep compact binary boards/legal masks in RAM; globally shuffle examples like the reference."""
    def __init__(self, root, split, batch_size):
        self.batch_size = batch_size
        self.dataset = ShardDataset(root, split)
        n = len(self.dataset)
        self.x = np.empty((n, 104), dtype=np.uint8)
        self.y = np.empty(n, dtype=np.int64)
        self.games = np.empty(n, dtype="U64")
        self.legal = np.zeros((n, len(MOVES) // 8), dtype=np.uint8)
        offset = 0
        for shard in self.dataset.parts:
            with np.load(Path(root) / shard["path"], allow_pickle=False) as archive:
                d = {key: archive[key] for key in archive.files}
                end = offset + shard["count"]
                self.x[offset:end], self.y[offset:end], self.games[offset:end] = d["x"], d["y"], d["games"]
                for i in range(shard["count"]):
                    ids = d["legal"][d["pointers"][i]:d["pointers"][i + 1]].astype(np.int64)
                    if int(d["y"][i]) not in ids or np.any(ids >= len(MOVES)):
                        raise ValueError("Invalid legal/played move in elite dataset")
                    # Duplicate byte indices need bitwise accumulation, not advanced assignment.
                    np.bitwise_or.at(self.legal[offset + i], ids // 8, (128 >> (ids % 8)).astype(np.uint8))
                offset = end

    def __len__(self):
        return len(self.y)

    def batches(self, device, order=None, start=0):
        order = np.arange(len(self)) if order is None else order
        for offset in range(start, len(order), self.batch_size):
            ids = order[offset:offset + self.batch_size]
            x = np.unpackbits(self.x[ids], axis=1).reshape(-1, 13, 8, 8).astype(np.float32)
            mask = np.unpackbits(self.legal[ids], axis=1).astype(bool)
            yield (torch.from_numpy(x).to(device), torch.from_numpy(self.y[ids]).to(device),
                   self.games[ids], torch.from_numpy(mask).to(device), offset + len(ids))


def evaluate_model(model, data, device):
    model.eval()
    games = {}
    with torch.inference_mode():
        for x, y, gids, mask, _ in data.batches(device):
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite elite validation/test logits")
            losses = torch.nn.functional.cross_entropy(logits, y, reduction="none")
            legal = logits.masked_fill(~mask, -float("inf"))
            target = legal.gather(1, y[:, None])
            ids = torch.arange(len(MOVES), device=device)[None, :]
            # Deterministic UCI-order ties, shared with NumPy inference.
            ranks = ((legal > target) | ((legal == target) & (ids < y[:, None]))).sum(1) + 1
            uniform = mask.sum(1).float().reciprocal()
            values = torch.stack([losses, (ranks == 1).float(), (ranks <= 3).float(), ranks.float(),
                                  uniform, (ranks == 1).float() - uniform, torch.ones_like(losses)], 1).cpu().numpy()
            for gid, value in zip(gids, values):
                games.setdefault(str(gid), np.zeros(7, dtype=np.float64))[:] += value
    if not games:
        raise ValueError("Empty elite evaluation split")
    total = np.stack(list(games.values())).sum(0)
    names = ["cross_entropy", "top1", "top3", "mean_rank", "uniform_top1", "top1_difference"]
    result = {name: float(total[i] / total[-1]) for i, name in enumerate(names)}
    result.update(positions=int(total[-1]), source_games=len(games))
    return result, games


def bootstrap(games, samples=1000, seed=20261124):
    values = np.stack([games[g] for g in sorted(games)])
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, 6))
    for i in range(samples):
        total = values[rng.integers(len(values), size=len(values))].sum(0)
        draws[i] = total[:6] / total[-1]
    return {name: np.quantile(draws[:, i], [0.025, 0.975]).tolist() for i, name in enumerate(
        ["cross_entropy", "top1", "top3", "mean_rank", "uniform_top1", "top1_difference"])}


def torch_from_export(path, device):
    policy = NumpyElitePolicy(path)
    model = build_policy().to(device)
    with torch.no_grad():
        for prefix, layer in [("c0", model.c0), ("c1", model.c1), ("f", model.context), ("o", model.output)]:
            layer.weight.copy_(torch.from_numpy(policy.arrays[prefix + "w"]).to(device))
            layer.bias.copy_(torch.from_numpy(policy.arrays[prefix + "b"]).to(device))
    return model, policy


def parity(path, root, splits=("train", "validation"), per_split=12):
    model, policy = torch_from_export(path, torch.device("cpu"))
    model.eval()
    checks = []
    for split in splits:
        data = ShardDataset(root, split)
        for index in np.linspace(0, len(data) - 1, min(per_split, len(data)), dtype=int):
            x, _, _, legal = data[index]
            legal = np.sort(legal)
            with torch.inference_mode():
                a = model(torch.from_numpy(x[None])).numpy()[0]
            b = policy.logits(x)
            pa = np.exp(a[legal].astype(np.float64) - float(a[legal].max()))
            pb = np.exp(b[legal].astype(np.float64) - float(b[legal].max()))
            pa, pb = pa / pa.sum(), pb / pb.sum()
            check = dict(split=split, index=int(index), logits_max_error=float(np.max(np.abs(a - b))),
                         probability_max_error=float(np.max(np.abs(pa - pb))),
                         same_move=bool(legal[np.argmax(a[legal])] == legal[np.argmax(b[legal])]))
            check["passed"] = bool(np.allclose(a, b, atol=2e-4, rtol=2e-5)
                                   and np.allclose(pa, pb, atol=2e-5, rtol=2e-4) and check["same_move"])
            checks.append(check)
    return dict(passed=all(c["passed"] for c in checks), checks=checks)


def train(config, root, seed, requested="cpu", resume=False, stop_after_steps=None):
    if requested not in ("cpu", "cuda") or (requested == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested elite training device unavailable; no fallback")
    if seed not in config["seeds"]:
        raise ValueError("Seed outside elite protocol")
    device, root = torch.device(requested), Path(root)
    if preflight(root)['config'] != config:
        raise ValueError('Elite dataset/config mismatch')
    signature = json_hash(dict(config=config, dataset=digest(root / "dataset.json"), seed=seed,
                               runtime=runtime(), source=scientific_hashes(), device=requested))
    seed_all(seed)
    commit = source_commit()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device.type == "cpu":
        torch.use_deterministic_algorithms(True)
    model = build_policy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    directory = root / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'failure.json').exists():
        raise ValueError('A failed scientific attempt cannot be silently retried')
    path = directory / "resume.pt"
    state = dict(epoch=0, cursor=0, steps=0, history=[], best=float("inf"), online_sum=0., online_count=0,
                 elapsed=0., signature=signature, source_commit=commit)
    if resume:
        loaded = trusted_load(path)
        if loaded["signature"] != signature:
            raise ValueError("Elite resume source/runtime/data/config mismatch")
        model.load_state_dict(loaded.pop("model"))
        optimizer.load_state_dict(loaded.pop("optimizer"))
        restore_rng(loaded.pop("rng"))
        cuda = loaded.pop("cuda_rng")
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda)
        state = loaded
    elif path.exists():
        raise ValueError("Existing elite run; use --resume")
    training = MemoryBatches(root, "train", config["training"]["batch_size"])
    validation = MemoryBatches(root, "validation", config["training"]["batch_size"])
    began = time.perf_counter()
    base_elapsed = state["elapsed"]

    def save():
        state["elapsed"] = base_elapsed + time.perf_counter() - began
        save_state(path, {**state, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "rng": rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        atomic_json(directory / "metrics.json", {**state, "complete": state["epoch"] >= config["training"]["epochs"],
                                                "device": requested})

    try:
        while state["epoch"] < config["training"]["epochs"]:
            epoch_start = time.perf_counter()
            order = np.random.default_rng(np.random.SeedSequence([seed, state["epoch"]])).permutation(len(training))
            model.train()
            for x, y, _, _, cursor in training.batches(device, order, state["cursor"]):
                if time.time() >= float(os.environ.get("ELITE_DEADLINE_UNIX", "inf")) - 60:
                    save()
                    return dict(state, paused="budget_deadline")
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise ValueError("Nonfinite elite training output/loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"],
                                                       error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite elite gradient")
                optimizer.step()
                state["online_sum"] += float(loss.detach()) * len(y)
                state["online_count"] += len(y)
                state["cursor"], state["steps"] = cursor, state["steps"] + 1
                if state["steps"] % config["training"].get("checkpoint_steps", 500) == 0:
                    save()
                if stop_after_steps is not None and state["steps"] >= stop_after_steps:
                    save()
                    return state
            val, _ = evaluate_model(model, validation, device)
            metadata = dict(seed=seed, epoch=state["epoch"] + 1, dataset_sha256=digest(root / "dataset.json"),
                            commit=commit,
                            study=config["study"], signature=signature, runtime=runtime(),
                            scientific_source=scientific_hashes(), config=config,
                            training_positions=len(training), evidence="elite-pytorch-baseline-20261124")
            if val["cross_entropy"] < state["best"]:
                state["best"] = val["cross_entropy"]
                export_policy(model, directory / "best.npz", metadata)
            export_policy(model, directory / "last.npz", metadata)
            state["history"].append(dict(epoch=state["epoch"] + 1, validation=val,
                                         online_training_loss=state["online_sum"] / state["online_count"],
                                         seconds=time.perf_counter() - epoch_start))
            state.update(epoch=state["epoch"] + 1, cursor=0, online_sum=0., online_count=0)
            save()
            print(f"Seed {seed} epoch {state['epoch']}: validation CE={val['cross_entropy']:.6f}", flush=True)
    except Exception as exc:
        # Keep the last finite resume state and record the failed attempt; never relabel it completed.
        atomic_json(directory / "failure.json", dict(error_type=type(exc).__name__, message=str(exc),
                                                    epoch=state["epoch"], step=state["steps"], signature=signature))
        raise
    return state


def evaluate_all(config, root, device="cpu"):
    root = Path(root)
    selections = []
    for seed in config["seeds"]:
        directory = root / f"seed-{seed}"
        failed = (directory / 'failure.json').exists()
        metrics = read_json(directory / "metrics.json") if (directory / 'metrics.json').exists() else {}
        if not failed and not metrics.get('complete'):
            raise ValueError("All elite seed attempts must finish before test opening")
        path = directory / 'best.npz'
        selections.append(dict(seed=seed, status='failed' if failed else 'complete',
            checkpoint_sha256=digest(path) if path.exists() else None,
            selected_epoch=NumpyElitePolicy(path).metadata['epoch'] if path.exists() else None))
    seal = dict(models=selections, selection_basis="validation-cross-entropy-only",
                representative_seed=42, dataset_sha256=digest(root / "dataset.json"))
    sealed = root / "selections.json"
    if sealed.exists() and read_json(sealed) != seal:
        raise ValueError("Sealed elite selections changed")
    atomic_json(sealed, seal)  # No test loader has been constructed yet.
    results = []
    for selected in selections:
        seed = selected["seed"]
        path = root / f"seed-{seed}" / "best.npz"
        if selected['status'] == 'failed':
            metrics = root / f'seed-{seed}/metrics.json'
            row = dict(selected, failure=read_json(root / f'seed-{seed}/failure.json'),
                       history=read_json(metrics)['history'] if metrics.exists() else [])
            results.append(row)
            if seed == 42:
                atomic_json(root / 'gate.json', dict(seed=42, passed=False, export_passed=False,
                    checkpoint_sha256=selected['checkpoint_sha256'], dataset_sha256=seal['dataset_sha256'],
                    selections_sha256=digest(sealed), reason='representative-seed-failed'))
            continue
        model, _ = torch_from_export(path, torch.device(device))
        metrics = read_json(root / f'seed-{seed}/metrics.json')
        row = dict(selected, history=metrics['history'], training_elapsed_seconds=metrics['elapsed'])
        for split in ["train", "validation", "test"]:
            data = MemoryBatches(root, split, config["training"]["batch_size"])
            metrics, games = evaluate_model(model, data, torch.device(device))
            row[split] = dict(metrics, intervals=bootstrap(games))
            del data
        row["export"] = parity(path, root)
        results.append(row)
        if seed == 42:
            ci = row["validation"]["intervals"]["top1_difference"]
            gate = dict(seed=42, checkpoint_sha256=selected["checkpoint_sha256"],
                        dataset_sha256=seal["dataset_sha256"], selections_sha256=digest(sealed),
                        validation_top1_difference_ci=ci, export_passed=row["export"]["passed"],
                        passed=bool(ci[0] > 0 and row["export"]["passed"]))
            atomic_json(root / "gate.json", gate)
        atomic_json(root / "results.json", dict(selections_sha256=digest(sealed), seeds=results,
                    uncertainty="95% whole-source-game bootstrap; seed variation reported separately",
                    interpretation="Human-move agreement is not playing-strength evidence"))
    scores = [r["test"]["top1"] for r in results if 'test' in r]
    summary = dict(selections_sha256=digest(sealed), seeds=results,
                   interpretation='Human-move agreement is not playing-strength evidence')
    summary["seed_variability"] = dict(evaluable_seeds=len(scores), failed_seeds=len(results) - len(scores),
        mean_top1=float(np.mean(scores)) if scores else None,
        sample_sd=float(np.std(scores, ddof=1)) if len(scores) > 1 else None)
    atomic_json(root / "results.json", summary)
    return summary


def benchmark(config, root, seconds=30):
    if not torch.cuda.is_available():
        raise ValueError("GPU benchmark requires CUDA")
    seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = MemoryBatches(root, "train", config["training"]["batch_size"])
    model = build_policy().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    start, count = time.perf_counter(), 0
    for x, y, _, _, _ in data.batches(torch.device("cuda")):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        count += len(y)
        if time.perf_counter() - start >= seconds:
            break
    elapsed = time.perf_counter() - start
    result = dict(examples=count, seconds=elapsed, examples_per_second=count / elapsed,
                  forecast_training_seconds=len(data) * config["training"]["epochs"] * len(config["seeds"])
                  / (count / elapsed), excludes="validation, export, setup and backups; add measured overhead",
                  runtime=runtime(), peak_vram_bytes=torch.cuda.max_memory_allocated(), updates_discarded=True)
    atomic_json(Path(root) / "benchmark.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["train", "evaluate", "benchmark"])
    p.add_argument("config")
    p.add_argument("--root", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    config = read_json(a.config)
    if a.action == "train":
        if a.seed is None:
            p.error("train requires --seed")
        train(config, a.root, a.seed, a.device, a.resume)
    elif a.action == "evaluate":
        evaluate_all(config, a.root, a.device)
    else:
        print(json.dumps(benchmark(config, a.root), indent=2))


if __name__ == "__main__":
    main()
