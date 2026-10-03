from __future__ import annotations

import random
import time
from pathlib import Path
import numpy as np
import torch
from .common import digest, json_hash, provenance, write_json
from .data import records, validate_splits
from .features import encode, restore, FEATURE_VERSION
from .model import build_model, export


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def device_for(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError("Unknown device")
    return requested


def save_state(path, state):
    temp = Path(str(path) + ".tmp")
    torch.save(state, temp)
    temp.replace(path)


def trusted_load(path):
    # Resume files are local trusted artifacts only. Public inference uses non-pickled NPZ.
    return torch.load(path, map_location="cpu", weights_only=False)


def fit(source, output, config, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config["seed"])
    rows = list(records(source))
    validate_splits(rows)
    subsets = {}
    for split in ("train", "validation"):
        selected = [r for r in rows if r["split"] == split]
        if not selected:
            raise ValueError(f"Missing {split} data")
        subsets[split] = (
            torch.from_numpy(np.stack([encode(restore(r)) for r in selected])),
            torch.tensor([r["target"] for r in selected], dtype=torch.float32),
        )
    device = device_for(config.get("device", "cpu"))
    model = build_model(config["hidden"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    identity = {k: v for k, v in config.items() if k != "epochs"}
    signature = json_hash({"data": digest(source), "config": identity, "features": FEATURE_VERSION})
    history, start_epoch, best, stale, elapsed = [], 0, float("inf"), 0, 0.0
    state_path = output / "resume.pt"
    if resume:
        state = trusted_load(state_path)
        if state["signature"] != signature:
            raise ValueError("Resume data/config mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, start_epoch, best, stale, elapsed = (
            state[k] for k in ["history", "epoch", "best", "stale", "elapsed"]
        )
        restore_rng(state["rng"])
    elif state_path.exists():
        raise ValueError("Output already contains a run; use --resume or a new directory")
    x, y = subsets["train"]
    xv, yv = subsets["validation"]
    for epoch in range(start_epoch, config["epochs"]):
        if stale >= config["patience"]:
            break
        started = time.perf_counter()
        model.train()
        total = 0
        order = torch.randperm(len(x))
        for ids in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[ids].to(device)).squeeze(-1)
            loss = torch.mean((prediction - y[ids].to(device)) ** 2)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        model.eval()
        with torch.no_grad():
            validation_total = 0
            for ids in torch.arange(len(xv)).split(config["batch_size"]):
                validation_total += (
                    ((model(xv[ids].to(device)).squeeze(-1) - yv[ids].to(device)) ** 2).sum().item()
                )
            val_loss = validation_total / len(xv)
        elapsed += time.perf_counter() - started
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(x),
                "validation_loss": val_loss,
                "update_seconds": elapsed,
            }
        )
        metadata = {
            "hidden": config["hidden"],
            "kind": "supervised",
            "config": config,
            "data_sha256": digest(source),
            "provenance": provenance(),
            "epoch": epoch + 1,
            "evidence": config.get("evidence", "pilot"),
        }
        if val_loss < best:
            best, stale = val_loss, 0
            export(model, output / "best.npz", metadata)
        else:
            stale += 1
        export(model, output / "last.npz", metadata)
        save_state(
            state_path,
            {
                "signature": signature,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(),
                "epoch": epoch + 1,
                "best": best,
                "stale": stale,
                "history": history,
                "elapsed": elapsed,
            },
        )
        write_json(
            output / "metrics.json",
            {
                "config": config,
                "device": device,
                "history": history,
                "best_validation_mse": best,
                "data_sha256": digest(source),
                "provenance": provenance(),
            },
        )
    return {"best_validation_mse": best, "epochs": len(history), "checkpoint": str(output / "best.npz")}
