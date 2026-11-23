"""Build a private, allowlisted Runpod training archive. Does not upload or launch."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    repository = Path(__file__).resolve().parents[1]
    dataset = args.dataset.resolve()
    meta = json.loads((dataset / "dataset.json").read_text())
    allowed = [Path("pyproject.toml"), Path("requirements-train.txt"), Path("requirements-runtime.txt"),
               Path("requirements-policy.txt"),
               Path("docs/stockfish-move-policy.md"),
               Path("scripts/run_move_policy.sh")]
    allowed += [f.relative_to(repository) for f in sorted((repository / "src/chessengine").glob("*.py"))]
    shards = []
    for shard in meta["shards"]:
        path = (dataset / shard["path"]).resolve()
        if not path.is_relative_to(dataset) or not path.is_file():
            raise ValueError("Unsafe or missing dataset shard")
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError("Corrupt dataset shard")
        shards.append(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stamp = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository).decode().strip(),
             "files": {str(p): hashlib.sha256((repository / p).read_bytes()).hexdigest() for p in allowed}}
    with tarfile.open(args.output, "w:gz") as archive:
        for name in allowed:
            archive.add(repository / name, arcname=str(name), recursive=False)
        archive.add(dataset / "dataset.json", arcname="runs/stockfish-move-policy/dataset.json")
        for path in shards:
            archive.add(path, arcname="runs/stockfish-move-policy/" + str(path.relative_to(dataset)))
        for name, value in [("source-revision.json", stamp),
                            ("configs/stockfish-move-policy.json", meta["config"])]:
            content = json.dumps(value, indent=2).encode()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    print(f"Private archive: {args.output}; no credentials, Git history, or released weights included")


if __name__ == "__main__":
    main()
