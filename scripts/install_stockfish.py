"""Install the pinned upstream release in the API image, never at request time."""
import hashlib
import json
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/"
ASSETS = {
    "x86_64": ("stockfish-linux-x86-64-universal", "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"),
    "aarch64": ("stockfish-linux-arm64-universal", "fe26cfd1d9db4c8af3d21e24d9ff34cacb31c1f940085a7583da11796f2bac01"),
}


def install(destination):
    name, expected = ASSETS[platform.machine()]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    url = RELEASE + name + ".tar.gz"
    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "stockfish.tar.gz"
        with urllib.request.urlopen(url, timeout=120) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Stockfish release archive checksum mismatch")
        # Keep the upstream source, license and authors with the binary. Only
        # regular files are copied; archive paths cannot escape the destination.
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "stockfish":
                    raise ValueError("Unexpected Stockfish archive path")
                target = destination / Path(*relative.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        binary = destination / name
        binary.rename(destination / "stockfish")
        (destination / "stockfish").chmod(0o755)
        if not (destination / "Copying.txt").is_file() or not (destination / "src").is_dir():
            raise ValueError("Upstream Stockfish license/source missing")
        (destination / "installation.json").write_text(json.dumps({
            "release": "sf_19", "archive_url": url, "archive_sha256": expected,
            "binary_sha256": hashlib.sha256((destination / "stockfish").read_bytes()).hexdigest(),
            "source": "https://github.com/official-stockfish/Stockfish/tree/sf_19",
        }, indent=2) + "\n")


if __name__ == "__main__":
    install(sys.argv[1])
