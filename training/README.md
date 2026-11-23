# Isolated GPU training environment

Use Python **3.12**. The production `pyproject.toml` is intentionally unchanged apart from packaging the CPU action mapping: production installs neither JAX nor PyTorch.

CPU development (tested on Apple Silicon):

```sh
python3.12 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install --require-hashes -r training/requirements.lock
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu doctor
PYTHONPATH=src .venv-gpu/bin/python -m pytest -q tests/gpu
```

Authorized NVIDIA Linux instance (CUDA 12, compatible host driver required):

```sh
python3.12 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install --require-hashes -r training/requirements-cuda.lock
PYTHONPATH=src .venv-gpu/bin/python -m chessengine.gpu doctor --require-nvidia
PYTHONPATH=src .venv-gpu/bin/python -m pytest -q tests/gpu
```

The CPU lock was installed and tested locally. The CUDA lock was installed on an authorized NVIDIA A40: doctor --require-nvidia and all 17 GPU tests passed. GPU/NumPy benchmark export difference was below 1e-5. These checks establish correctness on that instance, not a completed learning study or playing strength. Keep source/runtime identities with each measurement and recheck the device after a host change. Stockfish labeling can run in the CPU environment separately from GPU fitting; do not use macOS CPU performance as a GPU-training forecast.

Core pins are JAX/JAXlib 0.6.2, PGX 2.6.0, mctx 0.0.5, evosax 0.2.0 and Optax 0.2.4. These are deliberately tested compatible versions, rather than unbounded latest releases. Full transitive versions and artifact hashes are in the locks. Re-resolving a lock is a protocol change; use a new study directory. Regeneration uses uv 0.8.22 and the `.in` files; the CUDA lock targets `x86_64-manylinux_2_28`.

Source references: [JAX installation](https://docs.jax.dev/en/latest/installation.html), [PGX](https://github.com/sotetsuk/pgx), [mctx](https://github.com/google-deepmind/mctx), [evosax](https://github.com/RobertTLange/evosax), [Optax](https://github.com/google-deepmind/optax). The rules adapter uses pinned PGX private interfaces, so upgrades require renewed parity tests.

GPU-first evolutionary calibration uses the same locked Python 3.12 environment; see [pilot operations](../docs/evolutionary-gpu-pilot.md). It is a separately identified study and does not resume historical runs.
