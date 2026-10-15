"""One local continuation: wait for the existing pilot, report forecast, run the frozen study."""

import argparse
import fcntl
import os
import signal
import threading
from importlib.metadata import version
import time
from pathlib import Path
from .common import read_json, write_json, digest
from .pv_study import run, peak_mb
from .research_catalog import publish


def continue_study(config, wait_pilot=False):
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "release-scope.json").exists():
        raise RuntimeError("Study closed at feasibility scope; a new authorization is required to resume")
    # This lock prevents duplicate continuation processes; never run a raw stage concurrently.
    with (root / "continuation.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A continuation already owns this study") from error

        def state(status, **fields):
            write_json(root / "continuation-status.json", dict(status=status, pid=os.getpid(), **fields))

        try:
            state("waiting_for_pilot" if wait_pilot else "checking_pilot")
            if wait_pilot:
                while not (root / "compute.json").exists():
                    time.sleep(10)
            status = read_json(root / "status.json")["status"] if (root / "status.json").exists() else None
            if status == "complete":
                state("complete")
                return dict(status="complete")
            if status == "failed":
                raise RuntimeError("Study failed; preserve the evidence and diagnose before resuming")
            if not (root / "pilot-decision.json").exists():
                run(config, "pilot")
            decision = read_json(root / "pilot-decision.json")
            if not decision["approved"]:
                state("pilot_rejected")
                return decision
            environment = dict(
                packages={name: version(name) for name in ["torch", "numpy", "chess", "threadpoolctl"]},
                pyproject_sha256=digest("pyproject.toml"),
            )
            environment_path = root / "environment.json"
            if environment_path.exists():
                saved = read_json(environment_path)
                if any(saved.get(key) != value for key, value in environment.items()):
                    raise RuntimeError(
                        "Runtime environment changed; restore the recorded CPU environment before resuming"
                    )
            else:
                write_json(environment_path, environment)
            forecast = read_json(root / "forecast.json")
            write_json(
                root / "launch-estimate.json",
                dict(
                    forecast_sha256=digest(root / "forecast.json"),
                    estimated_hours=forecast["seconds"] / 3600,
                    paid_cost_cad=0,
                ),
            )
            # Publication and output of the estimate precede the main launch. No paid resources.
            publish()
            print(
                f"Measured full-study estimate: {forecast['seconds'] / 3600:.1f} hours ({forecast['seconds'] / 86400:.1f} days), including 50% margin. Local CPU; C$0. Starting the authorized frozen protocol.",
                flush=True,
            )
            state("main_running", forecast_hours=forecast["seconds"] / 3600)
            ledger = root / "continuation-compute.json"
            elapsed_before = read_json(ledger)["main_runner_seconds"] if ledger.exists() else 0.0
            started = time.perf_counter()
            stop = threading.Event()

            def account():
                write_json(
                    ledger,
                    dict(
                        main_runner_seconds=elapsed_before + time.perf_counter() - started,
                        peak_ram_mb=peak_mb(),
                        paid_cost_cad=0,
                    ),
                )

            def heartbeat():
                while not stop.wait(30):
                    account()

            account()
            monitor = threading.Thread(target=heartbeat, daemon=True)
            monitor.start()
            try:
                result = run(config, "all")
            finally:
                stop.set()
                monitor.join()
                account()
            state("complete" if result.get("status") == "complete" else "stopped")
            return result
        except KeyboardInterrupt:
            state("interrupted")
            raise
        except Exception as error:
            state("failed", reason=str(error))
            raise
        finally:
            publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument(
        "--wait-pilot",
        action="store_true",
        help="Wait for the already running pilot; do not start another copy",
    )
    args = parser.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    result = continue_study(read_json(args.config), args.wait_pilot)
    print({"status": result.get("status", "pilot_rejected")}, flush=True)


if __name__ == "__main__":
    main()
