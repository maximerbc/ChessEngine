"""Exercise the actual CI health probes against a Docker-style startup race."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import shlex
import shutil
import subprocess
from threading import Thread

import pytest


PROBES = [
    shlex.split(line.strip())
    for line in Path(".github/workflows/deployment.yml").read_text().splitlines()
    if line.strip().startswith("curl ")
]


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl is required for CI health probes")
@pytest.mark.parametrize("probe", PROBES, ids=["api", "website"])
def test_health_probe_retries_empty_startup_reply(probe):
    class StartingService(BaseHTTPRequestHandler):
        attempts = 0

        def do_GET(self):
            type(self).attempts += 1
            if self.attempts == 1:
                # Docker can accept the connection before the app is listening,
                # then close it without an HTTP response (curl exit code 52).
                self.close_connection = True
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), StartingService) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                [*probe[:-1], f"http://127.0.0.1:{server.server_port}/health"],
                capture_output=True, text=True, timeout=15,
            )
            assert result.returncode == 0, result.stderr
            assert result.stdout == "ok"
            assert StartingService.attempts == 2
        finally:
            server.shutdown()
            thread.join()
