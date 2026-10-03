import asyncio
import time
from unittest.mock import patch
import httpx
from chessengine.api import create_app
from chessengine.engine import Search


def test_concurrent_request_gets_busy_response(tmp_path):
    real = Search.run

    def delayed(self, board):
        time.sleep(0.15)
        return real(self, board)

    async def run():
        transport = httpx.ASGITransport(app=create_app(tmp_path / "missing"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(client.post("/analyze", json={"nodes": 1}))
            await asyncio.sleep(0.03)
            second = await client.post("/analyze", json={"nodes": 1})
            assert second.status_code == 429
            assert (await first).status_code == 200
            assert (await client.get("/health")).json()["busy"] is False

    with patch.object(Search, "run", delayed):
        asyncio.run(run())
