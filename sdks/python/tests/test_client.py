"""Tests for the AnamDB Python client.

These tests mock the TCP connection to verify protocol encoding/decoding
without requiring a running AnamDB server.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from anamdb.client import AnamClient
from anamdb.exceptions import ConnectionError, QueryError
from anamdb.models import QueryResult, ServerHealth


# ── Helpers ──────────────────────────────────────────────────────────


async def _make_mock_server(
    responses: list[dict],
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[asyncio.Server, int]:
    """Start a mock TCP server that returns pre-canned JSON responses."""
    response_iter = iter(responses)

    async def handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        while True:
            line = await reader.readline()
            if not line:
                break
            resp = next(response_iter, {"ok": False, "error": "no more responses"})
            payload = json.dumps(resp) + "\n"
            writer.write(payload.encode())
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(handler, host, port)
    actual_port = server.sockets[0].getsockname()[1]
    return server, actual_port


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health() -> None:
    server, port = await _make_mock_server(
        [
            {
                "status": "SERVING",
                "version": "1.0.0",
                "tables": 3,
                "models": 2,
                "rules": 5,
            }
        ]
    )
    async with server:
        async with AnamClient(f"127.0.0.1:{port}") as client:
            health = await client.health()

        assert isinstance(health, ServerHealth)
        assert health.status == "SERVING"
        assert health.version == "1.0.0"
        assert health.table_count == 3
        assert health.model_count == 2
        assert health.rule_count == 5


@pytest.mark.asyncio
async def test_query_success() -> None:
    server, port = await _make_mock_server(
        [
            {
                "ok": True,
                "ipc_bytes": 1024,
                "reasoning_tree": "high_risk <- fraud_prob > 0.90",
                "anomalies": [],
            }
        ]
    )
    async with server:
        async with AnamClient(f"127.0.0.1:{port}") as client:
            result = await client.query("SELECT * FROM txns LIMIT 10")

        assert isinstance(result, QueryResult)
        assert result.row_count == 1024
        assert result.reasoning_tree == "high_risk <- fraud_prob > 0.90"
        assert result.anomalies == []


@pytest.mark.asyncio
async def test_query_error() -> None:
    server, port = await _make_mock_server(
        [{"ok": False, "error": "table 'missing' not found"}]
    )
    async with server:
        async with AnamClient(f"127.0.0.1:{port}") as client:
            with pytest.raises(QueryError, match="table 'missing' not found"):
                await client.query("SELECT * FROM missing")


@pytest.mark.asyncio
async def test_register_table() -> None:
    server, port = await _make_mock_server(
        [{"ok": True, "message": "table 'txns' registered"}]
    )
    async with server:
        async with AnamClient(f"127.0.0.1:{port}") as client:
            resp = await client.register_table("txns", "/data/txns.lance")

        assert resp.success is True
        assert "txns" in resp.message


@pytest.mark.asyncio
async def test_register_rule() -> None:
    server, port = await _make_mock_server(
        [{"ok": True, "message": "rule 'high_risk' registered"}]
    )
    async with server:
        async with AnamClient(f"127.0.0.1:{port}") as client:
            resp = await client.register_rule(
                "high_risk", "fraud_prob > 0.90 AND amount > 10000"
            )

        assert resp.success is True
        assert "high_risk" in resp.message


@pytest.mark.asyncio
async def test_connection_refused() -> None:
    client = AnamClient("127.0.0.1:19999", connect_timeout=0.5)
    with pytest.raises(ConnectionError):
        await client.connect()


@pytest.mark.asyncio
async def test_context_manager_auto_close() -> None:
    server, port = await _make_mock_server(
        [{"status": "SERVING", "version": "1.0.0", "tables": 0, "models": 0, "rules": 0}]
    )
    async with server:
        async with AnamClient(f"127.0.0.1:{port}") as client:
            assert client._writer is not None
            await client.health()

        # After exiting, writer should be None.
        assert client._writer is None
