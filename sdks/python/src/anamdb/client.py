"""Async client for connecting to a running AnamDB server.

Uses the JSON-over-TCP wire protocol defined in the AnamDB server module.
Each command is a single JSON line; each response is a single JSON line back.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from anamdb.exceptions import (
    AnamDBError,
    ConnectionError,
    ProtocolError,
    QueryError,
)
from anamdb.models import (
    ModelResponse,
    QueryResult,
    RuleResponse,
    ServerHealth,
    TableResponse,
)

logger = logging.getLogger("anamdb")


class AnamClient:
    """Async client for the AnamDB neurosymbolic database engine.

    Connects to a running AnamDB server over TCP and communicates using
    the JSON-over-TCP wire protocol.

    Use as an async context manager::

        async with AnamClient("127.0.0.1:8080") as client:
            result = await client.query("SELECT * FROM txns LIMIT 10")

    Or manage the connection manually::

        client = AnamClient("127.0.0.1:8080")
        await client.connect()
        result = await client.query("SELECT * FROM txns LIMIT 10")
        await client.close()
    """

    def __init__(
        self,
        addr: str = "127.0.0.1:8080",
        *,
        connect_timeout: float = 5.0,
        max_retries: int = 3,
    ) -> None:
        self._addr = addr
        self._connect_timeout = connect_timeout
        self._max_retries = max_retries

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    # ── Connection lifecycle ──────────────────────────────────────────

    async def connect(self) -> None:
        """Establish the TCP connection to the AnamDB server."""
        if self._writer is not None:
            return  # Already connected.

        host, _, port_str = self._addr.rpartition(":")
        if not host:
            host = "127.0.0.1"
        port = int(port_str) if port_str else 8080

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self._connect_timeout,
            )
            logger.info("Connected to AnamDB at %s", self._addr)
        except asyncio.TimeoutError as exc:
            raise ConnectionError(
                f"Connection to {self._addr} timed out after {self._connect_timeout}s"
            ) from exc
        except OSError as exc:
            raise ConnectionError(
                f"Failed to connect to {self._addr}: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the TCP connection."""
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass  # Best effort.
            self._writer = None
            self._reader = None
            logger.info("Disconnected from AnamDB")

    async def __aenter__(self) -> "AnamClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ── Wire protocol ────────────────────────────────────────────────

    async def _send_command(self, cmd: dict) -> dict:
        """Send a JSON command and receive the JSON response."""
        if self._reader is None or self._writer is None:
            raise ConnectionError("Not connected — call connect() first")

        payload = json.dumps(cmd, separators=(",", ":")) + "\n"
        self._writer.write(payload.encode())
        await self._writer.drain()

        line = await self._reader.readline()
        if not line:
            raise ConnectionError("Server closed the connection")

        try:
            return json.loads(line.decode())
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"Invalid JSON response: {exc}") from exc

    async def _send_with_retry(self, cmd: dict) -> dict:
        """Send a command with automatic retry on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._send_command(cmd)
            except ConnectionError as exc:
                last_exc = exc
                logger.warning(
                    "Attempt %d/%d failed: %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
                # Try to reconnect.
                await self.close()
                try:
                    await self.connect()
                except ConnectionError:
                    pass
        raise last_exc or ConnectionError("All retry attempts failed")

    # ── Public API ───────────────────────────────────────────────────

    async def query(self, sql: str) -> QueryResult:
        """Execute a SQL query on the AnamDB server.

        Args:
            sql: The SQL query string.

        Returns:
            A :class:`QueryResult` with row count, reasoning tree, and anomalies.

        Raises:
            QueryError: If the server reports a query execution error.
        """
        resp = await self._send_with_retry({"method": "query", "sql": sql})

        if not resp.get("ok", False):
            raise QueryError(
                resp.get("error", "unknown server error"),
                sql=sql,
            )

        return QueryResult(
            row_count=resp.get("ipc_bytes", 0),
            reasoning_tree=resp.get("reasoning_tree") or None,
            anomalies=resp.get("anomalies", []),
            raw_response=resp,
        )

    async def register_table(self, name: str, lance_path: str) -> TableResponse:
        """Register a Lance dataset as a queryable table.

        Args:
            name: Logical table name.
            lance_path: Filesystem path to the Lance dataset.
        """
        resp = await self._send_with_retry(
            {"method": "register_table", "name": name, "lance_path": lance_path}
        )
        return TableResponse(
            success=resp.get("ok", False),
            message=resp.get("message", ""),
        )

    async def register_rule(self, name: str, datalog: str) -> RuleResponse:
        """Register a Datalog rule as a query filter.

        Args:
            name: Rule name.
            datalog: Datalog expression (e.g. ``"fraud_prob > 0.90 AND amount > 10000"``).
        """
        resp = await self._send_with_retry(
            {"method": "register_rule", "name": name, "datalog": datalog}
        )
        return RuleResponse(
            success=resp.get("ok", False),
            message=resp.get("message", ""),
        )

    async def load_model(
        self,
        name: str,
        version: str,
        model_path: str,
        function_id: str,
        *,
        num_features: int = 3,
        avg_latency_ms: float = 1.0,
        accuracy: float = 0.95,
    ) -> ModelResponse:
        """Load an ONNX model into the AnamDB model registry.

        Args:
            name: Model name (becomes the SQL function name).
            version: Model version string.
            model_path: Path to the ONNX model file.
            function_id: SQL function identifier.
            num_features: Number of input features.
            avg_latency_ms: Expected average latency in milliseconds.
            accuracy: Expected model accuracy (0.0–1.0).
        """
        resp = await self._send_with_retry(
            {
                "method": "load_model",
                "name": name,
                "version": version,
                "model_path": model_path,
                "function_id": function_id,
                "num_features": num_features,
                "avg_latency_ms": avg_latency_ms,
                "accuracy": accuracy,
            }
        )
        return ModelResponse(
            success=resp.get("ok", False),
            model_id=resp.get("model_id", ""),
            message=resp.get("message", ""),
        )

    async def health(self) -> ServerHealth:
        """Check the health of the AnamDB server.

        Returns:
            A :class:`ServerHealth` with server status and resource counts.
        """
        resp = await self._send_with_retry({"method": "health"})
        return ServerHealth(
            status=resp.get("status", "UNKNOWN"),
            version=resp.get("version", "?"),
            table_count=resp.get("tables", 0),
            model_count=resp.get("models", 0),
            rule_count=resp.get("rules", 0),
        )
