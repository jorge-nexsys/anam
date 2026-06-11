"""Data models for AnamDB responses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Result of a SQL query execution."""

    row_count: int
    """Number of rows returned (from server-reported ipc_bytes, or 0)."""

    reasoning_tree: str | None = None
    """Provenance reasoning trace (if provenance mode is enabled)."""

    anomalies: list[str] = field(default_factory=list)
    """Semantic anomaly descriptions detected during execution."""

    raw_response: dict = field(default_factory=dict, repr=False)
    """Full JSON response from the server."""


@dataclass(frozen=True, slots=True)
class ServerHealth:
    """Server health status."""

    status: str
    """``'SERVING'`` or ``'NOT_SERVING'``."""

    version: str
    """AnamDB server version string."""

    table_count: int = 0
    """Number of registered tables."""

    model_count: int = 0
    """Number of loaded ONNX models."""

    rule_count: int = 0
    """Number of active Datalog rules."""


@dataclass(frozen=True, slots=True)
class TableResponse:
    """Response from a table registration request."""

    success: bool
    message: str


@dataclass(frozen=True, slots=True)
class RuleResponse:
    """Response from a Datalog rule registration request."""

    success: bool
    message: str


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Response from a model loading request."""

    success: bool
    model_id: str
    message: str
