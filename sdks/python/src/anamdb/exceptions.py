"""Exception hierarchy for the AnamDB Python SDK."""


class AnamDBError(Exception):
    """Base exception for all AnamDB client errors."""


class ConnectionError(AnamDBError):
    """Raised when the client cannot connect to the AnamDB server."""


class QueryError(AnamDBError):
    """Raised when a SQL query fails on the server."""

    def __init__(self, message: str, sql: str | None = None):
        self.sql = sql
        super().__init__(message)


class ProtocolError(AnamDBError):
    """Raised when the server sends an invalid or unexpected response."""


class TimeoutError(AnamDBError):
    """Raised when an operation exceeds the configured timeout."""
