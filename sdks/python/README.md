# AnamDB Python SDK

Python client for [AnamDB](https://github.com/AnamDB/anam-db) — the AI-native neurosymbolic database engine.

## Installation

```bash
pip install anamdb
```

For Arrow IPC support (decode query results into PyArrow tables):

```bash
pip install anamdb[arrow]
```

## Quick Start

```python
import asyncio
from anamdb import AnamClient

async def main():
    # Connect to a running AnamDB server
    async with AnamClient("127.0.0.1:8080") as client:
        # Check server health
        health = await client.health()
        print(f"Server: {health.status} (v{health.version})")

        # Register a table
        await client.register_table("txns", "/data/transactions.lance")

        # Register a Datalog rule
        await client.register_rule("high_risk", "fraud_prob > 0.90 AND amount > 10000")

        # Run a SQL query
        result = await client.query(
            "SELECT region, COUNT(1) AS count "
            "FROM txns WHERE fraud_prob > 0.90 "
            "GROUP BY region ORDER BY count DESC"
        )

        print(f"Rows: {result.row_count}")
        if result.reasoning_tree:
            print(f"Reasoning: {result.reasoning_tree}")

asyncio.run(main())
```

## API Reference

### `AnamClient(addr, *, connect_timeout=5.0, max_retries=3)`

Async context manager for connecting to an AnamDB server.

**Methods:**

| Method | Description |
|:---|:---|
| `query(sql)` | Execute a SQL query, returns `QueryResult` |
| `register_table(name, lance_path)` | Register a Lance dataset as a table |
| `register_rule(name, datalog)` | Register a Datalog rule |
| `load_model(name, version, path, ...)` | Load an ONNX model |
| `health()` | Server health check |

### `QueryResult`

| Field | Type | Description |
|:---|:---|:---|
| `row_count` | `int` | Number of rows returned |
| `reasoning_tree` | `str \| None` | Provenance reasoning trace |
| `anomalies` | `list[str]` | Semantic anomaly descriptions |
| `raw_response` | `dict` | Full JSON response from server |

### `ServerHealth`

| Field | Type | Description |
|:---|:---|:---|
| `status` | `str` | `"SERVING"` or `"NOT_SERVING"` |
| `version` | `str` | AnamDB server version |
| `table_count` | `int` | Number of registered tables |
| `model_count` | `int` | Number of loaded models |
| `rule_count` | `int` | Number of Datalog rules |

## License

Apache License 2.0
