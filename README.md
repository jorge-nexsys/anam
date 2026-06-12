<div align="center">

<img src="./assets/full_logo.png" alt="AnamDB Logo" width="480">

<b>AnamDB</b>: a neurosymbolic database engine for AI agents, fraud detection, and production reasoning. Built from scratch in Rust.
<br/><br/>
<h3>
  <a href="https://anamdb.github.io/anam-db/">docs</a> |
  <a href="https://crates.io/crates/anamdb">crates.io</a> |
  <a href="https://pypi.org/project/anamdb">pypi</a>
</h3>

[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://anamdb.github.io/anam-db/)
[![crates.io](https://img.shields.io/crates/v/anamdb)](https://crates.io/crates/anamdb)
[![PyPI](https://img.shields.io/pypi/v/anamdb)](https://pypi.org/project/anamdb)
[![GitHub Repo stars](https://img.shields.io/github/stars/AnamDB/anam-db)](https://github.com/AnamDB/anam-db/stargazers)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

</div>

<hr>

AnamDB is a database engine that unifies SQL queries, Datalog logic rules, and ONNX model inference in a single kernel. Every result carries a provenance trace — a full lineage of which data, rules, and model versions produced it.

You don't need a separate vector database, ML serving layer, rules engine, or observability pipeline. AnamDB gives your agents explainable, guardrailed access to data with model inference built in — not bolted on.

## Getting Started

### 1. Install the CLI

```bash
cargo install anam-cli
```

Or build from source:

```bash
cargo install --git https://github.com/AnamDB/anam-db anam-cli
```

### 2. Initialize a project

```bash
mkdir my-project && cd my-project
anam init
```

This scaffolds `anamdb.toml`, example queries, environment templates, and directory structure.

### 3. Start the server

```bash
anam start
```

Or launch the interactive REPL:

```bash
anam
```

### 4. Run your first query

```sql
-- Load a Lance dataset
anam> .load /path/to/transactions.lance txns

-- Register a Datalog guardrail
anam> .logic high_risk "fraud_prob > 0.90 AND amount > 10000"

-- Query with model inference + logic filtering
anam> SELECT region, COUNT(1) AS count, ROUND(AVG(fraud_prob), 4) AS avg_fraud
       FROM txns WHERE fraud_prob > 0.90 AND amount > 10000
       GROUP BY region ORDER BY avg_fraud DESC;

-- See exactly WHY the engine made those decisions
anam> .explain
```

```
═══ AnamDB Reasoning Trace ═══
  Provenance: Polynomial (full lineage tracking)
  Rules: high_risk ← fraud_prob > 0.90 AND amount > 10000
  Pareto Frontier: fraud_fast (0.050ms / 75%) ★ fraud_detector (0.500ms / 95%)
```

## Alternative installs

### Python SDK

```bash
pip install anamdb
```

```python
from anamdb import AnamClient

async with AnamClient("127.0.0.1:8080") as client:
    result = await client.query("SELECT * FROM txns LIMIT 10")
    health = await client.health()
```

### Docker

```bash
docker run -p 8080:8080 ghcr.io/anamdb/anam-db
```

Or with Docker Compose:

```bash
git clone https://github.com/AnamDB/anam-db && cd anam-db
docker compose up
```

### Rust library

```toml
[dependencies]
anamdb = "1.0"
```

## How it works

AnamDB's 5-stage pipeline processes every query through:

1. **Parse** — SQL is parsed and extended with model-aware operators
2. **Logic** — Datalog rules are compiled into query filters via differentiable semiring evaluation
3. **Optimize** — Pareto frontier selects the best model across latency, accuracy, and cost
4. **Execute** — Apache DataFusion runs the query with ONNX model inference inlined as UDFs
5. **Explain** — Semiring provenance traces every output row to source records, rules, and model versions

| Layer | Technology |
|:---|:---|
| Kernel | Rust 2024 + Tokio |
| Query engine | Apache DataFusion |
| Logic | Differentiable Datalog (semiring provenance) |
| Models | ONNX Runtime — CPU, CUDA, Metal, NPU |
| Storage | Lance (Arrow-backed columnar + vector) |
| Wire protocol | JSON-over-TCP |

## Status

AnamDB is in active development. The core engine is functional and tested:

```
$ cargo test
test result: ok. 54 passed; 0 failed; 0 ignored
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
