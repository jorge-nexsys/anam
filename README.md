# AnamDB
### *The AI-Native Neurosymbolic Database Engine*

**AnamDB** is a vertical-agnostic, neurosymbolic database engine built in Rust. It natively integrates probabilistic neural perception with deterministic symbolic reasoning into a unified architecture — from a single-node kernel to a distributed multi-agent reasoning plane.

Unlike traditional vector databases that rely on semantic similarity or bolt-on LLMs, AnamDB treats **Models as First-Class Citizens** and **Logic as a Verifiable Blueprint**.

---

## Why AnamDB?

| Capability | AnamDB | Vector DBs | SQL + ML | LLM Pipelines |
|:---|:---|:---|:---|:---|
| **Explainability** | Semiring provenance — every result traced to source | Similarity score only | No lineage | Black box |
| **Safety** | Datalog guardrails block hallucinations at kernel level | None | Post-hoc validation | Prompt engineering |
| **Optimization** | Pareto frontier (latency × accuracy × cost) | Latency only | Latency only | Token cost only |
| **Hardware** | Metal / CUDA / NPU heterogeneous dispatch | CPU only | CPU + external GPU | API calls |
| **Models** | AI-Tables — first-class model lifecycle management | External | External endpoints | Hardcoded |
| **Human-in-Loop** | Semantic anomaly detection with interactive triage | Silent failures | Error logs | Chat-based retry |
| **Distribution** | Network-aware task routing + global lineage | Sharding only | Federated queries | N/A |

---

## Tech Stack

| Layer | Component | Technology |
|:---|:---|:---|
| **Kernel** | Async runtime | Rust 2024 + `tokio` |
| **Query Engine** | Optimizer + execution | Apache DataFusion (extended) |
| **Logic** | Differentiable Datalog | `scallop-core` |
| **Models** | AI-Tables + FAO registry | ONNX Runtime |
| **Storage** | Columnar + vector | Lance 2.2 (Arrow-backed) |
| **SDK** | Logic Packs + Explainer | JSON-based bundles |
| **Distribution** | Task routing + BCNF catalog | Multi-agent cluster |

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/your-org/anam.git && cd anam
cargo build

# Set up LLM (optional, for .nl command)
echo "OPENAI_API_KEY=sk-..." > .env

# Run the full demo
./demo/run_demo.sh

# Start the interactive REPL
cargo run --bin anam

# Or start the production-ready server
cargo run --bin anam -- serve --port 8080
```

### Community Hub

AnamDB includes a built-in package manager for models and logic:

```bash
# Search for community logic packs
anam hub search fraud

# Install the financial compliance pack
anam hub install anamdb/financial-compliance@1.0.0
```

### Interactive Session

```
anam> .ingest demo/data/transactions_large.csv demo/data/transactions_large.lance
✓ Ingested 100,000 rows

anam> .load demo/data/transactions_large.lance txns
Registered table 'txns'

anam> .model load demo/models/fraud_detector.onnx fraud_detector 3 5.0 0.95
✓ Loaded ONNX model 'fraud_detector'

anam> .logic high_risk "fraud_prob > 0.90 AND amount > 10000"
✓ Registered rule 'high_risk'

anam> SELECT region, COUNT(1) AS count, ROUND(AVG(fraud_prob), 4) AS avg_fraud
       FROM txns GROUP BY region ORDER BY avg_fraud DESC;
+--------+-------+-----------+
| region | count | avg_fraud |
+--------+-------+-----------+
| APAC   | 5321  | 0.7233    |
| EU     | 36033 | 0.1374    |
| US     | 48018 | 0.0800    |
+--------+-------+-----------+

anam> .explain
═══ AnamDB Reasoning Trace ═══
  Provenance: Polynomial (full lineage tracking)
  Rules: high_risk ← fraud_prob > 0.90 AND amount > 10000
  Pareto Frontier: fraud_fast (0.050ms / 75%) ★ fraud_detector (0.500ms / 95%)
```

### Rust SDK

```rust
use anamdb::sdk::LogicPack;
use anamdb::core::session::Session;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = Session::new().await?;

    // Load a domain-specific Logic Pack (rules + models in one JSON)
    let pack = LogicPack::from_file("demo/packs/financial_compliance.json")?;
    session.load_logic_pack(&pack)?;

    // Query with automatic Pareto optimization
    let batches = session.query("SELECT * FROM HighRisk").await?;

    // Explain results with provenance tracing
    let explanation = session.explain_query(&batches, ExplainLevel::Coarse)?;
    println!("{}", explanation.display());

    Ok(())
}
```

---

## Architecture

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                     AnamDB v1.0 Coordinator                      │
 │                                                                  │
 │  ┌─────────────┐ ┌──────────────┐ ┌──────────────────────────┐  │
 │  │ BCNF Policy │ │ Distributed  │ │ Global Lineage           │  │
 │  │ Catalog     │ │ Optimizer    │ │ Tracer                   │  │
 │  └──────┬──────┘ └──────┬───────┘ └──────────┬───────────────┘  │
 │         ▼               ▼                     ▼                  │
 │  ┌──────────────────────────────────────────────────────────┐    │
 │  │                    Task Router                            │    │
 │  │     Perception → Edge  |  Symbolic → Core  |  Mixed → Hybrid │
 │  └──────┬────────────────┬────────────────────┬─────────────┘    │
 └─────────┼────────────────┼────────────────────┼─────────────────┘
           │                │                    │
    ┌──────▼──────┐  ┌──────▼──────┐  ┌──────────▼──────┐
    │  Edge Node  │  │  Core Node  │  │  Hybrid Node    │
    │  NPU + 4GB  │  │  64GB RAM   │  │  GPU + 32GB     │
    │             │  │             │  │                  │
    │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────────┐ │
    │ │ 5-Stage │ │  │ │ 5-Stage │ │  │ │ 5-Stage     │ │
    │ │Pipeline │ │  │ │Pipeline │ │  │ │ Pipeline    │ │
    │ └─────────┘ │  │ └─────────┘ │  │ └─────────────┘ │
    └─────────────┘  └─────────────┘  └──────────────────┘
```

---

## Documentation

| Document | Description |
|:---|:---|
| [DEMO.md](DEMO.md) | Full 17-step walkthrough with output examples |
| [SPECS.md](SPECS.md) | System specifications (Alpha → Beta → v1.0) |
| [ROADMAP.md](ROADMAP.md) | Development roadmap and status |

---

## Roadmap

- [x] **Alpha** — Kernel: Datalog engine, Lance storage, Semiring provenance, AI-Tables, Pareto optimizer, heterogeneous dispatch.
- [x] **Beta** — Developer Experience: Logic Pack SDK, NL-to-Datalog compilation, syntactic self-repair, query explainer, interactive triage.
- [x] **v1.0** — Distributed Reasoning Plane: 5-stage pipeline, BCNF policy catalog, multi-agent task routing, network-aware optimizer, global lineage.
- [x] **Engine Integration** — Native DataFusion UDFs for models, streaming table providers, persistent SQLite catalog.
- [x] **Server & SDKs** — gRPC/JSON-over-TCP wire protocol, PyO3 Python SDK, native Rust SDK.
- [x] **Live Demo** — Web-based Interactive Playground with WebSockets.
- [x] **Ecosystem Hub** — Built-in `hub` package manager, spatial/temporal audio abstractions, model distillation engine.
- [x] **Commercial Readiness** — API key auth middleware, Token-Bucket rate limiting, Stripe-ready usage metering, BSL 1.1 license.

---

## Test Suite

```
$ cargo test

test result: ok. 38 passed; 0 failed; 0 ignored
```

## License

AnamDB is licensed under the **Business Source License 1.1 (BSL)**. 

It is completely free to use for development, evaluation, and startups with under $1M in annual revenue. The license automatically converts to **Apache 2.0** after 4 years from each release. See [LICENSE](LICENSE) for details.

---

<p align="center">
<b>Every other database stores data. AnamDB reasons about it.</b>
</p>
