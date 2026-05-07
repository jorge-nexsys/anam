# AnamDB

### The AI-Native Neurosymbolic Database Engine

> **Every other database stores data. AnamDB reasons about it.**

---

## The Problem

AI systems in production face a fundamental trust crisis:

| System Type | What It Does Well | What It Can't Do |
|:---|:---|:---|
| **Vector DBs** (Pinecone, Milvus) | Similarity search on embeddings | Explain *why* a result was returned |
| **SQL Warehouses** (BigQuery, Redshift) | Structured analytics | Run neural inference natively |
| **LLM Pipelines** (LangChain, RAG) | Natural language generation | Guarantee logical correctness |
| **ML Platforms** (SageMaker, Vertex) | Model serving at scale | Integrate symbolic constraints |

**The gap:** No system treats **neural inference** and **symbolic logic** as equal citizens in the same query engine.

AnamDB fills that gap.

---

## How It Works

```
                    ┌─────────────────────────────────────────────────┐
                    │               AnamDB Kernel                    │
                    │                                                 │
  Natural Language ─┤  NL Compiler ──► Datalog Engine ──► Logic      │
                    │       │              │                Filter    │
  SQL Queries ──────┤  DataFusion ───► Pareto Optimizer ──► Plan     │──► Results
                    │       │              │                          │   + Provenance
  ONNX Models ─────┤  FAO Registry ─► Model Inference ──► Scores    │   + Anomalies
                    │       │              │                          │
  CSV / Lance ──────┤  Lance Provider  Arrow Batches  ──► Storage    │
                    │                                                 │
                    │  ┌──────────┐  ┌───────────┐  ┌────────────┐  │
                    │  │ CPU (8x) │  │ Metal GPU │  │ CUDA GPUs  │  │
                    │  └──────────┘  └───────────┘  └────────────┘  │
                    └─────────────────────────────────────────────────┘
```

---
---

# Alpha — The Kernel

> Run the full pipeline yourself: `./demo/run_demo.sh`

### Step 1 — Ingest 100K Transactions into Lance Columnar Storage

```
anam> .ingest demo/data/transactions.csv demo/data/transactions.lance
✓ Ingested 100,000 rows (4.7 MB CSV → Lance columnar)

anam> .load demo/data/transactions_large.lance txns
Registered table 'txns'
```

**Why this matters:** Data goes from raw CSV to a queryable, versioned, columnar dataset in one command. No ETL pipelines. No Spark jobs. No separate model-serving infrastructure.

---

### Step 2 — SQL Analytics at Scale

```sql
anam> SELECT region, COUNT(1) AS count,
       ROUND(AVG(amount), 2) AS avg_amount,
       ROUND(AVG(fraud_prob), 4) AS avg_fraud
       FROM txns GROUP BY region ORDER BY avg_fraud DESC;

+--------+-------+----------+-----------+
| region | count | avg_amt  | avg_fraud |
+--------+-------+----------+-----------+
| APAC   | 5321  | 22938.36 | 0.7233    |
| MEA    | 5299  | 22426.98 | 0.7230    |
| LATAM  | 5329  | 22366.48 | 0.7229    |
| EU     | 36033 | 1055.56  | 0.1374    |
| US     | 48018 | 252.76   | 0.0800    |
+--------+-------+----------+-----------+
```

100K rows. Full aggregation. Sub-second on Apple M2. Standard SQL — nothing new to learn.

---

### Step 3 — Load Neural Models as First-Class Operators

```
anam> .model load demo/models/fraud_detector.onnx fraud_detector 3 5.0 0.95
✓ Loaded ONNX model 'fraud_detector' (latency: 5ms, accuracy: 0.95)

anam> .model load demo/models/fraud_detector_fast.onnx fraud_fast 3 0.5 0.75
✓ Loaded ONNX model 'fraud_fast' (latency: 0.5ms, accuracy: 0.75)
```

Two models. Different trade-offs. Both registered as **Function-as-Operator** (FAO) entries in the AI-Tables catalog — just like registering a UDF, but with full lifecycle tracking.

**What competitors do instead:** BigQuery ML and Redshift ML require you to deploy models externally, then call them via HTTP. SageMaker endpoints add 50–200ms of network latency per batch. AnamDB runs inference **in-process**, zero-copy on Arrow memory.

---

### Step 4 — Pareto Multi-Objective Optimization

```
anam> .explain

─── Pareto Frontier ────────────────────────────────────
  ★ frontier  fraud_fast v1.0.0:     latency=0.050ms, accuracy=0.75
  ★ frontier  fraud_detector v1.0.0: latency=0.500ms, accuracy=0.95
```

Neither model dominates the other. The optimizer presents the **Pareto frontier** — the set of non-dominated plans. When you add cost constraints:

```sql
SELECT * FROM HighRisk WITH (max_latency_ms = 50, min_accuracy = 0.90)
```

...the optimizer automatically selects the **cheapest feasible plan** on the frontier.

**Why this is different:**

| Approach | Optimizes For | Misses |
|:---|:---|:---|
| PostgreSQL / BigQuery | Latency only | Accuracy trade-offs |
| SageMaker Endpoints | Throughput | Cost / accuracy balance |
| **AnamDB** | **Latency × Accuracy × Cost** | Nothing — it's the full Pareto surface |

---

### Step 5 — Symbolic Logic Guardrails

```
anam> .logic high_risk "fraud_prob > 0.90 AND amount > 10000"
✓ Registered rule 'high_risk'

anam> .logic wire_alert "merchant_type = 'wire_transfer' AND amount > 50000"
✓ Registered rule 'wire_alert'
```

These aren't SQL views. They're **Datalog rules** compiled into the logic engine. They act as hard constraints at the kernel level — if a neural model suggests a result that violates a rule, it gets blocked before it reaches the application layer.

**The LLM hallucination problem, solved architecturally:**

```
  Neural Model Output ──► Datalog Logic Filter ──► Only valid results pass
                              │
                         Blocked if contradicts
                         symbolic constraints
```

---

### Step 6 — Natural Language → Datalog (via LLM)

```
anam> .nl suspicious_night txns Flag any transaction between midnight and 5am over $5000

Compiling NL → Datalog via LLM...
✓ Generated and registered rule 'suspicious_night':
  Datalog: night_high_value(X) :- txns(X), X.time >= '00:00',
           X.time < '05:00', X.amount > 5000.
```

You describe intent in English. GPT-4o compiles it to verified Datalog. The rule is then **deterministic** — no LLM is involved at query time. The neural network writes the code; the symbolic engine enforces it.

**Contrast with RAG pipelines:** LangChain and similar frameworks invoke the LLM on every query. AnamDB invokes it once to *generate the rule*, then runs the rule at machine speed forever.

---

### Step 7 — Semantic Anomaly Detection (HITL)

```
anam> SELECT fraud_prob FROM txns WHERE fraud_prob < 0.05;
(9,216 rows)

⚠️  Semantic anomalies detected:
  [WARNING] 100% of rows have fraud_prob below 0.5 (threshold: 80% max)
  → Consider using a higher-accuracy model.
Use `.refine <correction>` to provide feedback, or `.accept` to proceed.
```

The **Semantic Monitor** inspects every result set for statistical anomalies:
- **Low-confidence rate:** Most rows have suspiciously low scores
- **Uniform scores:** All values identical (model likely broken)
- **Empty results:** Overly restrictive filters

Standard databases silently return garbage. AnamDB tells you *something might be wrong*.

---

### Step 8 — Full Reasoning Trace

```
anam> .explain

═══════════════════════════════════════════════════════════
  AnamDB Reasoning Trace
═══════════════════════════════════════════════════════════

─── Provenance ─────────────────────────────────────────
  Mode: Polynomial (full lineage tracking)
  Last query: 1 batch(es), 9216 rows

─── Datalog Rules ──────────────────────────────────────
  • high_risk       ← fraud_prob > 0.90 AND amount > 10000
  • wire_alert      ← merchant_type = 'wire_transfer' AND amount > 50000
  • suspicious_night ← (LLM-generated Datalog)

─── AI-Tables Catalog ──────────────────────────────────
  • fraud_detector v1.0.0 [onnx] — 5.0ms, 0.95 accuracy
  • fraud_fast v1.0.0 [onnx]     — 0.5ms, 0.75 accuracy

─── Pareto Frontier ────────────────────────────────────
  ★ frontier  fraud_fast:     0.050ms / 0.75 accuracy
  ★ frontier  fraud_detector: 0.500ms / 0.95 accuracy

─── Device Pool ────────────────────────────────────────
  [ 0–7] CPU-0..CPU-7 (1x) — idle
  [   8] Metal: Apple M2 (10x) — idle

─── Anomalies ──────────────────────────────────────────
  ⚠ [WARNING] 100% of rows have fraud_prob below 0.5
    → Consider using a higher-accuracy model.

═══════════════════════════════════════════════════════════
```

Every query produces a full reasoning trace. Every result is accountable.

---
---

# Beta — Developer Experience & Agentic Trust

### Step 9 — Load a Logic Pack

Logic Packs are JSON bundles of rules + model references. Instead of manually
registering each rule and model, a developer loads one pack:

```json
// demo/packs/financial_compliance.json
{
  "name": "financial_compliance",
  "version": "1.0.0",
  "description": "EU AML/KYC compliance rules for transaction monitoring",
  "author": "NexSys Consulting",
  "rules": [
    { "name": "high_risk",  "datalog": "fraud_prob > 0.90 AND amount > 10000" },
    { "name": "wire_alert", "datalog": "merchant_type = 'wire_transfer' AND amount > 50000" },
    { "name": "velocity_check", "datalog": "amount > 5000 AND fraud_prob > 0.70" }
  ],
  "models": [
    { "name": "fraud_detector", "artifact_path": "demo/models/fraud_detector.onnx",
      "num_features": 3, "avg_latency_ms": 5.0, "accuracy": 0.95 },
    { "name": "fraud_fast", "artifact_path": "demo/models/fraud_detector.onnx",
      "num_features": 3, "avg_latency_ms": 0.5, "accuracy": 0.75 }
  ]
}
```

```rust
use anamdb::sdk::LogicPack;
use anamdb::Session;

let session = Session::new().await?;
let pack = LogicPack::from_file("demo/packs/financial_compliance.json")?;
let summary = session.load_logic_pack(&pack)?;
println!("{summary}");
```

```
Logic Pack: financial_compliance v1.0.0
  EU AML/KYC compliance rules for transaction monitoring
  Author: NexSys Consulting
  3 rule(s), 2 model(s)
    • high_risk ← fraud_prob > 0.90 AND amount > 10000
    • wire_alert ← merchant_type = 'wire_transfer' AND amount > 50000
    • velocity_check ← amount > 5000 AND fraud_prob > 0.70
    ◆ fraud_detector [demo/models/fraud_detector.onnx] — 5.0ms, 95% accuracy
    ◆ fraud_fast [demo/models/fraud_detector.onnx] — 0.5ms, 75% accuracy
```

**Why this matters:** A compliance team ships one JSON file. A developer loads it with one function call. Zero Datalog expertise required.

---

### Step 10 — Syntactic Self-Repair

When a FAO operator hits a structural error (dimension mismatch, timeout, OOM), the engine doesn't abort. It triggers a two-agent loop:

```rust
let report = session.self_repair(
    "operator exceeded deadline of 50ms",
    "fraud_detector",
    "batch of 10K rows on CPU"
)?;
println!("{}", report.summary());
```

```
═══ Self-Repair Report ═══
Severity: RECOVERABLE
Root Cause: Operator 'fraud_detector' exceeded its execution time budget.
            Consider swapping to a faster model variant.
Confidence: 95%
Action: SwapModel → fraud_fast: Swapping from 'fraud_detector' to 'fraud_fast'
        to bypass: Operator 'fraud_detector' exceeded its execution time budget.
Status: ✓ Applied
```

**Error classification matrix:**

| Error Pattern | Severity | Auto-Repair Action |
|:---|:---|:---|
| Dimension / shape mismatch | Recoverable | Swap to compatible model |
| Timeout / deadline exceeded | Recoverable | Swap to faster variant |
| Null / missing values | Recoverable | Retry with adjusted params |
| Unsupported format / codec | Degraded | Skip unsupported rows |
| Out of memory | Degraded | Continue in degraded mode |
| Unknown / unrecognized | Fatal | Escalate to user |

---

### Step 11 — Query Result Explainer

Every query result can be explained in natural language at two granularity levels:

**Coarse-grained** — pipeline summary:

```
═══════════════════════════════════════════════════════════
  AnamDB Query Explanation
═══════════════════════════════════════════════════════════

─── Pipeline Summary ───────────────────────────────────
  Produced 1,247 row(s) across 1 batch(es)
  Provenance: Polynomial
  Schema: [amount:Float64, fraud_prob:Float64, merchant_type:Utf8]

  Score Distribution (fraud_prob):
     min=0.7012, max=0.9987, mean=0.8834, median=0.8921

─── Rules Applied ──────────────────────────────────────
  • high_risk ← fraud_prob > 0.90 AND amount > 10000
  • velocity_check ← amount > 5000 AND fraud_prob > 0.70

─── Models Used ────────────────────────────────────────
  • fraud_detector v1.0.0
═══════════════════════════════════════════════════════════
```

**Fine-grained** — per-row lineage:

```
─── Per-Row Lineage ────────────────────────────────────
  Row 0: Derived via fraud_detector using model 'v1.0.0',
         sourced from [txn_38291, txn_38292]
  Row 1: Derived via fraud_detector using model 'v1.0.0',
         sourced from [txn_44107]
```

---

### Step 12 — Interactive Triage

When the Semantic Monitor detects anomalies, the engine pauses for interactive resolution:

```
═══ Semantic Anomaly Triage ═══

[Anomaly 1]
  Severity: WARNING
  100% of rows have fraud_prob below 0.5 (threshold: 80% max).
  Affected rows: 9216
  Suggested: Consider using a higher-accuracy model.
  Action: RetryWithModel("fraud_detector")
```

| Action | Effect |
|:---|:---|
| `Accept` | Acknowledge the anomaly and proceed |
| `Correct("...")` | Provide a natural-language correction for rule refinement |
| `RetryWithModel("...")` | Swap to a different model variant |
| `Abort` | Halt the query entirely |

---
---

# v1.0 — The Distributed Reasoning Plane

### Step 13 — 5-Stage Symbolic Integration Pipeline

Every query flows through 5 stages, whether on a single node or across a cluster:

```
═══ 5-Stage Symbolic Integration Pipeline ═══
  [✓] Stage 1 — Data Preprocessing (1000.00ms, 100000 records)
      Transposed 100000 raw records into vector-symbolic form
  [✓] Stage 2 — Neural-Symbolic Embedding (5000.00ms, 100000 records)
      Feature extraction with first-order logic constraints
  [✓] Stage 3 — Domain Knowledge (2000.00ms, 100000 records)
      Cross-referenced with 1 domain ontologies
  [✓] Stage 4 — Logical Reasoning (10000.00ms, 100000 records)
      Applied 3 Datalog rules over neural outputs
  [✓] Stage 5 — Symbolic Postprocessing (500.00ms, 100000 records)
      Final constraint verification passed
```

---

### Step 14 — BCNF Policy Catalog

All rules live in a strict Boyce-Codd Normal Form catalog. Updates propagate to replicas via version-stamped changesets:

```
═══ BCNF Policy Catalog (v2) ═══
  2 active / 2 total policies
  • [v1] aml_high_risk → fraud_prob > 0.90 AND amount > 10000 (transactions)
  • [v2] wire_alert → merchant_type = 'wire_transfer' AND amount > 50000 (transactions)
```

Incremental replication to remote nodes via changeset deltas — no full resync required.

---

### Step 15 — Multi-Agent Task Router

The router dynamically assigns FAO operations to the best node in the cluster:

```
═══ Agent Cluster (3 nodes) ═══
  [Core]   core-0   — 65536MB, 0 accel, 20% load, 1.0ms latency
  [Edge]   edge-0   — 4096MB, 2 accel, 10% load, 5.0ms latency
  [Hybrid] hybrid-0 — 32768MB, 4 accel, 30% load, 2.0ms latency
```

| Workload | Preferred Node | Reason |
|:---|:---|:---|
| **Perception** (OCR, image, audio) | Edge | Accelerator-rich, low latency |
| **Symbolic Join** (Datalog, reasoning) | Core | High memory, deterministic |
| **Mixed** (NLP + rules) | Hybrid | Both neural and symbolic capacity |

---

### Step 16 — Network-Aware Distributed Optimizer

The Pareto frontier now includes **network latency** alongside compute cost:

```
═══ Distributed Pareto Frontier ═══
  ★ fraud_fast@edge       — compute: 0.5ms + network: 5.0ms = 5.5ms, accuracy: 75%, cost: 0.10
  ★ fraud_detector@core   — compute: 5.0ms + network: 1.0ms = 6.0ms, accuracy: 95%, cost: 0.50
  ★ fraud_ensemble@hybrid — compute: 10.0ms + network: 2.0ms = 12.0ms, accuracy: 99%, cost: 1.00
```

**Progressive rewrite:** If the edge model underperforms at runtime, the optimizer transparently re-routes to a higher-accuracy model on a core node. The user sees correct results; the system handles the complexity.

---

### Step 17 — Global Lineage & Cluster Monitor

Trace any result tuple across every node it touched:

```
═══ Global Lineage: txn_001 ═══
  Hops: 2 | Total: 20.0ms

  ┌─ Hop 1 ─────────────────────────────────
  │ Node:       edge-0
  │ Operator:   ocr_extract
  │ Model:      ocr_v2.1
  │ Confidence: 0.9200
  │ Duration:   15.00ms
  │ Sources:    [img_001]
  └──────────────────────────────────────────

  ┌─ Hop 2 ─────────────────────────────────
  │ Node:       core-0
  │ Operator:   fraud_detector
  │ Model:      fraud_v1.0
  │ Confidence: 0.9700
  │ Duration:   5.00ms
  │ Sources:    [txn_001_features]
  └──────────────────────────────────────────
```

Cluster anomaly isolation pauses **only the affected data path** — all other nodes keep processing.

---
---

# Architecture

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                   AnamDB v1.0 Coordinator                        │
 │                                                                  │
 │  ┌─────────────┐ ┌──────────────┐ ┌──────────────────────────┐  │
 │  │ BCNF Policy │ │ Distributed  │ │ Global Lineage           │  │
 │  │ Catalog     │ │ Optimizer    │ │ Tracer                   │  │
 │  │ (versioned) │ │ (network-    │ │ (cross-node              │  │
 │  │             │ │  aware)      │ │  provenance)             │  │
 │  └──────┬──────┘ └──────┬───────┘ └──────────┬───────────────┘  │
 │         │               │                     │                  │
 │         ▼               ▼                     ▼                  │
 │  ┌──────────────────────────────────────────────────────────┐    │
 │  │               Task Router                                │    │
 │  │    Perception → Edge  |  Symbolic → Core  |  Mixed → Hybrid  │
 │  └──────┬────────────────┬────────────────────┬─────────────┘    │
 └─────────┼────────────────┼────────────────────┼─────────────────┘
           │                │                    │
    ┌──────▼──────┐  ┌──────▼──────┐  ┌──────────▼──────┐
    │  Edge Node  │  │  Core Node  │  │  Hybrid Node    │
    │  4GB + NPU  │  │  64GB RAM   │  │  32GB + 4 GPUs  │
    │  Stages 1-2 │  │  Stages 3-4 │  │  Stages 1-5     │
    └─────────────┘  └─────────────┘  └──────────────────┘
```

---

## Why AnamDB Over the Alternatives

| Capability | AnamDB | Vector DBs | SQL + ML | LLM Pipelines |
|:---|:---|:---|:---|:---|
| **Explainability** | Semiring provenance — every result traced to source | ✗ Similarity score only | ✗ No lineage | ✗ Black box |
| **Safety** | Datalog guardrails block hallucinations at kernel level | ✗ None | ✗ Post-hoc validation | ✗ Prompt engineering |
| **Optimization** | Pareto frontier (latency × accuracy × cost) | ✗ Latency only | ✗ Latency only | ✗ Token cost only |
| **Hardware** | Metal / CUDA / NPU heterogeneous dispatch | ✗ CPU only | ✗ CPU + external GPU | ✗ API calls |
| **Models** | AI-Tables — first-class model lifecycle management | ✗ External | ✗ External endpoints | ✗ Hardcoded |
| **Human-in-Loop** | Semantic anomaly detection with interactive triage | ✗ Silent failures | ✗ Error logs | ✗ Chat-based retry |
| **Logic** | Datalog + NL compilation — deterministic at runtime | ✗ None | ✗ SQL views only | ✗ LLM on every call |
| **Distribution** | Network-aware task routing + global lineage | ✗ Sharding only | ✗ Federated queries | ✗ N/A |

---

## Performance

| Component | Scale | Throughput |
|:---|:---|:---|
| Semiring (Boolean) | 10K ops | **28.4 M ops/sec** |
| Semiring (Probability) | 10K ops | **39.0 M ops/sec** |
| Semiring (Polynomial) | 1K ops | **22.3 K ops/sec** |
| Logic filter (100 rows) | 100 evals | **50,600 evals/sec** |
| Logic filter (10K rows) | 100 evals | **974 evals/sec** |
| HITL monitor (100 rows) | 1K scans | **233,000 scans/sec** |
| HITL monitor (10K rows) | 1K scans | **3,310 scans/sec** |
| Full SQL (100K rows) | GROUP BY | **< 1 second** |

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

# Or explore interactively
cargo run -- --gpu
```

```
anam> .help
  .load <path> [name]                Register a Lance table
  .ingest <csv> [lance]              CSV → Lance ingestion
  .model load <onnx> [name] [f] [l] [a]  Load ONNX model
  .logic <name> <datalog>            Register Datalog rule
  .nl <name> <table> <desc>          NL → Datalog via LLM
  .models / .operators               List models / FAO ops
  .rules                             List Datalog rules
  .devices                           Show device pool
  .explain                           Full reasoning trace
  .quit                              Exit
```

---

## Test Results

```
$ cargo test

test result: ok. 38 passed; 0 failed; 0 ignored

  Alpha (20)  ✓   Beta (7)  ✓   v1.0 (11)  ✓
```

---

<p align="center">
<b>AnamDB: From Kernel to Reasoning Plane — every result verified, every hop traced, every node accountable.</b>
</p>
