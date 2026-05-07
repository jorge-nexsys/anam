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

## Live Demo

> Run the full pipeline yourself: `./demo/run_demo.sh`

### Step 1 — Ingest 100K Transactions into Lance Columnar Storage

```
anam> .ingest demo/data/transactions_large.csv demo/data/transactions_large.lance
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

## Performance

Measured on Apple M2 (8-core CPU + Metal GPU), `cargo test bench_quick`:

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

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        AnamDB Kernel                           │
├────────────┬──────────────┬──────────────┬────────────────────┤
│  Interface │  Intelligence│  Execution   │  Infrastructure    │
├────────────┼──────────────┼──────────────┼────────────────────┤
│ SQL CLI    │ NL Compiler  │ DataFusion   │ Lance Storage      │
│ Rust API   │ (GPT-4o)     │ (Extended)   │ (Arrow columnar)   │
│ .env Config│              │              │                    │
├────────────┼──────────────┼──────────────┼────────────────────┤
│ Dot-Cmds   │ Datalog      │ Pareto       │ ONNX Runtime       │
│ (.load,    │ Engine       │ Optimizer    │ (ORT 2.0)          │
│  .model,   │ (Scallop)    │ (3-objective)│                    │
│  .logic,   │              │              │                    │
│  .nl)      │              │              │                    │
├────────────┼──────────────┼──────────────┼────────────────────┤
│            │ Semantic     │ Heterogeneous│ AI-Tables          │
│            │ Monitor      │ Dispatcher   │ Model Registry     │
│            │ (HITL Triage)│ (CPU/GPU/NPU)│ (FAO Operators)    │
├────────────┼──────────────┼──────────────┼────────────────────┤
│            │ Provenance   │ Device Pool  │                    │
│            │ (Semiring    │ (Metal/CUDA/ │                    │
│            │  Polynomial) │  nvidia-smi) │                    │
└────────────┴──────────────┴──────────────┴────────────────────┘
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

<p align="center">
<b>AnamDB: Turning Probabilistic AI into Deterministic Infrastructure.</b>
</p>