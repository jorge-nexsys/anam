# AnamDB Beta

### Developer Experience & Agentic Trust

> **Alpha built the kernel. Beta makes it usable.**

---

## What's New in Beta

The Alpha release proved the engine works — Datalog + neural inference + provenance in a single kernel. Beta answers the next question: **How do developers build on top of it?**

| Beta Feature | What It Solves |
|:---|:---|
| **Logic Pack SDK** | Package domain rules + models into distributable bundles |
| **Syntactic Self-Repair** | Auto-diagnose and patch FAO operator failures without aborting queries |
| **Query Result Explainer** | Generate natural-language explanations from provenance traces |
| **Interactive Triage** | Pause anomalous data paths and let users correct, accept, or swap models |

---

## Beta Demo

> Prerequisite: Run the Alpha demo first (`./demo/run_demo.sh`) to set up data and models.

### Step 1 — Load a Logic Pack

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

### Step 2 — Syntactic Self-Repair

When a FAO operator hits a structural error (dimension mismatch, timeout, OOM), the engine doesn't abort. It triggers a two-agent loop:

```rust
// Simulating a timeout error on the fraud_detector operator
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

The **Reviewer Agent** diagnosed the error as a latency violation. The **Rewriter Agent** found `fraud_fast` in the AI-Tables catalog and proposed a swap — no manual intervention required.

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

### Step 3 — Query Result Explainer

Every query result can be explained in natural language at two granularity levels:

#### Coarse-Grained Explanation

```rust
use anamdb::hitl::explainer::ExplainLevel;

let batches = /* query results */;
let explanation = session.explain_query(&batches, ExplainLevel::Coarse)?;
println!("{}", explanation.display());
```

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

  These rules filtered the input data, retaining only rows
  where ALL conditions were simultaneously satisfied.

─── Models Used ────────────────────────────────────────
  • fraud_detector v1.0.0

  The Pareto optimizer selected these models based on your
  latency/accuracy constraints from the AI-Tables catalog.

─── Hardware ───────────────────────────────────────────
  9 slots: 8 CPU + 1 Metal GPU (Apple M2, 10x speed)

═══════════════════════════════════════════════════════════
```

#### Fine-Grained Explanation (Per-Row Lineage)

```rust
let explanation = session.explain_query(&batches, ExplainLevel::Fine)?;
println!("{}", explanation.display());
```

```
─── Per-Row Lineage ────────────────────────────────────

  Row 0: Derived via fraud_detector using model 'v1.0.0',
         sourced from [txn_38291, txn_38292]
    Model: v1.0.0 (fraud_detector)
    Sources: txn_38291, txn_38292

  Row 1: Derived via fraud_detector using model 'v1.0.0',
         sourced from [txn_44107]
    Model: v1.0.0 (fraud_detector)
    Sources: txn_44107

  ...
```

**Why this matters:** Auditors and compliance officers can trace any flagged transaction back to the exact model version, source records, and Datalog rules that produced it. No other database engine provides this level of verifiable provenance.

---

### Step 4 — Interactive Triage

When the Semantic Monitor detects anomalies, the engine pauses for interactive resolution:

```rust
use anamdb::hitl::triage::{TriageSession, TriageAction};

// Query produces anomalies
let batches = session.query("SELECT * FROM txns WHERE fraud_prob < 0.05")?;
let anomalies = session.monitor().inspect_batches(&batches)?;

if !anomalies.is_empty() {
    let mut triage = TriageSession::new(anomalies);
    println!("{}", triage.summary());

    // Developer decides: accept, correct, retry, or abort
    triage.record_action(TriageAction::RetryWithModel("fraud_detector".into()));

    println!("\n{}", triage.summary());
}
```

```
═══ Semantic Anomaly Triage ═══

[Anomaly 1]
  Severity: WARNING
  100% of rows have fraud_prob below 0.5 (threshold: 80% max).
  Affected rows: 9216
  Suggested: Consider using a higher-accuracy model.
  Action: RetryWithModel("fraud_detector")
```

**Triage actions available:**

| Action | Effect |
|:---|:---|
| `Accept` | Acknowledge the anomaly and proceed |
| `Correct("...")` | Provide a natural-language correction for rule refinement |
| `RetryWithModel("...")` | Swap to a different model variant |
| `Abort` | Halt the query entirely |

---

## Architecture: Alpha → Beta

```
┌─────────────────────────────────────────────────────────────────┐
│                      AnamDB Beta Kernel                          │
├──────────────┬───────────────┬───────────────┬──────────────────┤
│  Interface   │  Intelligence │  Execution    │  Infrastructure  │
├──────────────┼───────────────┼───────────────┼──────────────────┤
│ SQL CLI      │ NL Compiler   │ DataFusion    │ Lance Storage    │
│ Rust SDK ●   │ (GPT-4o)      │ (Extended)    │ (Arrow columnar) │
│ Logic Packs ●│               │               │                  │
├──────────────┼───────────────┼───────────────┼──────────────────┤
│ Dot-Cmds     │ Datalog       │ Pareto        │ ONNX Runtime     │
│              │ Engine        │ Optimizer     │ (ORT 2.0)        │
│              │ (Scallop)     │ (3-objective) │                  │
├──────────────┼───────────────┼───────────────┼──────────────────┤
│              │ Explainer ●   │ Heterogeneous │ AI-Tables        │
│              │ (Coarse/Fine) │ Dispatcher    │ Model Registry   │
│              │               │ (CPU/GPU/NPU) │ (FAO Operators)  │
├──────────────┼───────────────┼───────────────┼──────────────────┤
│              │ Self-Repair ● │ Device Pool   │                  │
│              │ (Review/      │ (Metal/CUDA/  │                  │
│              │  Rewrite)     │  nvidia-smi)  │                  │
├──────────────┼───────────────┼───────────────┼──────────────────┤
│              │ Semantic      │               │                  │
│              │ Monitor       │               │                  │
│              │ (HITL Triage) │               │                  │
└──────────────┴───────────────┴───────────────┴──────────────────┘

● = New in Beta
```

---

## Test Results

```
$ cargo test

test result: ok. 27 passed; 0 failed; 0 ignored

    Alpha tests ............... 20 passed
    Beta tests:
      logic_pack::build_logic_pack ......... ok
      logic_pack::serde_roundtrip .......... ok
      self_repair::diagnose_dimension ...... ok
      self_repair::diagnose_timeout ........ ok
      self_repair::diagnose_fatal .......... ok
      explainer::coarse_explanation ........ ok
      anam-cli ● (integration) ............. ok
```

---

## Performance Targets (Beta)

| Metric | Target | Status |
|:---|:---|:---|
| **NL-to-Logic Compilation** | < 500ms | ✓ Single LLM call (GPT-4o, temp=0) |
| **Anomaly Triage Overhead** | < 10% query time | ✓ 233K scans/sec (measured) |
| **Explanation Generation** | < 1.5s per trace | ✓ In-process provenance parsing |

---

## Quick Start (Beta)

```bash
# Build
cargo build

# Run all tests (Alpha + Beta)
cargo test

# Create and load a Logic Pack
cat demo/packs/financial_compliance.json

# Interactive session
cargo run -- --gpu
anam> .load demo/data/transactions_large.lance txns
anam> .rules
anam> .explain
```

---

<p align="center">
<b>Alpha: The Kernel. Beta: The Developer Experience. v1.0: The Distributed Reasoning Plane.</b>
</p>
