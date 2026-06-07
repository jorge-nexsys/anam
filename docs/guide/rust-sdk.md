# Rust SDK Integration

AnamDB is built from the ground up as a native Rust crate. You can integrate the `anamdb` kernel directly into your own applications, backend microservices, or agentic loops.

---

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
anamdb = { git = "https://github.com/jam5991/anam.git" }
tokio = { version = "1.0", features = ["full"] }
```

If you plan to use GPU acceleration, enable the `cuda` feature flag:

```toml
[dependencies]
anamdb = { git = "https://github.com/jam5991/anam.git", features = ["cuda"] }
```

---

## Quick Start Example

Below is a complete, runnable example demonstrating how to initialize a session, ingest and register data, load an ONNX model, configure Datalog guardrails, run a query, and print the reasoning trace.

```rust
use std::sync::Arc;
use anamdb::Session;
use anamdb::core::session::{SessionConfig, ExplainLevel};
use anamdb::sdk::LogicPack;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize the session.
    // By default, this uses the CPU, with Polynomial semiring provenance.
    let session = Session::new().await?;

    // 2. Register a table (pre-created Lance dataset).
    session.register_table("txns", "demo/data/transactions_large.lance").await?;

    // 3. Register a model variant (Function-as-Operator).
    // Arguments: name, path, function_id, num_features, average_latency_ms, accuracy
    session.load_onnx_model_with_metrics(
        "fraud_detector",
        "1.0.0",
        "demo/models/fraud_detector.onnx",
        "fraud_detector",
        3,     // 3 input features
        5.0,   // 5ms average latency
        0.95,  // 95% accuracy
    )?;

    // 4. Define Datalog logical guardrails.
    session.register_logic(
        "high_risk",
        "fraud_prob > 0.90 AND amount > 10000"
    )?;

    // 5. Execute a query with multi-objective constraint specifications.
    // The query includes Pareto constraints at the end.
    let query = "SELECT region, COUNT(1) AS count, AVG(amount) as avg_amount \
                 FROM txns \
                 WHERE fraud_prob > 0.90 \
                 GROUP BY region \
                 WITH (max_latency_ms = 50, min_accuracy = 0.90)";
    
    let result = session.sql(query).await?;

    // 6. Handle the result batches (Arrow RecordBatches).
    for batch in &result.batches {
        println!("Batch schema: {:?}", batch.schema());
        println!("Number of rows: {}", batch.num_rows());
    }

    // 7. Inspect any anomalies detected by the Semantic Monitor.
    if result.requires_clarification() {
        println!("⚠️  Semantic anomalies detected!");
        for anomaly in &result.anomalies {
            println!("  - Anomaly: {}", anomaly.description);
        }
    }

    // 8. Generate and display the reasoning trace explanation.
    let explanation = session.explain_query(&result.batches, ExplainLevel::Coarse)?;
    println!("{}", explanation.display());

    Ok(())
}
```

---

## Detailed API Walkthrough

### 1. Custom Session Configurations
Customize session parameters using the `SessionConfig` struct:

```rust
use anamdb::core::session::{SessionConfig, ProvenanceMode};

let config = SessionConfig {
    provenance_mode: ProvenanceMode::Probability, // Boolean, Probability, or Polynomial
    enable_hardware_accel: true,                  // Force GPU/NPU dispatching
    llm_api_key: Some("sk-...".to_string()),      // For .nl compiler
    anomaly_threshold: 0.70,                      // Aggressiveness of HITL monitor
    ..Default::default()
};

let session = Session::with_config(config).await?;
```

Alternatively, you can load configuration options from a configuration TOML file:

```rust
let config = SessionConfig::load_from_toml("config/anam.toml")?;
let session = Session::with_config(config).await?;
```

### 2. Loading Logic Packs
Instead of manually registering individual rules and model structures, load a self-contained domain bundle:

```rust
let pack = LogicPack::from_file("demo/packs/financial_compliance.json")?;
let summary = session.load_logic_pack(&pack)?;
println!("{summary}");
```

### 3. Programmatic Self-Repair
If an operator crashes or timeouts at runtime, call the self-repair supervisor agent:

```rust
let report = session.self_repair(
    "operator exceeded deadline of 50ms",
    "fraud_detector",
    "batch of 10K rows on CPU"
)?;

if report.is_recoverable() {
    println!("Auto-repair suggestion applied: {:?}", report.action);
}
```
