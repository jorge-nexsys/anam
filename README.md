# AnamDB (MVP) 
### *The AI-Native, Differentiable Logic Kernel for Autonomous Agents*

**AnamDB** is a vertical-agnostic, neurosymbolic database engine built in Rust. It is designed to natively integrate probabilistic neural perception with deterministic symbolic reasoning into a unified architecture. 

Unlike traditional vector databases that rely on semantic similarity ("vibes") or bolt-on LLMs, AnamDB treats **Models as First-Class Citizens** and **Logic as a Verifiable Blueprint**. It enables agents to perform rigorous reasoning over multi-modal unstructured data while balancing execution speed and accuracy dynamically.

---

## 🧠 Why AnamDB?

Traditional RAG (Retrieval-Augmented Generation) is hitting the "Trust Wall," while standard databases cannot natively reason over unstructured data. AnamDB bridges this gap by addressing the core challenges of AI-native data management:

* **LLM-Assisted Rule Generation:** Overcome the UX bottleneck of manual logic programming. AnamDB allows users to define constraints in natural language, which are automatically compiled into verifiable Datalog rules.
* **Multi-Objective Query Optimization:** Neural operators force a trade-off between speed and accuracy. Our optimizer explicitly calculates the Pareto frontier, dynamically selecting the best model and execution path based on your latency and cost constraints. 
* **Explicit Model Management (AI-Tables):** AI models are managed natively alongside data. "AI-Tables" store metadata (inference speed, accuracy, versioning) for all available models, abstracting away the complexity of model lifecycles.
* **Human-in-the-Loop Debugging:** When semantic anomalies occur (logically valid but contextually incorrect outputs), an agentic monitor pauses execution and interacts with the user to clarify intent and fix the logic on the fly.
* **Verifiable Proof Traces:** Every query returns a "Reasoning Tree" (fine-grained lineage) proving exactly *how* a result was derived across heterogeneous data sources.
* **Zero-Copy Performance:** Built on **Apache Arrow**, allowing neural models and the logic engine to share the same memory space for sub-millisecond local retrieval.

---

## Tech Stack

* **Core Engine:** Rust (using `tokio` for async execution).
* **Query Optimizer:** Extended **Apache DataFusion** with custom Neuro-Operators and Multi-Objective Cost Modeling.
* **Logic Layer:** Differentiable Datalog (via `scallop-core`).
* **Model Manager:** Native registry utilizing **AI-Tables** and **Function-as-Operator (FAO)** architecture.
* **Storage Format:** **Lance** (Optimized for hybrid Vector + Relational data).
* **Inference:** ONNX Runtime / Burn (NPU-accelerated).

---

## Quick Start (MVP Preview)

AnamDB allows you to seamlessly join natural language constraints with raw perception and hard business logic.

### 1. Define Constraints via Natural Language
Instead of writing raw Prolog, define your rules conversationally. The LLM translates this into strict Datalog.
```rust
// AnamDB parses: "Flag a transaction as 'High Risk' if the model is 90% sure it's 
// fraudulent AND it exceeds $10k in the EU."
ctx.register_logic_from_nl("HighRisk", "transactions", "fraud_prob > 0.90 AND amount > 10000 AND region = 'EU'").await?;
```

### 2. Execute via Rust API with Cost Constraints
```rust
use Anamdb::core::Session;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Session::new_with_npu().await?;
    
    // 1. Ingest raw data (Lance Format)
    ctx.register_table("transactions", "data/tx_2026.lance").await?;

    // 2. Run Neurosymbolic Query with a Multi-Objective Constraint
    // The Model Manager automatically selects the optimal model from AI-Tables 
    // to meet the latency/accuracy requirements (Pareto frontier calculation).
    let query = "SELECT * FROM HighRisk WITH (max_latency_ms = 50, min_accuracy = 0.95)";
    let mut results = ctx.sql(query).await?;

    // 3. Human-in-the-Loop (HITL) Triage
    // If the semantic monitor detects an anomaly or uncertainty, it yields for feedback.
    if results.requires_clarification() {
        let correction = "Only count EU transactions if they are outside of Germany.";
        results = ctx.refine_query(correction).await?;
    }

    // 4. Inspect the Proof Trace
    results.explain_reasoning().await?;
    
    Ok(())
}
```

---

## Architecture: The Kernel Strategy

AnamDB is Vertical-Agnostic. It provides the kernel; you provide the "Logic Packs."

| Layer | Component | Function |
| :--- | :--- | :--- |
| **Interface** | NL-to-Logic + SQL | Conversational intent translation and standard query entry. |
| **Agentic Monitor** | HITL Debugger | Detects semantic anomalies and interacts with users for on-the-fly correction. |
| **Logic Layer** | Symbolic Engine | Compiles Datalog into differentiable logical circuits. |
| **Model Manager** | AI-Tables & FAO | Manages AI models as first-class citizens, tracking inference speed, accuracy, and versioning. |
| **Execution** | DataFusion (Extended)| Calculates the Pareto frontier to balance latency vs. accuracy for physical execution plans. |
| **Storage** | Lance/Arrow | Columnar storage tracking fine-grained lineage and provenance. |

---

## Roadmap

- [x] **Pre-Alpha:** Embedded Rust Engine with basic Datalog support.
- [x] **Alpha (Current):** Implement explicit Model Manager (AI-Tables) and Multi-Objective Query Optimizer (Pareto balancing).
- [ ] **Beta:** "Logic Pack" SDK with an Interactive Human-in-the-Loop debugging channel for resolving semantic anomalies.
- [ ] **v1.0:** Distributed "Reasoning Plane" for multi-agent clusters with automated LLM-to-Datalog generation pipelines.

