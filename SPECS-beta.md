# AxiomDB System Specifications (v0.5-beta)
**Release Date:** Q3 2026
**Status:** Beta Specification
**Architecture:** Heterogeneity-Native Neurosymbolic Kernel with SDK
**Maintainer:** NexSys Consulting (NSC)

---

## 1. System Architecture Evolution
While the Alpha release established the core Heterogeneity-Native Kernel, zero-copy Lance storage, and the Pareto Multi-Objective Optimizer, the Beta release shifts focus to **Developer Experience (UX) and Trust**. 

AxiomDB Beta introduces the **"Logic Pack" SDK** to decentralize business logic and an **Agentic Debugging Channel** to interactively resolve semantic anomalies, transforming the database from a passive storage engine into an active, collaborative reasoning agent.

---

## 2. "Logic Pack" SDK & NL-to-Logic Compilation
AxiomDB Beta abstracts away raw Datalog programming by allowing third-party developers to build domain-specific "Logic Packs" (e.g., Financial Compliance, Healthcare Advising) using Natural Language and modular APIs.

### 2.1 Conversational Intent Translation
Developers define constraints in natural language. AxiomDB utilizes an embedded LLM interpreter to parse these requests into strict, differentiable Datalog.
*   **Dual-Stage Parsing:** Natural language passes through Intent/Named Entity Recognition (NER) to extract entities, which are then compiled into Datalog constraints. 
*   **Strict Bounding:** To prevent LLM hallucinations, the generated logic must compile against the unified relational schema. If the system lacks the necessary context to enforce a rule, it forces an explicit `INSUFFICIENT_CONTEXT` fallback rather than guessing.

### 2.2 Function-as-Operator (FAO) Registration
Logic Packs register both symbolic rules and neural models as packaged operators. The system uses a JSON-based schema for FAO registration, explicitly defining the `inputs`, `output`, and a natural language `description` for the query optimizer to utilize during execution planning.

---

## 3. Interactive Human-in-the-Loop (HITL) Debugging
Traditional databases abort queries on runtime errors, while pure neural pipelines silently return hallucinated garbage. AxiomDB Beta introduces a two-tiered Agentic Monitor that pauses execution for user feedback.

### 3.1 Syntactic Self-Repair
If a neural operator encounters a structural error (e.g., an unsupported file format or dimension mismatch), a two-agent loop is triggered:
*   A **Reviewer Agent** diagnoses the exception.
*   A **Rewriter Agent** patches the function code, increments the `ver_id` in the AI-Table, and resumes execution from the point of failure while unaffected tuples continue processing in parallel.

### 3.2 Semantic Anomaly Resolution
Semantic errors occur when code runs perfectly but produces contextually illogical outputs (e.g., a visual classifier confidently matching a single entity to logically contradictory events). 
*   **Anomaly Detection:** The Semantic Monitor inspects intermediate result tables for contradictions or extreme confidence uniformities. 
*   **Interactive Triage:** The monitor pauses the anomalous data path and yields to the developer via the SDK. It presents a natural language explanation of the likely cause (e.g., "The model assumed a 1:1 mapping, but 1:N was detected") and asks the user to explicitly accept, adjust, or rewrite the rule before resuming.

---

## 4. Query Result Explainer (Lineage Queries)
Because AxiomDB tracks Semiring Polynomial ($\mathbb{N}[X]$) provenance, developers can query the reasoning behind any result through the SDK. The Explainer supports two modes:
*   **Coarse-Grained Explanation:** Summarizes the high-level physical plan, translating the sequence of transformations (e.g., "Filtered transactions based on visual OCR features and matched against the EU compliance rule").
*   **Fine-Grained Explanation:** Takes a specific tuple ID (`lid`) and traces its exact parent tuples, the model versions (`ver_id`) used, and the intermediate confidence scores that derived the final output.

---

## 5. SDK Interface & Client API
The Beta SDK provides native bindings for Rust and Python, allowing seamless integration with existing agentic frameworks.

```rust
use axiomdb::sdk::LogicPack;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut session = axiomdb::Client::connect("localhost:8080").await?;
    
    // Load a domain-specific Logic Pack
    session.load_pack(LogicPack::FinancialCompliance).await?;

    // Execute with an interactive HITL callback channel
    let mut stream = session.query_interactive("SELECT * FROM HighRisk").await?;
    
    while let Some(event) = stream.next().await {
        match event {
            Event::Result(data) => println!("Verified Tuple: {:?}", data),
            Event::SemanticAnomaly(anomaly) => {
                println!("Anomaly Detected: {}", anomaly.description);
                // Yield back to the engine with a correction
                stream.provide_feedback("Require 2FA verification for this specific user.").await?;
            }
        }
    }
    Ok(())
}
```

---

## 6. Performance Targets (Beta)
| Metric | Definition | Target |
| :--- | :--- | :--- |
| **NL-to-Logic Compilation** | Time to parse natural language to Datalog. | < 500ms |
| **Anomaly Triage Overhead** | Cost of semantic monitoring on intermediate tables. | < 10% query time |
| **Explanation Generation** | Time to generate coarse/fine-grained lineage text. | < 1.5s per trace |

---
*Roadmap Horizon: v1.0 Distributed Reasoning Plane*
