# AnamDB System Specifications
**Architecture:** Distributed Neurosymbolic Kernel & Multi-Agent Reasoning Plane

---

# Phase 1 — Alpha: The Heterogeneity-Native Kernel

## System Architecture Overview
AnamDB is a **differentiable database engine** written in Rust. It utilizes a unified memory architecture to fuse neural inference with symbolic query execution, abstracting away the complexities of AI model lifecycles.

### The "Logic-over-Tensor" Stack
*   **Kernel:** Rust (Edition 2024) utilizing `tokio` for async heterogeneous hardware orchestration.
*   **Execution Engine:** Extended **Apache DataFusion** (Heterogeneity-Native Dispatcher).
*   **Logic Subsystem:** Differentiable Datalog (based on the **Scallop-core** runtime).
*   **Model Manager:** Native registry utilizing AI-Tables and Function-as-Operator (FAO) designs.
*   **Storage:** **Lance 2.2** backing a Unified Relational Semantic Abstraction layer.

---

## Logical Data Model: Semantic Abstraction & Provenance
AnamDB does not just store raw embeddings; it aligns disparate data modalities under a unified relational semantic schema to enable reliable symbolic reasoning.

### Unified Relational Semantic Abstraction
Raw unstructured data is automatically translated into structured relational views:
*   **Image/Video as Scene Graphs:** Visual content is modeled as objects interacting in space and time. The schema natively tracks `Objects` (bounding boxes, class labels), `Relationships` (spatial or temporal links between objects), and `Attributes`.
*   **Text as Semantic Entity Graphs:** Unstructured text is resolved into an `Entities` table (with unique IDs shared across document mentions), a `Mentions` table (character spans), and `Relationships` between those entities.

### Semiring Provenance & Lineage
Every tuple $t$ in a relation $R$ is associated with a value from a commutative semiring $K$.
*   **$\mathbb{B}$ (Boolean):** Standard SQL behavior.
*   **$$ (Probability):** For neural confidence scores.
*   **$\mathbb{N}[X]$ (Polynomials):** Fine-grained lineage tracking. AnamDB records exactly which model version (`ver_id`), function (`func_id`), and source records produced each output tuple to ensure 100% explainability.

---

## Model Management & Function-as-Operator (FAO)
AI models are treated as first-class citizens alongside relational data.

*   **AI-Tables:** AnamDB stores the essential metadata of all available AI models (identifiers, inference speed, accuracy, network sizes, and versioning) directly within system tables.
*   **Function-as-Operator (FAO):** Query steps are compiled into explicit, version-stamped functions (e.g., `gen_excitement_score_v1`). This separation allows the system to effortlessly swap models (e.g., a fast lightweight model vs. a slow, highly accurate one) and track exactly which implementation derived a specific tuple.

---

## Query Execution & Multi-Objective Optimization

### Progressive Multi-Objective Optimizer
Because neural operators force a compromise between accuracy and speed, traditional latency-only cost models are insufficient.
*   **Pareto Frontier Calculation:** The optimizer balances execution latency, hardware cost, and result accuracy to determine the optimal physical execution plan.
*   **Progressive Refinement:** The optimizer dynamically monitors intermediate runtime statistics. If initial predictions miss accuracy thresholds, the system progressively adjusts the execution plan on the fly to route data to more accurate models.

### Heterogeneity-Native Query Execution
AnamDB physical plans consist of Mixed Directed Acyclic Graphs (DAGs).
*   **Fine-Grained Jobs:** Physical operators are decomposed into lightweight execution "jobs" (run-to-completion units).
*   **Dynamic Dispatch:** The execution engine automatically schedules these jobs across heterogeneous hardware (CPUs, GPUs, and NPUs) to guarantee load balancing and maximize inter- and intra-parallelism.

---

## Interactive Human-in-the-Loop (HITL) Debugging
AnamDB acknowledges that neural perception is imperfect. Instead of aborting queries on unexpected results, it introduces a conversational debugging channel.

*   **Agentic Semantic Monitor:** An internal monitor inspects intermediate tables. If it detects a "semantic anomaly" (a result that executes without syntactic errors but produces logically unexpected or contradictory outputs), it flags the operation.
*   **Interactive Triage:** Execution is paused on the anomalous data paths while unaffected tuples continue processing. AnamDB yields to the user with a natural language explanation of the anomaly, allowing them to clarify intent or patch the logic on the fly before resuming.

---

## Storage Engine: Lance 2.2
AnamDB leverages the **Lance 2.2** format to ensure zero-copy data flow between the DB and AI models.
*   **Multimodal Block Compression:** Uses LZ4 for text/metadata and specialized lossy/lossless codecs for tensor data.
*   **Versioning:** Snapshot isolation is built-in; every query can specify `AS OF 'timestamp'` for reproducible agent reasoning.

---

## Alpha Performance Targets
| Metric | Definition | Target |
| :--- | :--- | :--- |
| **Reasoning Latency** | Time to produce a proven logical conclusion. | < 200ms (per 10k rows) |
| **Pareto Accuracy Bound** | Minimum accuracy guaranteed under strict latency limits. | User-defined per query |
| **Proof Trace Overhead** | Latency cost of generating the provenance graph. | < 15% of total query time |
| **Heterogeneous Throughput** | Rows processed per second across CPU/GPU/NPU. | 5,000+ rows/sec |

---
---

# Phase 2 — Beta: Developer Experience & Agentic Trust

## Architecture Evolution
While the Alpha release established the core Heterogeneity-Native Kernel, zero-copy Lance storage, and the Pareto Multi-Objective Optimizer, the Beta release shifts focus to **Developer Experience (UX) and Trust**.

AnamDB Beta introduces the **"Logic Pack" SDK** to decentralize business logic and an **Agentic Debugging Channel** to interactively resolve semantic anomalies, transforming the database from a passive storage engine into an active, collaborative reasoning agent.

---

## "Logic Pack" SDK & NL-to-Logic Compilation
AnamDB Beta abstracts away raw Datalog programming by allowing third-party developers to build domain-specific "Logic Packs" (e.g., Financial Compliance, Healthcare Advising) using Natural Language and modular APIs.

### Conversational Intent Translation
Developers define constraints in natural language. AnamDB utilizes an embedded LLM interpreter to parse these requests into strict, differentiable Datalog.
*   **Dual-Stage Parsing:** Natural language passes through Intent/Named Entity Recognition (NER) to extract entities, which are then compiled into Datalog constraints.
*   **Strict Bounding:** To prevent LLM hallucinations, the generated logic must compile against the unified relational schema. If the system lacks the necessary context to enforce a rule, it forces an explicit `INSUFFICIENT_CONTEXT` fallback rather than guessing.

### Function-as-Operator (FAO) Registration
Logic Packs register both symbolic rules and neural models as packaged operators. The system uses a JSON-based schema for FAO registration, explicitly defining the `inputs`, `output`, and a natural language `description` for the query optimizer to utilize during execution planning.

---

## Enhanced HITL Debugging
Traditional databases abort queries on runtime errors, while pure neural pipelines silently return hallucinated garbage. AnamDB Beta introduces a two-tiered Agentic Monitor that pauses execution for user feedback.

### Syntactic Self-Repair
If a neural operator encounters a structural error (e.g., an unsupported file format or dimension mismatch), a two-agent loop is triggered:
*   A **Reviewer Agent** diagnoses the exception.
*   A **Rewriter Agent** patches the function code, increments the `ver_id` in the AI-Table, and resumes execution from the point of failure while unaffected tuples continue processing in parallel.

### Semantic Anomaly Resolution
Semantic errors occur when code runs perfectly but produces contextually illogical outputs (e.g., a visual classifier confidently matching a single entity to logically contradictory events).
*   **Anomaly Detection:** The Semantic Monitor inspects intermediate result tables for contradictions or extreme confidence uniformities.
*   **Interactive Triage:** The monitor pauses the anomalous data path and yields to the developer via the SDK. It presents a natural language explanation of the likely cause (e.g., "The model assumed a 1:1 mapping, but 1:N was detected") and asks the user to explicitly accept, adjust, or rewrite the rule before resuming.

---

## Query Result Explainer (Lineage Queries)
Because AnamDB tracks Semiring Polynomial ($\mathbb{N}[X]$) provenance, developers can query the reasoning behind any result through the SDK. The Explainer supports two modes:
*   **Coarse-Grained Explanation:** Summarizes the high-level physical plan, translating the sequence of transformations (e.g., "Filtered transactions based on visual OCR features and matched against the EU compliance rule").
*   **Fine-Grained Explanation:** Takes a specific tuple ID (`lid`) and traces its exact parent tuples, the model versions (`ver_id`) used, and the intermediate confidence scores that derived the final output.

---

## SDK Interface & Client API
The Beta SDK provides native bindings for Rust and Python, allowing seamless integration with existing agentic frameworks.

```rust
use anamdb::sdk::LogicPack;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut session = anamdb::Client::connect("localhost:8080").await?;

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

## Beta Performance Targets
| Metric | Definition | Target |
| :--- | :--- | :--- |
| **NL-to-Logic Compilation** | Time to parse natural language to Datalog. | < 500ms |
| **Anomaly Triage Overhead** | Cost of semantic monitoring on intermediate tables. | < 10% query time |
| **Explanation Generation** | Time to generate coarse/fine-grained lineage text. | < 1.5s per trace |

---
---

# Phase 3 — v1.0: The Distributed Reasoning Plane

## Architecture Evolution
While the Alpha and Beta releases established a single-node, heterogeneity-native engine with an interactive SDK, the v1.0 release introduces the **Distributed Reasoning Plane**. AnamDB v1.0 scales logical inference and neural perception across multi-agent clusters. The system transitions from a localized database into a distributed cognitive engine, ensuring that agents can collaborate, share verified knowledge, and execute complex workflows in dynamically changing environments.

---

## The 5-Stage Symbolic Integration Pipeline
To standardize how neural and symbolic components interact across a distributed cluster, AnamDB v1.0 formally adopts a 5-stage symbolic integration framework:
1.  **Data Preprocessing:** Raw input (e.g., text, video, distributed logs) is transposed into structured vector-symbolic representations.
2.  **Neural-Symbolic Embedding:** Neural networks extract features utilizing first-order logic and mathematical constraints integrated directly within the network nodes to guarantee early-stage adherence to business logic.
3.  **Domain Knowledge Incorporation:** Extracted features are cross-referenced with distributed symbolic representations and domain ontologies to provide prior knowledge inference.
4.  **Logical Reasoning Modules:** Core execution of Datalog/Prolog logic programs. The system applies strict IF-THEN rules over the neural outputs to infer actions.
5.  **Symbolic Postprocessing:** A final layer of constraint checking is applied before the system outputs the result or triggers an autonomous agent action.

---

## Distributed Storage & BCNF Policy Enforcement
Managing rules and raw data across a distributed cluster requires robust consistency models.
*   **Distributed Lance Storage:** Arrow-backed Lance storage is partitioned across the cluster, maintaining zero-copy data reads for local agent nodes while allowing distributed similarity searches.
*   **BCNF Normalized Knowledge Base:** To prevent the "brittleness" of hard-coded logic when institutional policies change, all Datalog rules, programmatic constraints, and schemas are stored in a strict Boyce-Codd Normal Form (BCNF) relational catalog. This ensures that any update to a constraint propagates cleanly across the distributed network without insertion or update anomalies, maintaining referential integrity across all nodes.

---

## Multi-Agent Task Planning & Execution
AnamDB v1.0 acts as the central coordination layer for autonomous agent clusters.
*   **Plugin Framework for Multi-Agent Systems:** AnamDB utilizes a plugin-based architecture allowing symbolic task planning to operate in parallel with neural reinforcement learning across multiple agents.
*   **Function-as-Operator (FAO) Routing:** Following the FAO paradigm, each step in a distributed query or agent workflow is compiled into an explicit, version-stamped function. The central query engine can route perception functions (e.g., running an NPU-accelerated OCR model) to edge nodes, while reserving symbolic join functions for high-memory core nodes.

---

## Distributed Multi-Objective Query Optimization
The Neural Cost-Based Optimizer (NCBO) is upgraded to handle network partitions.
*   **Network-Aware Pareto Frontiers:** The optimizer continues to calculate the Pareto frontier to balance latency, hardware cost, and result accuracy. However, it now explicitly includes network routing costs and data-movement overhead in its cost estimation.
*   **Progressive Refinement at Scale:** During execution, intermediate runtime statistics are monitored. If an edge node's lightweight neural model fails to meet accuracy constraints, the progressive optimizer dynamically rewrites the physical execution plan on the fly to route the remaining data to a more accurate, high-capacity model on a central node.

---

## Global Lineage & Decentralized HITL Triage
*   **Distributed Semiring Provenance:** The Polynomial ($\mathbb{N}[X]$) lineage tracking is serialized across network boundaries. Users can query the lineage of any result tuple and trace its derivation back through intermediate views, specific agent nodes, and precise model versions (`ver_id`).
*   **Cluster-Wide Agentic Monitoring:** When a semantic anomaly is detected (e.g., logically valid but contextually incorrect outputs), the Agentic Monitor isolates the anomalous data path locally, allowing the rest of the cluster to continue processing. It then initiates an interactive Human-in-the-Loop (HITL) triage session, asking the user to provide corrective feedback or adjust the BCNF-stored rules before resuming.

---

## v1.0 Performance Targets
| Metric | Definition | Target |
| :--- | :--- | :--- |
| **Cluster Reasoning Latency** | Time to execute logical constraints across distributed shards. | < 500ms (per 100k rows) |
| **Rule Propagation Time** | Latency for a BCNF catalog update to reach all agents. | < 50ms |
| **Distributed Provenance Trace** | Time to reconstruct a full reasoning tree across nodes. | < 2.0s |
