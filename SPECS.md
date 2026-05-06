### AnamDB System Specifications (v0.1-alpha)
<br>**Release Date:** May 5, 2026
<br>**Status:** MVP Specification
<br>**Architecture:** Heterogeneity-Native Neurosymbolic Kernel
<br>**Maintainer:** NexSys Consulting (NSC)

---

#### 1. System Architecture Overview
AnamDB is a **differentiable database engine** written in Rust. It utilizes a unified memory architecture to fuse neural inference with symbolic query execution, abstracting away the complexities of AI model lifecycles.

##### 1.1 The "Logic-over-Tensor" Stack
*   **Kernel:** Rust (Edition 2024) utilizing `tokio` for async heterogeneous hardware orchestration.
*   **Execution Engine:** Extended **Apache DataFusion** (Heterogeneity-Native Dispatcher).
*   **Logic Subsystem:** Differentiable Datalog (based on the **Scallop-core** runtime).
*   **Model Manager:** Native registry utilizing AI-Tables and Function-as-Operator (FAO) designs.
*   **Storage:** **Lance 2.2** backing a Unified Relational Semantic Abstraction layer.

---

#### 2. The Logical Data Model: Semantic Abstraction & Provenance
AnamDB does not just store raw embeddings; it aligns disparate data modalities under a unified relational semantic schema to enable reliable symbolic reasoning.

##### 2.1 Unified Relational Semantic Abstraction
Raw unstructured data is automatically translated into structured relational views:
*   **Image/Video as Scene Graphs:** Visual content is modeled as objects interacting in space and time. The schema natively tracks `Objects` (bounding boxes, class labels), `Relationships` (spatial or temporal links between objects), and `Attributes`.
*   **Text as Semantic Entity Graphs:** Unstructured text is resolved into an `Entities` table (with unique IDs shared across document mentions), a `Mentions` table (character spans), and `Relationships` between those entities. 

##### 2.2 Semiring Provenance & Lineage
Every tuple $t$ in a relation $R$ is associated with a value from a commutative semiring $K$.
*   **$\mathbb{B}$ (Boolean):** Standard SQL behavior.
*   **$$ (Probability):** For neural confidence scores.
*   **$\mathbb{N}[X]$ (Polynomials):** Fine-grained lineage tracking. AnamDB records exactly which model version (`ver_id`), function (`func_id`), and source records produced each output tuple to ensure 100% explainability.

---

#### 3. Model Management & Function-as-Operator (FAO)
AI models are treated as first-class citizens alongside relational data.

*   **AI-Tables:** AnamDB stores the essential metadata of all available AI models (identifiers, inference speed, accuracy, network sizes, and versioning) directly within system tables.
*   **Function-as-Operator (FAO):** Query steps are compiled into explicit, version-stamped functions (e.g., `gen_excitement_score_v1`). This separation allows the system to effortlessly swap models (e.g., a fast lightweight model vs. a slow, highly accurate one) and track exactly which implementation derived a specific tuple.

---

#### 4. Query Execution & Multi-Objective Optimization
##### 4.1 Progressive Multi-Objective Optimizer
Because neural operators force a compromise between accuracy and speed, traditional latency-only cost models are insufficient.
*   **Pareto Frontier Calculation:** The optimizer balances execution latency, hardware cost, and result accuracy to determine the optimal physical execution plan.
*   **Progressive Refinement:** The optimizer dynamically monitors intermediate runtime statistics. If initial predictions miss accuracy thresholds, the system progressively adjusts the execution plan on the fly to route data to more accurate models.

##### 4.2 Heterogeneity-Native Query Execution
AnamDB physical plans consist of Mixed Directed Acyclic Graphs (DAGs).
*   **Fine-Grained Jobs:** Physical operators are decomposed into lightweight execution "jobs" (run-to-completion units).
*   **Dynamic Dispatch:** The execution engine automatically schedules these jobs across heterogeneous hardware (CPUs, GPUs, and NPUs) to guarantee load balancing and maximize inter- and intra-parallelism.

---

#### 5. Interactive Human-in-the-Loop (HITL) Debugging
AnamDB acknowledges that neural perception is imperfect. Instead of aborting queries on unexpected results, it introduces a conversational debugging channel.

*   **Agentic Semantic Monitor:** An internal monitor inspects intermediate tables. If it detects a "semantic anomaly" (a result that executes without syntactic errors but produces logically unexpected or contradictory outputs), it flags the operation.
*   **Interactive Triage:** Execution is paused on the anomalous data paths while unaffected tuples continue processing. AnamDB yields to the user with a natural language explanation of the anomaly, allowing them to clarify intent or patch the logic on the fly before resuming.

---

#### 6. Storage Engine: Lance 2.2
AnamDB leverages the **Lance 2.2** format to ensure zero-copy data flow between the DB and AI models.
*   **Multimodal Block Compression:** Uses LZ4 for text/metadata and specialized lossy/lossless codecs for tensor data.
*   **Versioning:** Snapshot isolation is built-in; every query can specify `AS OF 'timestamp'` for reproducible agent reasoning.

---

#### 7. Performance Metrics & Benchmarks
AnamDB prioritizes verifiable reasoning overhead and Pareto optimization.
| Metric | Definition | Target (MVP) |
| :--- | :--- | :--- |
| **RL (Reasoning Latency)** | Time to produce a proven logical conclusion. | < 200ms (per 10k rows) |
| **Pareto Accuracy Bound** | Minimum accuracy guaranteed under strict latency limits. | User-defined per query |
| **Proof Trace Overhead** | Latency cost of generating the provenance graph. | < 15% of total query time |
| **Heterogeneous Throughput**| Rows processed per second across CPU/GPU/NPU. | 5,000+ rows/sec |
