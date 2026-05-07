# AnamDB Development Roadmap
**Vision:** To build the first AI-native, heterogeneity-native neurosymbolic database that treats probabilistic models and deterministic logic as equal citizens.

This roadmap outlines the path from our current single-node logic kernel to a fully distributed, multi-agent reasoning plane.

---

## Phase 1: Alpha (Current) — The Heterogeneity-Native Kernel
*Focus: Establishing the core execution engine, zero-copy memory architecture, and verifiable provenance.*

- [x] **Embed Datalog Engine:** Integrate `scallop-core` into the Rust environment for differentiable logic execution.
- [x] **Zero-Copy Multimodal Storage:** Implement Apache Arrow and Lance 2.2 for sub-millisecond local retrieval and unified relational semantic abstraction.
- [x] **Semiring Provenance:** Implement $\mathbb{N}[X]$ (Polynomial) provenance tracking to trace the lineage of every tuple to its exact source records and model versions.
- [x] **Explicit Model Manager:** Build the "AI-Tables" registry to store model metadata (inference speed, accuracy, versioning) and deploy the Function-as-Operator (FAO) architecture.
- [x] **Multi-Objective Query Optimizer:** Extend Apache DataFusion to dynamically calculate the Pareto frontier, balancing execution latency, hardware cost, and result accuracy.
- [x] **Heterogeneous Dispatcher:** Enable the execution engine to dynamically schedule "jobs" across mixed hardware (CPUs, GPUs, and NPUs) to maximize inter- and intra-parallelism.

---

## Phase 2: Beta — Developer Experience (UX) & Agentic Trust
*Focus: Decentralizing business logic via SDKs, automating rule generation, and implementing interactive debugging.*

- [x] **"Logic Pack" SDK:** Release a modular Rust and Python SDK allowing third-party developers to build domain-specific neurosymbolic rulesets (e.g., Healthcare, Financial Compliance).
- [x] **LLM-Assisted Rule Generation:** Build the conversational intent translation pipeline that uses an embedded LLM to parse natural language constraints into strict, differentiable Datalog. 
- [x] **Syntactic Self-Repair:** Develop a two-agent loop (Reviewer & Rewriter) to automatically patch structural errors (like unsupported file formats) on the fly without aborting queries.
- [x] **Semantic Anomaly Resolution (HITL):** Implement the Agentic Semantic Monitor to detect logically valid but contextually bizarre outputs (e.g., suspicious uniform confidence scores). The system will pause anomalous data paths and yield to the user for interactive triage.
- [x] **Query Result Explainer:** Add natural language lineage querying, allowing users to ask for coarse-grained pipeline summaries or fine-grained tuple derivations via the SDK.

---

## Phase 3: v1.0 — The Distributed Reasoning Plane
*Focus: Scaling logical inference and neural perception across multi-agent clusters.*

- [ ] **Distributed Lance Storage:** Partition the Arrow-backed Lance storage across the cluster, maintaining zero-copy reads for local edge nodes while enabling distributed similarity searches.
- [ ] **BCNF Policy Enforcement:** Migrate all Datalog rules, programmatic constraints, and schema definitions into a strict Boyce-Codd Normal Form (BCNF) relational catalog to ensure seamless, anomaly-free policy updates across the cluster.
- [ ] **Distributed Multi-Objective Optimization:** Upgrade the Neural Cost-Based Optimizer (NCBO) to include network routing costs and data-movement overhead in its Pareto frontier calculations.
- [ ] **Multi-Agent Task Routing:** Implement FAO routing to dynamically dispatch perception functions (e.g., running an NPU-accelerated OCR model) to edge nodes while reserving high-memory symbolic joins for core nodes.
- [ ] **Global Lineage & Decentralized Triage:** Serialize the Semiring Provenance across network boundaries and enable cluster-wide Agentic Monitoring that isolates anomalous data paths locally without blocking the rest of the distributed system.

---

## Phase 4: Beyond v1.0 — Ecosystem & Customization
*Focus: Broadening the ecosystem and lowering the barrier to entry for highly specialized domains.*

- [ ] **Automated Model Distillation:** Allow users to automatically distill large, slow models (e.g., VideoMAEV2) into smaller, faster equivalents directly within the database to shift the Pareto frontier favorably.
- [ ] **Expanded Semantic Abstractions:** Native support for 3D spatial representations and advanced temporal audio graphs within the unified relational layer.
- [ ] **Community AI-Tables Hub:** A centralized package manager for developers to share pre-trained FAO models and Datalog constraints. 
