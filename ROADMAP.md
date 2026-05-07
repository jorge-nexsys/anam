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

- [x] **Distributed Lance Storage:** Partition the Arrow-backed Lance storage across the cluster, maintaining zero-copy reads for local edge nodes while enabling distributed similarity searches.
- [x] **BCNF Policy Enforcement:** Migrate all Datalog rules, programmatic constraints, and schema definitions into a strict Boyce-Codd Normal Form (BCNF) relational catalog to ensure seamless, anomaly-free policy updates across the cluster.
- [x] **Distributed Multi-Objective Optimization:** Upgrade the Neural Cost-Based Optimizer (NCBO) to include network routing costs and data-movement overhead in its Pareto frontier calculations.
- [x] **Multi-Agent Task Routing:** Implement FAO routing to dynamically dispatch perception functions (e.g., running an NPU-accelerated OCR model) to edge nodes while reserving high-memory symbolic joins for core nodes.
- [x] **Global Lineage & Decentralized Triage:** Serialize the Semiring Provenance across network boundaries and enable cluster-wide Agentic Monitoring that isolates anomalous data paths locally without blocking the rest of the distributed system.

---

## Phase 4: Engine Integration — Neural-Symbolic Query Fusion
*Focus: Wiring existing subsystems into a single unified query pipeline where models, logic, and provenance execute inline.*

- [x] **ONNX Models as DataFusion UDFs:** Register FAO operators as scalar UDFs so `SELECT fraud_detector(amount, region, time) FROM txns` runs ONNX inference inline during query execution.
- [x] **Datalog Rules as Query Filters:** Wire the LogicEngine into DataFusion's physical plan so registered rules act as post-query filters that block results violating symbolic constraints.
- [x] **Provenance Attachment:** Build a custom `ExecutionPlan` wrapper that appends a `provenance: Binary` column (Polynomial semiring) to every output batch, encoding lineage per-row.
- [x] **Streaming Lance TableProvider:** Replace the eager `scan_to_memtable` with a proper `TableProvider` that wraps Lance's streaming scanner with push-down projection and filter support.
- [x] **Write Path (INSERT / UPDATE / DELETE):** Implement SQL mutation support via Lance's append, merge, and delete APIs.
- [x] **Persistent Catalog:** SQLite-backed catalog that stores registered tables, Datalog rules, model metadata, BCNF policies, and session config across restarts.

---

## Phase 5: Server & SDKs — Production Wire Protocol
*Focus: `cargo install anamdb` → `anamdb serve` → connect from any application.*

- [x] **gRPC Server (`anam-server`):** Production gRPC server using `tonic` with streaming result delivery, health checks, and reflection.
- [x] **Wire Protocol:** SQL submission, streaming Arrow IPC result batches, dot-command RPC, session management, and authentication stubs.
- [x] **Rust Client SDK (`anam-client`):** Native async Rust client with connection pooling, retry logic, and typed query builders.
- [x] **Python SDK (`pyanamdb`):** PyO3 native bindings wrapping the Rust client for zero-overhead Python integration.
- [x] **CLI & Packaging:** `cargo install anamdb`, `anamdb init ./data`, `anamdb serve --port 8080 --gpu`, config file (`anamdb.toml`).
- [x] **Docker Image:** `docker run anamdb` with pre-configured GPU passthrough and volume mounts.

---

## Phase 6: Live Demo — Interactive Playground (Local + Hosted)
*Focus: A web-based playground backed by a live AnamDB instance, deployable both locally and to cloud.*

- [x] **Web Frontend:** SQL editor (Monaco/CodeMirror) + streaming results grid + reasoning trace panel + dark-mode UI.
- [x] **WebSocket Bridge:** Real-time streaming from the gRPC backend to the browser via a thin WebSocket relay.
- [x] **Demo Dataset & Logic Pack:** One-click setup with pre-loaded 100K transactions, ONNX models, and Financial Compliance Logic Pack.
- [x] **Pareto Visualization:** Interactive scatter plot showing the latency × accuracy × cost frontier with selectable plan highlighting.
- [x] **Provenance Tree Viewer:** Click any result row → visual derivation tree showing source records, model versions, and confidence at each hop.
- [x] **HITL Triage UI:** Accept / Correct / RetryWithModel / Abort buttons for interactive anomaly resolution in the browser.
- [x] **Hosted Deployment:** Cloud-hosted instance (Fly.io / Railway / AWS) with shareable URL for investors and partners.

---

## Phase 7: Ecosystem & Customization
*Focus: Broadening the ecosystem and lowering the barrier to entry for highly specialized domains.*

- [ ] **Automated Model Distillation:** Allow users to automatically distill large, slow models (e.g., VideoMAEV2) into smaller, faster equivalents directly within the database to shift the Pareto frontier favorably.
- [ ] **Expanded Semantic Abstractions:** Native support for 3D spatial representations and advanced temporal audio graphs within the unified relational layer.
- [ ] **Community AI-Tables Hub:** A centralized package manager for developers to share pre-trained FAO models and Datalog constraints.

