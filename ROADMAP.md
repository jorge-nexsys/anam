# AnamDB Roadmap

This document outlines the planned future work and major milestones for the AnamDB neurosymbolic database engine.

---

## 🛠️ Engine Core & Performance

### Stress & Soak Testing
We need to hammer the `DevicePool` and `PredictExec` nodes with highly concurrent queries (e.g., 10,000+ concurrent connections) to hunt down any obscure Rust async deadlocks or memory leaks over extended periods.

### Hybrid Neurosymbolic Search
Integrate Lance's dense vector search capabilities with BM25 full-text indexing. This will unify dense and sparse search under a single SQL-plus-logic query optimization pipeline.

---

## 🔒 Security & Operations

### Secrets Management & Access Control
Currently, the LLM integration relies on a local `.env` file for the `OPENAI_API_KEY`. For production deployments, this needs to be integrated with a secure secrets manager (like AWS Secrets Manager or HashiCorp Vault).

### Logic-Based Security (Datalog RBAC)
Define and enforce database access control policies using the internal Datalog reasoning engine (e.g., matching user sessions and security properties dynamically at the query execution level).

---

## 🌐 Agent-Centric Capabilities

### Native MCP Server Mode (`anam serve --mcp`)
Build a native Model Context Protocol (MCP) server option directly into AnamDB. This will expose Datalog logic rules, saved queries, and reasoning schemas to LLM agents as standard discoverable tools.

### Cloud & Hub Synchronizer (`anam push` / `anam sync`)
Add CLI commands to pair local project environments with a remote registry, enabling developers to publish, manage, and sync custom logic packs and model assets easily.

---

## 📈 Distributed Scale-Out

### Distributed Reasoning Plane
AnamDB is currently a single-node engine. To support petabyte-scale datasets, the system needs a distributed consensus layer (e.g., Raft integration) to distribute DataFusion query execution plans across a cluster of worker nodes.