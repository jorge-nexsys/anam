# Limits & Benchmarks

Operational parameters, hardware requirements, and system bounds for AnamDB.

---

## Memory & Sizing Limits

* **Maximum Query Payload:** Default SQL query payload is capped at 10 MB per statement.
* **Vector Dimensionality:** Lance indexes support up to 4096 dimensions for embedding attributes.
* **Datalog Rule Sizing:** Logic compilation supports up to 100 concurrent terms per Datalog rule.

---

## Performance Targets

AnamDB is optimized for:
* **Sub-millisecond Logic Resolution:** Utilizing differential evaluation pipelines for Datalog.
* **Pareto Optimization Overhead:** Restricting multi-objective plan evaluation cost to less than 1% of total execution time.
* **Device Acceleration Scalability:** Parallelizing batch inference runs automatically across available CPU and GPU pools.
