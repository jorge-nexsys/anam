# REPL: Reasoning Explanation

Review query provenance and multi-dimensional execution traces.

---

## `.explain`
Prints the execution and reasoning summary trace from the last executed SQL statement.

* **Syntax**: `.explain`
* **Output Sections**:
  1. **Provenance:** Tracing modes (Boolean, Probability, Polynomial).
  2. **Datalog Rules:** Active logic filtering constraints.
  3. **AI-Tables Catalog:** Registered ONNX graphs and active operators.
  4. **Pareto Frontier:** Model candidate evaluations based on latency, accuracy, and cost metrics.
  5. **Device Pool:** Current core/acceleration engine status.
  6. **Anomalies:** Verification details of any semantic anomalies flagged by rules.

---

## Example

```sql
anam> .explain
═══════════════════════════════════════════════════════════
  AnamDB Reasoning Trace
═══════════════════════════════════════════════════════════

─── Provenance ─────────────────────────────────────────
  Mode: Polynomial
  Last query: 1 batch(es), 42 rows

─── Datalog Rules ──────────────────────────────────────
  • high_risk ← fraud_prob > 0.90 AND amount > 10000

─── AI-Tables Catalog ──────────────────────────────────
  • fraud_detector v1.0.0 [ONNX] — latency: 5.0ms, accuracy: 0.95
  • fraud_fast v1.0.0 [ONNX] — latency: 0.5ms, accuracy: 0.75

─── Pareto Frontier ────────────────────────────────────
  ★ frontier  fraud_fast v1.0.0: latency=0.500ms, accuracy=0.75, cost=0.0005
  ★ frontier  fraud_detector v1.0.0: latency=5.000ms, accuracy=0.95, cost=0.0050

─── Device Pool ────────────────────────────────────────
  • Device pool summary: 8 CPU cores (1.0x), Metal Core (10.0x)

═══════════════════════════════════════════════════════════
```
