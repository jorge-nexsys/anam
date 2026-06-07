# Explainability & HITL

AnamDB ensures trust and reliability in AI-driven data pipelines using a three-tier system: **Explainability**, **Human-in-the-Loop (HITL) semantic monitoring**, and **Syntactic Self-Repair**.

---

## The Query Explainer

Every query in AnamDB can generate a reasoning trace. You can extract explanations at two levels of detail.

### 1. Coarse-Grained Summary
A high-level summary of the rules applied, the models loaded, execution device statistics, and output probability distributions.

```
═══════════════════════════════════════════════════════════
  AnamDB Query Explanation
  Level: Coarse
═══════════════════════════════════════════════════════════

─── Pipeline Summary ───────────────────────────────────
  Produced 1,247 row(s) across 1 batch(es)
  Provenance Mode: Polynomial
  Schema: [amount:Float64, fraud_prob:Float64, merchant_type:Utf8]

  Score Distribution (fraud_prob):
     min=0.7012, max=0.9987, mean=0.8834, median=0.8921

─── Rules Applied ──────────────────────────────────────
  • high_risk       ← fraud_prob > 0.90 AND amount > 10000
  • velocity_check  ← amount > 5000 AND fraud_prob > 0.70

─── Models Used ────────────────────────────────────────
  • fraud_detector v1.0.0 (ONNX, avg_latency: 5.0ms)

─── Device Pool ────────────────────────────────────────
  • CPU: 8 cores active
  • Apple Metal: 1 device active (M2 GPU)
═══════════════════════════════════════════════════════════
```

### 2. Fine-Grained Lineage
A detailed, mathematical trace of every row in the result set back to its input row IDs and model version identifiers.

```
─── Per-Row Lineage ────────────────────────────────────
  Row 0: Derived via fraud_detector using model 'v1.0.0',
         sourced from [txn_38291, txn_38292]
  Row 1: Derived via fraud_detector using model 'v1.0.0',
         sourced from [txn_44107]
```

---

## Semantic Anomaly Monitor

Traditional databases silently return empty tables or biased predictions if a neural model degrades. AnamDB runs an in-line **Semantic Monitor** that scans output Arrow batches for statistical anomalies:

| Anomaly Pattern | Cause | Severity | Warning |
|:---|:---|:---|:---|
| **Low-Confidence Rate** | Model outputs fall below threshold | Warning | *100% of rows have fraud_prob below 0.5* |
| **Uniform Score Distribution** | Model output is flat/stuck | Warning / Critical | *All rows have identical predictions* |
| **Skewed / Empty Outputs** | Logic filter is overly restrictive | Info / Warning | *99% of input rows filtered out* |

---

## Interactive Triage

When the Semantic Monitor detects an anomaly, the database session pauses query execution and enters **Triage Mode**. Developers or administrators can resolve anomalies through a interactive loop:

```
═══ Semantic Anomaly Triage ═══

[Anomaly 1]
  Severity: WARNING
  100% of rows have fraud_prob below 0.5 (threshold: 80% max).
  Affected rows: 9,216
  Suggested: Consider using a higher-accuracy model.
  Action: RetryWithModel("fraud_detector")
```

The session prompts for one of four actions:
- `Accept`: Acknowledge the anomaly and continue execution.
- `Correct("...")`: Provide natural-language feedback (e.g., *"Filter out APAC region transactions"*). AnamDB translates this to a Datalog rule patch on-the-fly and runs the query again.
- `RetryWithModel("...")`: Dynamically swap the model operator for a different version on the Pareto frontier and re-run.
- `Abort`: Cancel execution immediately.

---

## Syntactic Self-Repair Loop

In addition to semantic checks, AnamDB runs a background self-repair agent to handle system and runtime exceptions automatically:

![Self-Repair Loop: the Query Engine reports an exception to the Self-Repair Agent, which diagnoses the issue, queries the Model Registry for alternatives, and swaps to a faster model before retrying.](/images/self-repair-sequence.png)

### Self-Repair Matrix

- **Recoverable Shape Mismatch**: Swap to compatible input shape model.
- **Recoverable Timeout / Deadline**: Swap to faster variant on the Pareto frontier.
- **Recoverable Missing Values**: Retry with adjusted null-imputation parameters.
- **Degraded Out-of-Memory (OOM)**: Swap to a quantized variant, or disable GPU dispatch and process on CPU.
- **Fatal Core Failures**: Escalate to the host application / developer.
