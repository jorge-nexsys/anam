# What is AnamDB?

**AnamDB** is a vertical-agnostic, neurosymbolic database engine built in Rust. 

Unlike traditional vector databases that rely on semantic similarity, or bolt-on LLM architectures that treat databases as external memory, AnamDB treats **Models as First-Class Citizens** and **Logic as a Verifiable Blueprint**.

## Core Capabilities

1. **Explainability**: Semiring provenance ensures every result is traced back to its exact source.
2. **Safety**: Datalog guardrails block AI hallucinations natively.
3. **Optimization**: Calculates the Pareto frontier (latency × accuracy × cost) dynamically.
4. **Hardware-Aware**: Metal / CUDA / NPU heterogeneous dispatch.

## Interactive REPL

AnamDB comes with a powerful CLI out of the box.

```bash
cargo install anam-cli
anam
```

Once inside, you can query models just like tables:

```sql
anam> .model load fraud_detector.onnx fraud_detector 3 5.0 0.95
anam> .logic high_risk "fraud_prob > 0.90 AND amount > 10000"
anam> SELECT * FROM txns WHERE high_risk = true;
```
