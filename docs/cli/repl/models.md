# REPL: AI & Model Management

Manage ONNX models and inspect target model metadata in the Pareto-frontier optimizer.

---

## `.model load`
Registers a local ONNX model graph into the session context as an in-process **Function-as-Operator (FAO)**.

* **Syntax**: `.model load <onnx_path> <name> <features> <latency_ms> <accuracy>`
* **Example**:
  ```sql
  anam> .model load demo/models/fraud_detector.onnx fraud_detector 3 5.0 0.95
  ✓ Loaded ONNX model 'fraud_detector'
  ```

---

## `.models`
Lists all loaded neural models currently registered in the catalog.

* **Example**:
  ```sql
  anam> .models
  +----------+----------------+---------+--------+--------------+----------+
  | ID       | Name           | Version | Format | Latency (ms) | Accuracy |
  +----------+----------------+---------+--------+--------------+----------+
  | 8d39f4e2 | fraud_detector | 1.0.0   | ONNX   | 5.0          | 0.95     |
  +----------+----------------+---------+--------+--------------+----------+
  ```

---

## `.operators`
Lists registered Function-as-Operator entries mapping to SQL execution engines.

* **Example**:
  ```sql
  anam> .operators
  +--------------+---------+----------+---------+----------+
  | Function     | Version | Model    | Latency | Accuracy |
  +--------------+---------+----------+---------+----------+
  | fraud_verify | 1.0.0   | 8d39f4e2 | 5.0ms   | 0.95     |
  +--------------+---------+----------+---------+----------+
  ```
