# CLI & REPL Reference

AnamDB includes a unified command-line tool `anam` (compiled from `crates/anam-cli`) that provides both an interactive REPL shell and a gRPC server.

---

## Server Mode

To start the database server, use the `serve` command. This launches the highly concurrent gRPC server ready to receive remote queries and synchronization changesets:

```bash
anam serve --port 8080
```

### Server Command Options
- `--port <PORT>`: The port to listen on (defaults to `8080`).
- `--host <IP>`: Host address to bind the server to (defaults to `127.0.0.1`).
- `--gpu`: Force enable GPU acceleration if compatible hardware (CUDA, Metal) is found.

---

## Interactive REPL

Run `anam` without arguments to enter the interactive SQL & shell runtime:

```bash
$ anam
anam> 
```

Inside the REPL, you can execute standard SQL queries directly or invoke metadata control operations using **dot-commands**.

---

## Command Reference

### `.ingest`
Ingests a raw CSV file and writes it as a versioned, Arrow-backed Lance dataset.
- **Syntax**: `.ingest <csv_path> [lance_path]`
- **Example**:
  ```sql
  anam> .ingest demo/data/transactions.csv demo/data/transactions.lance
  ✓ Ingested 100,000 rows (4.7 MB CSV → Lance columnar)
  ```

### `.load`
Loads a Lance table into the current query session catalog.
- **Syntax**: `.load <lance_path> [table_name]`
- **Example**:
  ```sql
  anam> .load demo/data/transactions.lance txns
  Registered table 'txns'
  ```

### `.model load`
Registers an ONNX model as an in-process **Function-as-Operator (FAO)**.
- **Syntax**: `.model load <onnx_path> <operator_name> <num_features> <latency_ms> <accuracy>`
- **Example**:
  ```sql
  anam> .model load demo/models/fraud_detector.onnx fraud_detector 3 5.0 0.95
  ✓ Loaded ONNX model 'fraud_detector'
  ```

### `.logic`
Compiles and registers a symbolic Datalog rule/constraint.
- **Syntax**: `.logic <rule_name> "<datalog_expression>"`
- **Example**:
  ```sql
  anam> .logic high_risk "fraud_prob > 0.90 AND amount > 10000"
  ✓ Registered rule 'high_risk'
  ```

### `.nl`
Translates a natural language prompt into a deterministic Datalog rule via an LLM.
- **Syntax**: `.nl <rule_name> <table_name> <english_description>`
- **Prerequisites**: Requires an `OPENAI_API_KEY` configured in the `.env` file.
- **Example**:
  ```sql
  anam> .nl night_risk txns Flag any transaction between midnight and 5am over $5000
  Compiling NL → Datalog via LLM...
  ✓ Generated and registered rule 'night_risk':
    Datalog: night_risk(X) :- txns(X), X.time >= '00:00', X.time < '05:00', X.amount > 5000.
  ```

### `.models` / `.operators`
Lists all loaded neural model operators registered in the AI-Tables catalog.
- **Example**:
  ```sql
  anam> .models
  • fraud_detector v1.0.0 [onnx] — 5.0ms, 0.95 accuracy
  • fraud_fast v1.0.0 [onnx]     — 0.5ms, 0.75 accuracy
  ```

### `.rules`
Lists all compiled and active symbolic Datalog constraints.
- **Example**:
  ```sql
  anam> .rules
  • high_risk       ← fraud_prob > 0.90 AND amount > 10000
  • night_risk      ← (LLM-generated Datalog)
  ```

### `.devices`
Displays the active device pool and current workload allocations.
- **Example**:
  ```sql
  anam> .devices
  [ 0–7] CPU-0..CPU-7 (1x) — idle
  [   8] Metal: Apple M2 (10x) — idle
  ```

### `.explain`
Prints the complete multi-dimensional reasoning trace for the most recently executed query. This trace shows provenance configuration, compiled logic rules, model statistics, the Pareto frontier, hardware utilization, and any semantic anomalies.
- **Example**: See [Explainability & HITL](/guide/explainability-hitl) for details.

### `.hub`
Manages downloading and searching for packages in the Community Hub registry.
- **Syntax**: 
  - `.hub search <query>`
  - `.hub install <package_id>`
- **Examples**:
  ```sql
  anam> .hub search fraud
  anam> .hub install anamdb/financial-compliance@1.0.0
  ```

### `.quit`
Closes the interactive shell session.
- **Syntax**: `.quit` or Ctrl+D.

---

## Executing SQL

You can run standard SQL commands directly. If your queries target tables with Datalog logic or neural operators defined on them, the engine transparently executes model inference and logic checks in-memory.

```sql
anam> SELECT region, COUNT(1) AS count FROM txns WHERE fraud_prob > 0.90 GROUP BY region;
+--------+-------+
| region | count |
+--------+-------+
| APAC   | 5321  |
| EU     | 36033 |
| US     | 48018 |
+--------+-------+
```
