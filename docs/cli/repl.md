# Interactive REPL

Running the `anam` command-line utility with no arguments launches the interactive SQL shell and rule runtime:

```bash
$ anam

   A N A M D B  —  AI-Native Neurosymbolic Database Kernel
   Version: 1.0.0 (Rust)
   [Loaded 12 local rules, 2 ONNX model functions]

Type SQL queries, or use dot-commands:
  .load <path>    — Register a Lance table (streaming)
  .ingest <csv>   — Ingest CSV → Lance dataset
  .hub <action>   — AI-Tables Hub (search, install, list)
  .logic <n> <d>  — Register a Datalog rule
  .models         — List registered AI models
  .rules          — List Datalog rules
  .devices        — List available compute devices
  .explain        — Explain the last query's reasoning
  .help           — Show all commands
  .quit           — Exit

anam> 
```

---

## Interactive Features

* **Command History:** Use Up and Down arrow keys to navigate previous commands. History is persisted in your home directory (`~/.anam_history` or system-specific equivalents).
* **Dot-Commands:** Control metadata (like models, logic, and hardware configs) directly using dot-prefixed shell instructions.
* **SQL Console:** Execute queries against loaded Arrow and Lance tables directly in the REPL. Datalog rules and ONNX model evaluation run transparently inline inside the SQL execution plans.

---

## Command Reference Sections

To read more about specific groups of dot-commands, select a category below:

1. **[Loading & Ingestion](./repl/load-ingest.md):** CSV to Lance imports and registration.
2. **[Datalog Rules](./repl/logic-rules.md):** Dynamic reasoning guardrails and query constraints.
3. **[AI & Models](./repl/models.md):** ONNX operator management and Pareto latency metrics.
4. **[Explainability](./repl/explain.md):** Provenance tracing and reasoning graphs.
5. **[Package Hub](./repl/hub.md):** Downloading logic packs from the registry.
6. **[Device Pool](./repl/devices.md):** Hardware acceleration settings (Metal, CUDA, CPU).
