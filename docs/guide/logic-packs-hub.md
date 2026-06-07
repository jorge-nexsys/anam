# Logic Packs & Hub

AnamDB decouples domain expertise from core engine updates using **Logic Packs**—modular, self-contained packages containing versioned rules, metadata, and neural model references. 

Developers publish these packs to the **AI-Tables Community Hub**, allowing other team members or external systems to install specialized logic bundles in a single command.

---

## What is a Logic Pack?

Instead of registering individual models and typing Datalog statements manually, you can load a Logic Pack JSON file. 

For example, a security audit team can write a single compliance pack, and engineers can load it programmatically or via the CLI to secure their workflows.

### Logic Pack Schema Reference

A Logic Pack is defined as a JSON object matching this schema:

```json
{
  "name": "company/financial-compliance",
  "version": "1.0.0",
  "description": "AML/KYC transaction compliance rules & models",
  "author": "SecOps Auditing Team",
  "rules": [
    {
      "name": "high_risk",
      "datalog": "fraud_prob > 0.90 AND amount > 10000",
      "description": "Flags high-value transactions with suspicious probability"
    },
    {
      "name": "wire_alert",
      "datalog": "merchant_type = 'wire_transfer' AND amount > 50000",
      "description": "Flags large wire transfers for secondary review"
    }
  ],
  "models": [
    {
      "name": "fraud_detector",
      "artifact_path": "demo/models/fraud_detector.onnx",
      "num_features": 3,
      "avg_latency_ms": 5.0,
      "accuracy": 0.95,
      "description": "Standard neural network for transaction scoring"
    },
    {
      "name": "fraud_fast",
      "artifact_path": "demo/models/fraud_detector_fast.onnx",
      "num_features": 3,
      "avg_latency_ms": 0.5,
      "accuracy": 0.75,
      "description": "Lightweight model for low-latency pre-filtering"
    }
  ]
}
```

---

## Creating Logic Packs Programmatically

If you are developing in Rust, you can use the `LogicPackBuilder` API to construct and serialize packs:

```rust
use anamdb::sdk::LogicPackBuilder;

let pack = LogicPackBuilder::new("company/financial-compliance", "1.0.0")
    .description("AML/KYC transaction compliance rules & models")
    .author("SecOps Auditing Team")
    .rule("high_risk", "fraud_prob > 0.90 AND amount > 10000")
    .rule("wire_alert", "merchant_type = 'wire_transfer' AND amount > 50000")
    .model_ref("fraud_detector", "demo/models/fraud_detector.onnx", 3, 5.0, 0.95)
    .model_ref("fraud_fast", "demo/models/fraud_detector_fast.onnx", 3, 0.5, 0.75)
    .build();

// Serialize the pack to a JSON file.
let json_str = pack.to_json()?;
std::fs::write("compliance_pack.json", json_str)?;
```

---

## Community Hub Package Registry

The Community Hub manages pack registries. AnamDB is configured to point to a registry index URL (defaults to the official repository index).

### CLI Package Management

Use the `anam` command-line utility or the REPL console to interact with the Hub:

1. **Search for Packs**:
   Search keyword matches package names, descriptions, or tags.
   ```bash
   anam hub search fraud
   # Or inside REPL:
   anam> .hub search fraud
   ```

2. **Install a Pack**:
   Downloads the pack, registers its ONNX models, and mounts the Datalog constraints.
   ```bash
   anam hub install anamdb/financial-compliance@1.0.0
   # Or inside REPL:
   anam> .hub install anamdb/financial-compliance@1.0.0
   ```

3. **Publish a Pack**:
   Pushes a local directory (containing a `manifest.json` and ONNX model files) to the index.
   ```bash
   anam hub publish ./my-custom-pack/
   # Or inside REPL:
   anam> .hub publish ./my-custom-pack/
   ```

4. **List Installed Packages**:
   Show local installed inventory.
   ```bash
   anam hub list
   ```

---

## SQL Hub Access

For systems executing queries over remote clients, the registry can be managed using standard SQL commands:

```sql
-- Search for a package
SELECT hub_search('imaging') AS results;

-- Install a package programmatically
SELECT hub_install('anamdb/medical-imaging@0.9.0') AS status;
```
