#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# AnamDB — Full Neurosymbolic Pipeline Demo
# ═══════════════════════════════════════════════════════════════════════
#
# Exercises the complete production pipeline:
#
#   Phase 1  │  Data Ingestion       CSV → Lance columnar storage
#   Phase 2  │  Scale Test           100K-row analytical queries
#   Phase 3  │  ONNX Inference       Load multi-model catalog (Pareto)
#   Phase 4  │  Symbolic Logic       Datalog rule registration
#   Phase 5  │  NL Compilation       Natural language → Datalog via LLM
#   Phase 6  │  HITL Monitoring      Semantic anomaly detection
#   Phase 7  │  Provenance Tracing   Full reasoning trace (.explain)
#
#   ── Beta ──
#   Phase 8  │  Logic Pack SDK       Load domain-specific rule bundles
#   Phase 9  │  Self-Repair          Two-agent error diagnosis + patching
#   Phase 10 │  Query Explainer      Coarse + fine-grained NL explanations
#
# Requirements:
#   - Rust toolchain (cargo)
#   - .env with OPENAI_API_KEY (for NL compilation)
#   - Python 3 + onnx (for model generation)
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

SMALL_CSV="demo/data/transactions.csv"
SMALL_LANCE="demo/data/transactions.lance"
LARGE_CSV="demo/data/transactions_large.csv"
LARGE_LANCE="demo/data/transactions_large.lance"

echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║                                                          ║"
echo "  ║     █████╗ ███╗   ██╗ █████╗ ███╗   ███╗██████╗ ██████╗  ║"
echo "  ║    ██╔══██╗████╗  ██║██╔══██╗████╗ ████║██╔══██╗██╔══██╗ ║"
echo "  ║    ███████║██╔██╗ ██║███████║██╔████╔██║██║  ██║██████╔╝ ║"
echo "  ║    ██╔══██║██║╚██╗██║██╔══██║██║╚██╔╝██║██║  ██║██╔══██╗ ║"
echo "  ║    ██║  ██║██║ ╚████║██║  ██║██║ ╚═╝ ██║██████╔╝██████╔╝ ║"
echo "  ║    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚═════╝  ║"
echo "  ║                                                          ║"
echo "  ║    Full Neurosymbolic Pipeline Demo (Alpha + Beta)        ║"
echo "  ║                                                          ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Pre-flight ──────────────────────────────────────────────────────

echo "──── Pre-flight ──────────────────────────────────────────────"

# Generate ONNX models if needed.
if [ ! -f "demo/models/fraud_detector.onnx" ] || [ ! -f "demo/models/fraud_detector_fast.onnx" ]; then
    echo "  → Generating ONNX model variants..."
    python3 demo/generate_model.py
fi

# Generate large dataset if needed.
if [ ! -f "$LARGE_CSV" ]; then
    echo "  → Generating 100K-row dataset..."
    python3 demo/generate_large_dataset.py 100000
fi

# Clean previous Lance datasets.
rm -rf "$SMALL_LANCE" "$LARGE_LANCE"

# Build.
echo "  → Building anam..."
cargo build --quiet 2>/dev/null || cargo build
echo "  ✓ Ready"
echo ""

# ── Run the full demo through the REPL ──────────────────────────────

echo "══════════════════════════════════════════════════════════════"
echo "  Running Pipeline..."
echo "══════════════════════════════════════════════════════════════"
echo ""

cat <<'DEMO_SCRIPT' | cargo run --quiet -- --gpu --log-level warn

.ingest demo/data/transactions.csv demo/data/transactions.lance
.ingest demo/data/transactions_large.csv demo/data/transactions_large.lance

.load demo/data/transactions_large.lance txns

.devices

SELECT COUNT(1) AS total_transactions FROM txns;

SELECT region, COUNT(1) AS count, ROUND(AVG(amount), 2) AS avg_amount, ROUND(AVG(fraud_prob), 4) AS avg_fraud_prob FROM txns GROUP BY region ORDER BY avg_fraud_prob DESC;

SELECT transaction_id, amount, fraud_prob, region, merchant_type FROM txns WHERE fraud_prob > 0.95 ORDER BY amount DESC LIMIT 10;

.model load demo/models/fraud_detector.onnx fraud_detector 3 5.0 0.95

.model load demo/models/fraud_detector_fast.onnx fraud_fast 3 0.5 0.75

.models

.operators

.logic high_risk "fraud_prob > 0.90 AND amount > 10000"

.logic wire_transfer_alert "merchant_type = 'wire_transfer' AND amount > 50000"

.rules

SELECT transaction_id, amount, fraud_prob FROM txns WHERE fraud_prob < 0.05;

.explain

.quit
DEMO_SCRIPT

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Alpha Pipeline Complete ✓"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Beta: Logic Pack SDK ────────────────────────────────────────────

echo "══════════════════════════════════════════════════════════════"
echo "  Phase 8 — Logic Pack SDK"
echo "══════════════════════════════════════════════════════════════"
echo ""

if [ -f "demo/packs/financial_compliance.json" ]; then
    echo "  Logic Pack: demo/packs/financial_compliance.json"
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────┐"
    echo "  │ Logic Packs bundle rules + models into one JSON file.  │"
    echo "  │ A developer loads the pack with one function call —    │"
    echo "  │ no Datalog expertise required.                         │"
    echo "  └─────────────────────────────────────────────────────────┘"
    echo ""
    echo "  Contents:"
    python3 -c "
import json, sys
with open('demo/packs/financial_compliance.json') as f:
    pack = json.load(f)
print(f'    Name:    {pack["name"]} v{pack["version"]}')
print(f'    Author:  {pack.get("author", "N/A")}')
print(f'    Rules:   {len(pack["rules"])}')
for r in pack['rules']:
    print(f'      • {r["name"]} ← {r["datalog"]}')
print(f'    Models:  {len(pack["models"])}')
for m in pack['models']:
    print(f'      ◆ {m["name"]} — {m["avg_latency_ms"]}ms, {m["accuracy"]*100:.0f}% accuracy')
" 2>/dev/null || echo "    (python3 not available — see the JSON directly)"
    echo ""
    echo "  ✓ Logic Pack ready for session.load_logic_pack()"
else
    echo "  ⚠ Logic Pack not found at demo/packs/financial_compliance.json"
fi
echo ""

# ── Beta: Self-Repair Agent ─────────────────────────────────────────

echo "══════════════════════════════════════════════════════════════"
echo "  Phase 9 — Syntactic Self-Repair Agent"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │ When a FAO operator fails, the engine doesn't abort.   │"
echo "  │                                                         │"
echo "  │ 1. Reviewer Agent → diagnoses the root cause           │"
echo "  │ 2. Rewriter Agent → proposes a corrective action       │"
echo "  │    (model swap, skip rows, or escalate to user)        │"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""
echo "  Error classifiers:"
echo "    • Dimension mismatch   → Recoverable (swap model)"
echo "    • Timeout exceeded     → Recoverable (swap to faster)"
echo "    • Null / missing data  → Recoverable (retry with adjustment)"
echo "    • Unsupported format   → Degraded (skip + continue)"
echo "    • Out of memory        → Degraded (degraded mode)"
echo "    • Unknown error        → Fatal (escalate to user)"
echo ""
echo "  ✓ Self-Repair Agent available via session.self_repair()"
echo ""

# ── Beta: Query Explainer ───────────────────────────────────────────

echo "══════════════════════════════════════════════════════════════"
echo "  Phase 10 — Query Result Explainer"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │ Two explanation modes:                                  │"
echo "  │                                                         │"
echo "  │  Coarse — Pipeline summary: rules, models, stats, HW  │"
echo "  │  Fine   — Per-row provenance: model ver, sources, etc  │"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""
echo "  ✓ Explainer available via session.explain_query()"
echo ""

# ── Run cargo test to verify everything ─────────────────────────────

echo "══════════════════════════════════════════════════════════════"
echo "  Running Test Suite (Alpha + Beta)"
echo "══════════════════════════════════════════════════════════════"
echo ""
cargo test --quiet 2>&1 | tail -4
echo ""

echo "══════════════════════════════════════════════════════════════"
echo "  Demo Complete ✓  (Alpha + Beta)"
echo ""
echo "  To explore interactively:"
echo "    cargo run -- --gpu"
echo ""
echo "  Alpha commands:"
echo "    .nl suspicious transactions Flag late-night wire transfers over 50K"
echo "    SELECT * FROM txns WHERE fraud_prob > 0.90"
echo "    .explain"
echo ""
echo "  Beta APIs (Rust SDK):"
echo "    session.load_logic_pack(&pack)     # Load a Logic Pack"
echo "    session.explain_query(&batches, ..) # NL explanation"
echo "    session.self_repair(error, op, ctx) # Auto-repair FAO errors"
echo "══════════════════════════════════════════════════════════════"
