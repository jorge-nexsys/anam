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
echo "  ║    Full Neurosymbolic Pipeline Demo                      ║"
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
echo "  Demo Complete ✓"
echo ""
echo "  To explore interactively:"
echo "    cargo run -- --gpu"
echo ""
echo "  Try these commands in the REPL:"
echo "    .nl suspicious transactions Flag late-night wire transfers over 50K"
echo "    SELECT * FROM txns WHERE fraud_prob > 0.90 WITH (max_latency_ms=50)"
echo "    .explain"
echo "══════════════════════════════════════════════════════════════"
