//! Global Lineage & Decentralized HITL Triage.
//!
//! Extends the single-node provenance system to work across distributed nodes.
//! Provides cluster-wide lineage tracing and anomaly isolation that doesn't
//! block unaffected data paths.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, instrument, warn};

use crate::core::error::Result;
use crate::hitl::triage::Anomaly;

/// A provenance trace segment from a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTraceSegment {
    /// Node that produced this segment.
    pub node_id: String,
    /// Function/operator that ran on this node.
    pub function_id: String,
    /// Model version used.
    pub model_ver_id: String,
    /// Source record IDs consumed.
    pub source_records: Vec<String>,
    /// Intermediate confidence score produced.
    pub confidence: f64,
    /// Duration in ms.
    pub duration_ms: f64,
}

/// A complete cross-node lineage trace for a single result tuple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalLineageTrace {
    /// The final tuple ID being traced.
    pub tuple_id: String,
    /// Ordered list of trace segments (from input → output).
    pub segments: Vec<NodeTraceSegment>,
    /// Total end-to-end latency in ms.
    pub total_latency_ms: f64,
    /// Number of nodes traversed.
    pub hop_count: usize,
}

impl GlobalLineageTrace {
    /// Get a formatted display of the trace.
    pub fn display(&self) -> String {
        let mut output = format!(
            "═══ Global Lineage: {} ═══\n  Hops: {} | Total: {:.1}ms\n",
            self.tuple_id, self.hop_count, self.total_latency_ms
        );

        for (i, seg) in self.segments.iter().enumerate() {
            output.push_str(&format!(
                "\n  ┌─ Hop {} ─────────────────────────────────\n\
                 \x20 │ Node:       {}\n\
                 \x20 │ Operator:   {}\n\
                 \x20 │ Model:      {}\n\
                 \x20 │ Confidence: {:.4}\n\
                 \x20 │ Duration:   {:.2}ms\n\
                 \x20 │ Sources:    [{}]\n\
                 \x20 └──────────────────────────────────────────\n",
                i + 1,
                seg.node_id,
                seg.function_id,
                seg.model_ver_id,
                seg.confidence,
                seg.duration_ms,
                seg.source_records.join(", ")
            ));
        }
        output
    }
}

/// The Global Lineage Tracer — reconstructs cross-node provenance.
#[derive(Debug)]
pub struct GlobalLineageTracer {
    /// Trace segments indexed by tuple_id.
    traces: HashMap<String, Vec<NodeTraceSegment>>,
}

impl GlobalLineageTracer {
    /// Create a new tracer.
    pub fn new() -> Self {
        Self {
            traces: HashMap::new(),
        }
    }

    /// Record a trace segment from a node.
    pub fn record_segment(&mut self, tuple_id: &str, segment: NodeTraceSegment) {
        self.traces
            .entry(tuple_id.to_string())
            .or_default()
            .push(segment);
    }

    /// Reconstruct the full lineage for a tuple.
    #[instrument(skip(self))]
    pub fn trace(&self, tuple_id: &str) -> Result<GlobalLineageTrace> {
        let segments = self.traces.get(tuple_id).cloned().unwrap_or_default();

        let total_latency_ms: f64 = segments.iter().map(|s| s.duration_ms).sum();
        let hop_count = segments.len();

        info!(
            tuple_id = tuple_id,
            hops = hop_count,
            total_ms = total_latency_ms,
            "reconstructed global lineage"
        );

        Ok(GlobalLineageTrace {
            tuple_id: tuple_id.to_string(),
            segments,
            total_latency_ms,
            hop_count,
        })
    }

    /// List all traced tuple IDs.
    pub fn traced_tuples(&self) -> Vec<&str> {
        self.traces.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for GlobalLineageTracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Cluster-wide anomaly isolation status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolatedAnomaly {
    /// The anomaly detected.
    pub anomaly: Anomaly,
    /// Node where the anomaly was detected.
    pub node_id: String,
    /// Whether this data path is isolated (paused).
    pub isolated: bool,
    /// Other nodes continue processing unaffected.
    pub cluster_healthy: bool,
}

/// Cluster-Wide Agentic Monitor — detects and isolates anomalies per-node
/// without blocking the rest of the cluster.
#[derive(Debug)]
pub struct ClusterMonitor {
    /// Isolated anomalies by node.
    isolated: Vec<IsolatedAnomaly>,
}

impl ClusterMonitor {
    /// Create a new cluster monitor.
    pub fn new() -> Self {
        Self {
            isolated: Vec::new(),
        }
    }

    /// Isolate an anomalous data path on a specific node.
    #[instrument(skip(self, anomaly))]
    pub fn isolate_anomaly(&mut self, node_id: &str, anomaly: Anomaly) -> IsolatedAnomaly {
        warn!(
            node = node_id,
            severity = %anomaly.severity,
            "isolating anomalous data path"
        );

        let isolated = IsolatedAnomaly {
            anomaly,
            node_id: node_id.to_string(),
            isolated: true,
            cluster_healthy: true,
        };

        self.isolated.push(isolated.clone());
        isolated
    }

    /// Resume a previously isolated data path.
    pub fn resume(&mut self, node_id: &str) {
        for item in &mut self.isolated {
            if item.node_id == node_id && item.isolated {
                item.isolated = false;
                info!(node = node_id, "resumed isolated data path");
            }
        }
    }

    /// Get all currently isolated anomalies.
    pub fn active_isolations(&self) -> Vec<&IsolatedAnomaly> {
        self.isolated.iter().filter(|i| i.isolated).collect()
    }

    /// Get a formatted summary.
    pub fn summary(&self) -> String {
        let active = self.active_isolations();
        if active.is_empty() {
            return "═══ Cluster Monitor: All nodes healthy ═══".to_string();
        }

        let mut lines = vec![format!(
            "═══ Cluster Monitor: {} isolated path(s) ═══",
            active.len()
        )];
        for iso in &active {
            lines.push(format!(
                "  ⚠ Node {}: [{}] {}",
                iso.node_id, iso.anomaly.severity, iso.anomaly.description
            ));
        }
        lines.join("\n")
    }
}

impl Default for ClusterMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_lineage_trace() {
        let mut tracer = GlobalLineageTracer::new();

        tracer.record_segment(
            "txn_001",
            NodeTraceSegment {
                node_id: "edge-0".into(),
                function_id: "ocr_extract".into(),
                model_ver_id: "ocr_v2.1".into(),
                source_records: vec!["img_001".into()],
                confidence: 0.92,
                duration_ms: 15.0,
            },
        );

        tracer.record_segment(
            "txn_001",
            NodeTraceSegment {
                node_id: "core-0".into(),
                function_id: "fraud_detector".into(),
                model_ver_id: "fraud_v1.0".into(),
                source_records: vec!["txn_001_features".into()],
                confidence: 0.97,
                duration_ms: 5.0,
            },
        );

        let trace = tracer.trace("txn_001").unwrap();
        assert_eq!(trace.hop_count, 2);
        assert!((trace.total_latency_ms - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cluster_isolation() {
        let mut monitor = ClusterMonitor::new();

        let _iso = monitor.isolate_anomaly(
            "edge-1",
            Anomaly {
                description: "Uniform scores on edge-1".into(),
                affected_rows: 500,
                severity: crate::hitl::triage::AnomalySeverity::Critical,
                suggested_action: "Check model weights".into(),
            },
        );

        assert_eq!(monitor.active_isolations().len(), 1);
        assert!(monitor.summary().contains("edge-1"));

        monitor.resume("edge-1");
        assert_eq!(monitor.active_isolations().len(), 0);
    }
}
