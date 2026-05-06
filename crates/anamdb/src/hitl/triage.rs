//! Interactive triage: types and logic for HITL anomaly resolution.

use serde::{Deserialize, Serialize};

/// Severity level of a detected semantic anomaly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalySeverity {
    /// Informational — the query completed but results look unusual.
    Info,
    /// Warning — results may be unreliable; user review recommended.
    Warning,
    /// Critical — the model or logic is likely producing garbage; pause execution.
    Critical,
}

impl std::fmt::Display for AnomalySeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalySeverity::Info => write!(f, "INFO"),
            AnomalySeverity::Warning => write!(f, "WARNING"),
            AnomalySeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A semantic anomaly detected by the [`SemanticMonitor`](super::monitor::SemanticMonitor).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Human-readable description of the anomaly.
    pub description: String,
    /// Number of rows affected.
    pub affected_rows: usize,
    /// Severity level.
    pub severity: AnomalySeverity,
    /// Suggested corrective action.
    pub suggested_action: String,
}

impl std::fmt::Display for Anomaly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} ({} rows affected)\n  → {}",
            self.severity, self.description, self.affected_rows, self.suggested_action
        )
    }
}

/// The outcome of a triage interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriageAction {
    /// Accept the results as-is despite the anomaly.
    Accept,
    /// Provide a natural-language correction to refine the query.
    Correct(String),
    /// Abort the query entirely.
    Abort,
    /// Retry with a different model.
    RetryWithModel(String),
}

/// A triage session manages the interactive resolution of anomalies.
#[derive(Debug)]
pub struct TriageSession {
    /// The anomalies to resolve.
    pub anomalies: Vec<Anomaly>,
    /// Collected actions from the user.
    pub actions: Vec<TriageAction>,
}

impl TriageSession {
    /// Create a new triage session.
    pub fn new(anomalies: Vec<Anomaly>) -> Self {
        Self {
            anomalies,
            actions: Vec::new(),
        }
    }

    /// Record a user's triage action.
    pub fn record_action(&mut self, action: TriageAction) {
        self.actions.push(action);
    }

    /// Check if all anomalies have been addressed.
    pub fn is_complete(&self) -> bool {
        self.actions.len() >= self.anomalies.len()
    }

    /// Get a formatted summary for display.
    pub fn summary(&self) -> String {
        let mut lines = vec!["═══ Semantic Anomaly Triage ═══".to_string()];
        for (i, anomaly) in self.anomalies.iter().enumerate() {
            lines.push(format!("\n[Anomaly {}]", i + 1));
            lines.push(format!("  Severity: {}", anomaly.severity));
            lines.push(format!("  {}", anomaly.description));
            lines.push(format!("  Affected rows: {}", anomaly.affected_rows));
            lines.push(format!("  Suggested: {}", anomaly.suggested_action));
            if let Some(action) = self.actions.get(i) {
                lines.push(format!("  Action: {:?}", action));
            }
        }
        lines.join("\n")
    }
}
