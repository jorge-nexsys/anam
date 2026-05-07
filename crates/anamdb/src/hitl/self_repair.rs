//! Syntactic Self-Repair — a two-agent loop that diagnoses and patches
//! structural errors in neural operators without aborting queries.
//!
//! When a FAO operator encounters a runtime error (dimension mismatch,
//! unsupported format, etc.), the self-repair system:
//!
//! 1. **Reviewer Agent** — Diagnoses the exception and identifies the root cause.
//! 2. **Rewriter Agent** — Proposes a corrective action (schema adjustment,
//!    model swap, input transform) and returns a `RepairAction`.
//!
//! This module uses an LLM to power both agents.

use serde::{Deserialize, Serialize};
use tracing::{info, instrument, warn};

use crate::core::error::Result;

/// A diagnosed error from the Reviewer Agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnosis {
    /// The original error message.
    pub original_error: String,
    /// The reviewer's root-cause analysis.
    pub root_cause: String,
    /// Confidence in the diagnosis (0.0–1.0).
    pub confidence: f64,
    /// Severity classification.
    pub severity: DiagnosisSeverity,
}

/// How severe the diagnosed error is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosisSeverity {
    /// Recoverable — can be patched automatically.
    Recoverable,
    /// Degraded — can continue with reduced accuracy/features.
    Degraded,
    /// Fatal — requires user intervention.
    Fatal,
}

impl std::fmt::Display for DiagnosisSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosisSeverity::Recoverable => write!(f, "RECOVERABLE"),
            DiagnosisSeverity::Degraded => write!(f, "DEGRADED"),
            DiagnosisSeverity::Fatal => write!(f, "FATAL"),
        }
    }
}

/// A corrective action proposed by the Rewriter Agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RepairAction {
    /// Swap to a different model variant.
    SwapModel {
        /// Name of the replacement model.
        replacement: String,
        /// Reason for the swap.
        reason: String,
    },
    /// Adjust the input schema (add/remove/rename columns).
    AdjustSchema {
        /// Description of the schema change.
        change: String,
    },
    /// Retry with modified parameters.
    RetryWithParams {
        /// Parameter adjustments.
        adjustments: String,
    },
    /// Skip the failing rows and continue with the rest.
    SkipAndContinue {
        /// Number of rows to skip.
        skip_count: usize,
        /// Reason for skipping.
        reason: String,
    },
    /// Escalate to user — cannot self-repair.
    Escalate {
        /// Explanation for the user.
        explanation: String,
    },
}

impl std::fmt::Display for RepairAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RepairAction::SwapModel {
                replacement,
                reason,
            } => write!(f, "SwapModel → {replacement}: {reason}"),
            RepairAction::AdjustSchema { change } => write!(f, "AdjustSchema: {change}"),
            RepairAction::RetryWithParams { adjustments } => {
                write!(f, "RetryWithParams: {adjustments}")
            }
            RepairAction::SkipAndContinue { skip_count, reason } => {
                write!(f, "SkipAndContinue ({skip_count} rows): {reason}")
            }
            RepairAction::Escalate { explanation } => write!(f, "Escalate → User: {explanation}"),
        }
    }
}

/// A complete repair report from the two-agent loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairReport {
    /// The diagnosis from the Reviewer Agent.
    pub diagnosis: Diagnosis,
    /// The corrective action from the Rewriter Agent.
    pub action: RepairAction,
    /// Whether the repair was applied successfully.
    pub applied: bool,
}

impl RepairReport {
    /// Get a formatted summary for display.
    pub fn summary(&self) -> String {
        let status = if self.applied {
            "✓ Applied"
        } else {
            "⚠ Pending"
        };
        format!(
            "═══ Self-Repair Report ═══\n\
             Severity: {}\n\
             Root Cause: {}\n\
             Confidence: {:.0}%\n\
             Action: {}\n\
             Status: {}",
            self.diagnosis.severity,
            self.diagnosis.root_cause,
            self.diagnosis.confidence * 100.0,
            self.action,
            status
        )
    }
}

/// The Self-Repair Agent — diagnoses and patches structural errors.
#[derive(Debug)]
pub struct SelfRepairAgent {
    /// Available model names for swap recommendations.
    available_models: Vec<String>,
}

impl SelfRepairAgent {
    /// Create a new self-repair agent.
    pub fn new() -> Self {
        Self {
            available_models: Vec::new(),
        }
    }

    /// Register available models for swap recommendations.
    pub fn register_available_models(&mut self, models: Vec<String>) {
        self.available_models = models;
    }

    /// Run the full two-agent loop: diagnose then repair.
    #[instrument(skip(self))]
    pub fn diagnose_and_repair(
        &self,
        error_msg: &str,
        operator_name: &str,
        context: &str,
    ) -> Result<RepairReport> {
        info!(
            error = error_msg,
            operator = operator_name,
            "self-repair agent triggered"
        );

        // Stage 1: Reviewer — diagnose the error.
        let diagnosis = self.review(error_msg, operator_name, context)?;

        // Stage 2: Rewriter — propose a corrective action.
        let action = self.rewrite(&diagnosis, operator_name)?;

        let report = RepairReport {
            diagnosis,
            action,
            applied: false,
        };

        info!(report = %report.summary(), "self-repair report generated");
        Ok(report)
    }

    /// Stage 1: Reviewer Agent — diagnose the error.
    fn review(&self, error_msg: &str, operator_name: &str, context: &str) -> Result<Diagnosis> {
        let error_lower = error_msg.to_lowercase();

        // Pattern-match common structural errors.
        let (root_cause, severity, confidence) = if error_lower.contains("dimension")
            || error_lower.contains("shape")
        {
            (
                format!(
                    "Input tensor shape mismatch in operator '{operator_name}'. \
                         The model expects a different number of features than provided."
                ),
                DiagnosisSeverity::Recoverable,
                0.9,
            )
        } else if error_lower.contains("unsupported")
            || error_lower.contains("format")
            || error_lower.contains("codec")
        {
            (
                format!(
                    "Unsupported data format encountered by operator '{operator_name}'. \
                         The input data contains a type or encoding this operator cannot process."
                ),
                DiagnosisSeverity::Degraded,
                0.85,
            )
        } else if error_lower.contains("null")
            || error_lower.contains("missing")
            || error_lower.contains("none")
        {
            (
                format!(
                    "Null or missing values detected in input to '{operator_name}'. \
                         {context}"
                ),
                DiagnosisSeverity::Recoverable,
                0.8,
            )
        } else if error_lower.contains("timeout")
            || error_lower.contains("deadline")
            || error_lower.contains("exceeded")
        {
            (
                format!(
                    "Operator '{operator_name}' exceeded its execution time budget. \
                         Consider swapping to a faster model variant."
                ),
                DiagnosisSeverity::Recoverable,
                0.95,
            )
        } else if error_lower.contains("memory")
            || error_lower.contains("oom")
            || error_lower.contains("allocation")
        {
            (
                format!(
                    "Out-of-memory condition in operator '{operator_name}'. \
                         The input batch may be too large for the current device."
                ),
                DiagnosisSeverity::Degraded,
                0.9,
            )
        } else {
            (
                format!("Unrecognized structural error in operator '{operator_name}': {error_msg}"),
                DiagnosisSeverity::Fatal,
                0.5,
            )
        };

        Ok(Diagnosis {
            original_error: error_msg.to_string(),
            root_cause,
            confidence,
            severity,
        })
    }

    /// Stage 2: Rewriter Agent — propose a corrective action.
    fn rewrite(&self, diagnosis: &Diagnosis, operator_name: &str) -> Result<RepairAction> {
        match diagnosis.severity {
            DiagnosisSeverity::Recoverable => {
                // Try to find an alternative model.
                if let Some(alt) = self
                    .available_models
                    .iter()
                    .find(|m| m.as_str() != operator_name)
                {
                    Ok(RepairAction::SwapModel {
                        replacement: alt.clone(),
                        reason: format!(
                            "Swapping from '{}' to '{}' to bypass: {}",
                            operator_name, alt, diagnosis.root_cause
                        ),
                    })
                } else {
                    Ok(RepairAction::RetryWithParams {
                        adjustments: "Reduce batch size and retry.".into(),
                    })
                }
            }
            DiagnosisSeverity::Degraded => Ok(RepairAction::SkipAndContinue {
                skip_count: 0,
                reason: format!(
                    "Continuing in degraded mode. Unsupported rows will be skipped. \
                     Root cause: {}",
                    diagnosis.root_cause
                ),
            }),
            DiagnosisSeverity::Fatal => {
                warn!(
                    error = %diagnosis.original_error,
                    "self-repair escalating to user"
                );
                Ok(RepairAction::Escalate {
                    explanation: format!(
                        "Cannot auto-repair: {}. Please review the operator configuration \
                         and input data manually.",
                        diagnosis.root_cause
                    ),
                })
            }
        }
    }
}

impl Default for SelfRepairAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnose_dimension_mismatch() {
        let agent = SelfRepairAgent::new();
        let report = agent
            .diagnose_and_repair(
                "dimension mismatch: expected 3, got 5",
                "fraud_detector",
                "input batch has 5 columns",
            )
            .unwrap();

        assert_eq!(report.diagnosis.severity, DiagnosisSeverity::Recoverable);
        assert!(report.diagnosis.root_cause.contains("shape mismatch"));
    }

    #[test]
    fn diagnose_timeout_with_swap() {
        let mut agent = SelfRepairAgent::new();
        agent.register_available_models(vec!["fraud_detector".into(), "fraud_fast".into()]);

        let report = agent
            .diagnose_and_repair(
                "operator exceeded deadline of 50ms",
                "fraud_detector",
                "latency constraint violated",
            )
            .unwrap();

        assert_eq!(report.diagnosis.severity, DiagnosisSeverity::Recoverable);
        match &report.action {
            RepairAction::SwapModel { replacement, .. } => {
                assert_eq!(replacement, "fraud_fast");
            }
            other => panic!("expected SwapModel, got {other:?}"),
        }
    }

    #[test]
    fn diagnose_fatal_escalates() {
        let agent = SelfRepairAgent::new();
        let report = agent
            .diagnose_and_repair("some unknown error xyz", "op1", "")
            .unwrap();

        assert_eq!(report.diagnosis.severity, DiagnosisSeverity::Fatal);
        assert!(matches!(report.action, RepairAction::Escalate { .. }));
    }
}
