//! Semantic anomaly monitor.
//!
//! Inspects intermediate `RecordBatch` results between operators, flagging
//! semantic anomalies — results that are syntactically valid but logically
//! unexpected or contradictory.

use datafusion::arrow::array::{Array, Float64Array, RecordBatch};
use tracing::{debug, warn};

use crate::core::error::Result;
use crate::hitl::triage::Anomaly;

/// Heuristic anomaly detectors.
#[derive(Debug, Clone, Copy)]
pub enum AnomalyHeuristic {
    /// Flag when > X% of rows have confidence below a threshold.
    LowConfidenceRate {
        /// Column name containing confidence scores.
        confidence_column: &'static str,
        /// Minimum acceptable confidence.
        confidence_threshold: f64,
        /// Maximum percentage of low-confidence rows before flagging.
        max_low_pct: f64,
    },
    /// Flag when the result set is unexpectedly empty.
    EmptyResultSet,
    /// Flag when all rows have identical scores (suspiciously uniform).
    UniformScores {
        /// Column name to check.
        score_column: &'static str,
    },
}

/// The semantic monitor inspects intermediate results and detects anomalies.
#[derive(Debug)]
pub struct SemanticMonitor {
    /// Global anomaly threshold (0.0–1.0).
    #[allow(dead_code)]
    threshold: f64,
    /// Registered heuristics.
    heuristics: Vec<AnomalyHeuristic>,
}

impl SemanticMonitor {
    /// Create a new monitor with default heuristics.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            heuristics: vec![
                AnomalyHeuristic::LowConfidenceRate {
                    confidence_column: "confidence",
                    confidence_threshold: 0.5,
                    max_low_pct: 0.8,
                },
                AnomalyHeuristic::EmptyResultSet,
                AnomalyHeuristic::UniformScores {
                    score_column: "confidence",
                },
            ],
        }
    }

    /// Inspect a set of batches and return any detected anomalies.
    pub fn inspect_batches(&self, batches: &[RecordBatch]) -> Result<Vec<Anomaly>> {
        let mut anomalies = Vec::new();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        if total_rows == 0 {
            anomalies.push(Anomaly {
                description: "Query returned zero rows. This may indicate overly restrictive \
                              constraints or missing data."
                    .into(),
                affected_rows: 0,
                severity: crate::hitl::triage::AnomalySeverity::Warning,
                suggested_action: "Relax the filter conditions or verify the data source.".into(),
            });
            return Ok(anomalies);
        }

        for batch in batches {
            for heuristic in &self.heuristics {
                if let Some(anomaly) = self.run_heuristic(heuristic, batch) {
                    anomalies.push(anomaly);
                }
            }
        }

        if !anomalies.is_empty() {
            warn!(
                count = anomalies.len(),
                "semantic monitor detected anomalies"
            );
        }

        Ok(anomalies)
    }

    fn run_heuristic(&self, heuristic: &AnomalyHeuristic, batch: &RecordBatch) -> Option<Anomaly> {
        match heuristic {
            AnomalyHeuristic::LowConfidenceRate {
                confidence_column,
                confidence_threshold,
                max_low_pct,
            } => {
                if let Some((col_idx, _)) = batch.schema().column_with_name(confidence_column) {
                    let col = batch.column(col_idx);
                    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        let num_rows = arr.len();
                        if num_rows == 0 {
                            return None;
                        }
                        let nulls = arr.nulls();
                        let low_count = (0..num_rows)
                            .filter(|&i| {
                                let valid = nulls.map_or(true, |n| n.is_valid(i));
                                valid && arr.value(i) < *confidence_threshold
                            })
                            .count();
                        let low_pct = low_count as f64 / num_rows as f64;

                        if low_pct > *max_low_pct {
                            return Some(Anomaly {
                                description: format!(
                                    "{:.0}% of rows have {confidence_column} below {confidence_threshold} \
                                     (threshold: {:.0}% max).",
                                    low_pct * 100.0,
                                    max_low_pct * 100.0
                                ),
                                affected_rows: low_count,
                                severity: crate::hitl::triage::AnomalySeverity::Warning,
                                suggested_action: format!(
                                    "Consider using a higher-accuracy model. Current low-confidence rate: {:.1}%.",
                                    low_pct * 100.0
                                ),
                            });
                        }
                    }
                }
                None
            }

            AnomalyHeuristic::EmptyResultSet => None,

            AnomalyHeuristic::UniformScores { score_column } => {
                if let Some((col_idx, _)) = batch.schema().column_with_name(score_column) {
                    let col = batch.column(col_idx);
                    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        let num_rows = arr.len();
                        if num_rows < 2 {
                            return None;
                        }

                        let first = arr.value(0);
                        let nulls = arr.nulls();
                        let all_same = (1..num_rows).all(|i| {
                            let valid = nulls.map_or(true, |n| n.is_valid(i));
                            !valid || (arr.value(i) - first).abs() < f64::EPSILON
                        });

                        if all_same {
                            return Some(Anomaly {
                                description: format!(
                                    "All {num_rows} rows in '{score_column}' have identical scores ({first:.4})."
                                ),
                                affected_rows: num_rows,
                                severity: crate::hitl::triage::AnomalySeverity::Critical,
                                suggested_action: "Verify the model is correctly loaded and the \
                                                   input features are meaningful."
                                    .into(),
                            });
                        }
                    }
                }
                None
            }
        }
    }
}
