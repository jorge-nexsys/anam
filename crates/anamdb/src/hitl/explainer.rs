//! Query Result Explainer — generates natural-language explanations of query
//! results using provenance traces.
//!
//! Supports two modes:
//! - **Coarse-grained**: Summarizes the physical plan, models used, and rules applied.
//! - **Fine-grained**: Traces a specific row's provenance polynomial to its
//!   exact source records, model versions, and intermediate confidence scores.

use datafusion::arrow::array::{Array, BinaryArray, Float64Array, RecordBatch};
use tracing::{debug, instrument};

use crate::core::error::Result;
use crate::core::provenance::{PolynomialSemiring, Semiring};

/// Granularity level for explanations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplainLevel {
    /// High-level summary of the pipeline.
    Coarse,
    /// Per-row provenance tracing.
    Fine,
}

/// Context provided to the explainer for generating explanations.
#[derive(Debug, Clone)]
pub struct ExplainContext {
    /// Active Datalog rules.
    pub rules: Vec<(String, String)>,
    /// Registered model names and versions.
    pub models: Vec<(String, String)>,
    /// Provenance mode name.
    pub provenance_mode: String,
    /// Device pool summary.
    pub device_summary: String,
}

/// A single row-level explanation.
#[derive(Debug, Clone)]
pub struct RowExplanation {
    /// Row index.
    pub row: usize,
    /// Source records that contributed to this row.
    pub source_records: Vec<String>,
    /// Model version that produced this row.
    pub model_version: String,
    /// Function ID used.
    pub function_id: String,
    /// Natural-language explanation.
    pub explanation: String,
}

/// A complete explanation of a query result.
#[derive(Debug, Clone)]
pub struct QueryExplanation {
    /// The explanation level used.
    pub level: ExplainLevel,
    /// Coarse-grained summary (always present).
    pub summary: String,
    /// Fine-grained per-row explanations (only for Fine level).
    pub row_explanations: Vec<RowExplanation>,
}

impl QueryExplanation {
    /// Get a formatted display of the explanation.
    pub fn display(&self) -> String {
        let mut output = String::new();
        output.push_str("═══════════════════════════════════════════════════════════\n");
        output.push_str("  AnamDB Query Explanation\n");
        output.push_str("═══════════════════════════════════════════════════════════\n\n");
        output.push_str(&self.summary);

        if !self.row_explanations.is_empty() {
            output.push_str("\n\n─── Per-Row Lineage ────────────────────────────────────\n");
            for row_exp in &self.row_explanations {
                output.push_str(&format!(
                    "\n  Row {}: {}\n",
                    row_exp.row, row_exp.explanation
                ));
                output.push_str(&format!(
                    "    Model: {} ({})\n",
                    row_exp.model_version, row_exp.function_id
                ));
                if !row_exp.source_records.is_empty() {
                    output.push_str(&format!(
                        "    Sources: {}\n",
                        row_exp.source_records.join(", ")
                    ));
                }
            }
        }

        output.push_str("\n═══════════════════════════════════════════════════════════\n");
        output
    }
}

/// The Query Result Explainer.
#[derive(Debug)]
pub struct Explainer;

impl Explainer {
    /// Generate an explanation for the given batches.
    #[instrument(skip(batches, context))]
    pub fn explain(
        level: ExplainLevel,
        batches: &[RecordBatch],
        context: &ExplainContext,
    ) -> Result<QueryExplanation> {
        let summary = Self::generate_coarse_summary(batches, context);

        let row_explanations = if level == ExplainLevel::Fine {
            Self::generate_fine_explanations(batches)?
        } else {
            Vec::new()
        };

        Ok(QueryExplanation {
            level,
            summary,
            row_explanations,
        })
    }

    /// Generate a coarse-grained summary of the query.
    fn generate_coarse_summary(batches: &[RecordBatch], context: &ExplainContext) -> String {
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let total_batches = batches.len();

        let mut summary = String::new();

        // Pipeline overview.
        summary.push_str("─── Pipeline Summary ───────────────────────────────────\n");
        summary.push_str(&format!(
            "  Produced {} row(s) across {} batch(es)\n",
            total_rows, total_batches
        ));
        summary.push_str(&format!("  Provenance: {}\n", context.provenance_mode));

        // Schema.
        if let Some(batch) = batches.first() {
            let fields: Vec<String> = batch
                .schema()
                .fields()
                .iter()
                .map(|f| format!("{}:{}", f.name(), f.data_type()))
                .collect();
            summary.push_str(&format!("  Schema: [{}]\n", fields.join(", ")));
        }

        // Score distribution.
        for batch in batches {
            for col_name in &["fraud_prob", "confidence", "score"] {
                let col_schema = batch.schema();
                if let Some((idx, _)) = col_schema.column_with_name(col_name)
                    && let Some(arr) = batch.column(idx).as_any().downcast_ref::<Float64Array>()
                {
                    let stats = Self::compute_stats(arr);
                    summary.push_str(&format!(
                        "\n  Score Distribution ({col_name}):\n\
                         \x20   min={:.4}, max={:.4}, mean={:.4}, median={:.4}\n",
                        stats.min, stats.max, stats.mean, stats.median
                    ));
                }
            }
        }

        // Rules applied.
        if !context.rules.is_empty() {
            summary.push_str("\n─── Rules Applied ──────────────────────────────────────\n");
            for (name, body) in &context.rules {
                summary.push_str(&format!("  • {name} ← {body}\n"));
            }
            summary.push_str(
                "\n  These rules filtered the input data, retaining only rows\n  \
                 where ALL conditions were simultaneously satisfied.\n",
            );
        }

        // Models used.
        if !context.models.is_empty() {
            summary.push_str("\n─── Models Used ────────────────────────────────────────\n");
            for (name, version) in &context.models {
                summary.push_str(&format!("  • {name} v{version}\n"));
            }
            summary.push_str(
                "\n  The Pareto optimizer selected these models based on your\n  \
                 latency/accuracy constraints from the AI-Tables catalog.\n",
            );
        }

        // Hardware.
        if !context.device_summary.is_empty() {
            summary.push_str("\n─── Hardware ───────────────────────────────────────────\n");
            summary.push_str(&format!("  {}\n", context.device_summary));
        }

        summary
    }

    /// Generate fine-grained per-row explanations from provenance columns.
    fn generate_fine_explanations(batches: &[RecordBatch]) -> Result<Vec<RowExplanation>> {
        let mut explanations = Vec::new();
        let mut global_row = 0;

        for batch in batches {
            let schema = batch.schema();
            let prov_col = schema.column_with_name("provenance");

            for row in 0..batch.num_rows() {
                let mut exp = RowExplanation {
                    row: global_row,
                    source_records: Vec::new(),
                    model_version: "unknown".into(),
                    function_id: "unknown".into(),
                    explanation: String::new(),
                };

                // Try to extract provenance from binary column.
                if let Some((idx, _)) = prov_col
                    && let Some(arr) = batch.column(idx).as_any().downcast_ref::<BinaryArray>()
                {
                    let nulls = arr.nulls();
                    if nulls.is_none_or(|n| n.is_valid(row)) {
                        let bytes = arr.value(row);
                        if let Ok(poly) = PolynomialSemiring::from_bytes(bytes) {
                            let trace = poly.explain();
                            debug!(row = global_row, trace = %trace, "provenance trace");

                            // Parse structured info from the token strings.
                            // Token format: "model_ver_id:func_id:[src1,src2]"
                            for token_str in poly.terms.keys() {
                                let parts: Vec<&str> = token_str.splitn(3, ':').collect();
                                if parts.len() >= 3 {
                                    exp.model_version = parts[0].to_string();
                                    exp.function_id = parts[1].to_string();
                                    let sources =
                                        parts[2].trim_start_matches('[').trim_end_matches(']');
                                    if !sources.is_empty() {
                                        exp.source_records.extend(
                                            sources.split(',').map(|s| s.trim().to_string()),
                                        );
                                    }
                                }
                            }

                            exp.explanation = format!(
                                "Derived via {} using model '{}', \
                                 sourced from [{}]",
                                exp.function_id,
                                exp.model_version,
                                exp.source_records.join(", ")
                            );
                        }
                    }
                }

                // If no provenance column, generate a basic explanation.
                if exp.explanation.is_empty() {
                    exp.explanation = format!("Row {global_row}: no provenance attached");
                }

                explanations.push(exp);
                global_row += 1;
            }
        }

        Ok(explanations)
    }

    /// Compute basic statistics for a float column.
    fn compute_stats(arr: &Float64Array) -> ColumnStats {
        let mut values: Vec<f64> = (0..arr.len())
            .filter(|&i| arr.nulls().is_none_or(|n| n.is_valid(i)))
            .map(|i| arr.value(i))
            .collect();

        if values.is_empty() {
            return ColumnStats {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
            };
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min = values[0];
        let max = values[values.len() - 1];
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let median = if values.len().is_multiple_of(2) {
            (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
        } else {
            values[values.len() / 2]
        };

        ColumnStats {
            min,
            max,
            mean,
            median,
        }
    }
}

struct ColumnStats {
    min: f64,
    max: f64,
    mean: f64,
    median: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Float64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn coarse_explanation() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "fraud_prob",
            DataType::Float64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![0.1, 0.5, 0.95]))],
        )
        .unwrap();

        let context = ExplainContext {
            rules: vec![("high_risk".into(), "fraud_prob > 0.90".into())],
            models: vec![("fraud_detector".into(), "1.0.0".into())],
            provenance_mode: "Polynomial".into(),
            device_summary: "8 CPUs + Metal M2".into(),
        };

        let explanation = Explainer::explain(ExplainLevel::Coarse, &[batch], &context).unwrap();
        assert!(explanation.summary.contains("3 row(s)"));
        assert!(explanation.summary.contains("high_risk"));
        assert!(explanation.row_explanations.is_empty());
    }
}
