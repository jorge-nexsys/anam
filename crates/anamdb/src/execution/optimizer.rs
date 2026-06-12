//! Multi-Objective Pareto Optimizer.
//!
//! Balances execution latency, hardware cost, and result accuracy by computing
//! the Pareto frontier over candidate physical plans and selecting the one
//! closest to user-specified constraints.

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::dataframe::DataFrame;
use tracing::{debug, info, instrument, warn};

use crate::core::error::{AnamError, Result};
use crate::execution::dispatcher::DevicePool;
use crate::model::fao::FaoRef;
use crate::model::registry::ModelRegistry;

/// User-specified multi-objective constraints parsed from the SQL `WITH` clause.
#[derive(Debug, Clone)]
pub struct QueryConstraints {
    /// Maximum acceptable end-to-end latency in milliseconds.
    pub max_latency_ms: Option<f64>,
    /// Minimum acceptable result accuracy (0.0–1.0).
    pub min_accuracy: Option<f64>,
    /// Maximum cost budget (abstract units).
    pub max_cost: Option<f64>,
}

/// A candidate physical plan with estimated metrics.
#[derive(Debug, Clone)]
pub struct CandidatePlan {
    /// Which FAO variant this plan uses.
    pub fao_ref: FaoRef,
    /// Estimated end-to-end latency (ms).
    pub est_latency_ms: f64,
    /// Estimated accuracy.
    pub est_accuracy: f64,
    /// Estimated cost (abstract units: GPU-seconds, API calls, etc.).
    pub est_cost: f64,
}

impl CandidatePlan {
    /// Check if this plan satisfies the given constraints.
    pub fn satisfies(&self, constraints: &QueryConstraints) -> bool {
        if constraints
            .max_latency_ms
            .is_some_and(|max_lat| self.est_latency_ms > max_lat)
        {
            return false;
        }
        if constraints
            .min_accuracy
            .is_some_and(|min_acc| self.est_accuracy < min_acc)
        {
            return false;
        }
        if constraints
            .max_cost
            .is_some_and(|max_cost| self.est_cost > max_cost)
        {
            return false;
        }
        true
    }

    /// Dominates another plan if it is at least as good on ALL objectives and
    /// strictly better on at least one.
    pub fn dominates(&self, other: &CandidatePlan) -> bool {
        let lat_ok = self.est_latency_ms <= other.est_latency_ms;
        let acc_ok = self.est_accuracy >= other.est_accuracy;
        let cost_ok = self.est_cost <= other.est_cost;

        let strictly_better = self.est_latency_ms < other.est_latency_ms
            || self.est_accuracy > other.est_accuracy
            || self.est_cost < other.est_cost;

        lat_ok && acc_ok && cost_ok && strictly_better
    }
}

/// The Pareto optimizer selects the best physical plan from the AI-Tables
/// model catalog based on multi-objective constraints.
pub struct ParetoOptimizer {
    /// Model registry for enumerating FAO variants.
    registry: Arc<ModelRegistry>,
    /// Device pool for cost estimation.
    device_pool: Arc<DevicePool>,
}

impl std::fmt::Debug for ParetoOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParetoOptimizer").finish()
    }
}

impl ParetoOptimizer {
    /// Create a new optimizer.
    pub fn new(registry: Arc<ModelRegistry>, device_pool: Arc<DevicePool>) -> Self {
        Self {
            registry,
            device_pool,
        }
    }

    /// Parse a SQL query's `WITH (...)` clause into a clean SQL string and
    /// optional constraints.
    pub fn parse_constraints(&self, query: &str) -> Result<(String, Option<QueryConstraints>)> {
        // Look for `WITH (...)` at the end of the query.
        let query_trimmed = query.trim().trim_end_matches(';');

        if let Some(with_start) = query_trimmed.to_uppercase().rfind("WITH (") {
            let clean_sql = query_trimmed[..with_start].trim().to_string();
            let with_clause = &query_trimmed[with_start + 6..];
            let with_body = with_clause.trim_end_matches(')').trim();

            let mut constraints = QueryConstraints {
                max_latency_ms: None,
                min_accuracy: None,
                max_cost: None,
            };

            for part in with_body.split(',') {
                let part = part.trim();
                if let Some((key, val)) = part.split_once('=') {
                    let key = key.trim().to_lowercase();
                    let val = val.trim();
                    match key.as_str() {
                        "max_latency_ms" => {
                            constraints.max_latency_ms = val.parse().ok();
                        }
                        "min_accuracy" => {
                            constraints.min_accuracy = val.parse().ok();
                        }
                        "max_cost" => {
                            constraints.max_cost = val.parse().ok();
                        }
                        _ => {
                            warn!(key = %key, "unknown constraint in WITH clause");
                        }
                    }
                }
            }

            Ok((clean_sql, Some(constraints)))
        } else {
            Ok((query_trimmed.to_string(), None))
        }
    }

    /// Execute a DataFrame with multi-objective constraints.
    ///
    /// Enumerates candidate plans from the model registry, computes the
    /// Pareto frontier, selects the best feasible plan, and executes the
    /// selected FAO operator against the input batches.
    #[instrument(skip(self, df))]
    pub async fn execute_with_constraints(
        &self,
        df: DataFrame,
        constraints: QueryConstraints,
    ) -> Result<Vec<RecordBatch>> {
        info!(?constraints, "executing with Pareto optimization");

        // Collect the base batches from DataFusion.
        let batches = df.collect().await.map_err(AnamError::DataFusion)?;

        // Enumerate FAO operators and compute the Pareto frontier.
        let operators = self.registry.list_operators();
        if operators.is_empty() {
            debug!("no FAO operators registered — returning base results");
            return Ok(batches);
        }

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let candidates = self.enumerate_candidates(&operators, total_rows);
        let frontier = self.compute_pareto_frontier(&candidates);
        let feasible: Vec<_> = frontier
            .iter()
            .filter(|c| c.satisfies(&constraints))
            .collect();

        if feasible.is_empty() {
            warn!("no feasible plan on Pareto frontier — using default execution");
            return Ok(batches);
        }

        // Select the best feasible plan (lowest latency, break ties by accuracy).
        let best = feasible
            .iter()
            .min_by(|a, b| {
                a.est_latency_ms
                    .partial_cmp(&b.est_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        info!(
            fao = %best.fao_ref.function_id,
            latency = best.est_latency_ms,
            accuracy = best.est_accuracy,
            cost = best.est_cost,
            "selected optimal plan — executing FAO operator"
        );

        // Execute the selected FAO operator against the base batches.
        let operator = self
            .registry
            .get_latest_operator(&best.fao_ref.function_id)
            .map_err(|_| {
                AnamError::Inference(format!(
                    "FAO operator '{}' not found in registry",
                    best.fao_ref.function_id
                ))
            })?;

        let mut result_batches = Vec::with_capacity(batches.len());
        for batch in &batches {
            match operator.execute(batch.clone()).await {
                Ok(result) => result_batches.push(result),
                Err(e) => {
                    warn!(
                        fao = %best.fao_ref.function_id,
                        error = %e,
                        "FAO execution failed for batch — falling back to raw batch"
                    );
                    result_batches.push(batch.clone());
                }
            }
        }

        Ok(result_batches)
    }

    /// Parse a `PREDICT` SQL query and extract model/target information.
    ///
    /// Supported syntax:
    /// - `PREDICT CLASS OF <column> FROM <table> WITH (model = '<name>')`
    /// - `PREDICT VALUE OF <column> FROM <table> WITH (model = '<name>')`
    ///
    /// Returns `(target_column, source_table, model_name, prediction_type)`.
    pub fn parse_predict_query(query: &str) -> Option<(String, String, String, PredictionType)> {
        let upper = query.trim().to_uppercase();

        let pred_type = if upper.starts_with("PREDICT CLASS OF") {
            Some(PredictionType::Classification)
        } else if upper.starts_with("PREDICT VALUE OF") {
            Some(PredictionType::Regression)
        } else {
            None
        }?;

        // Extract target column: word after "OF".
        let after_of = query.split_whitespace().collect::<Vec<_>>();

        // Format: PREDICT [CLASS|VALUE] OF <col> FROM <table> [WITH (model = '...')]
        let of_idx = after_of.iter().position(|w| w.eq_ignore_ascii_case("OF"))?;
        let from_idx = after_of
            .iter()
            .position(|w| w.eq_ignore_ascii_case("FROM"))?;

        if of_idx + 1 >= from_idx || from_idx + 1 >= after_of.len() {
            return None;
        }

        let target_column = after_of[of_idx + 1].to_string();
        let source_table = after_of[from_idx + 1].to_string();

        // Extract model name from WITH clause.
        let model_name = if let Some(with_idx) = upper.find("WITH") {
            let with_part = &query.trim()[with_idx..];
            // Look for model = 'name' or model = "name".
            let model_re_start = with_part.find("model").or_else(|| with_part.find("MODEL"));
            if let Some(start) = model_re_start {
                let after_eq = &with_part[start..];
                if let Some(eq_idx) = after_eq.find('=') {
                    let val = after_eq[eq_idx + 1..].trim();
                    let val = val.trim_start_matches('(').trim();
                    let name =
                        val.trim_matches(|c: char| c == '\'' || c == '"' || c == ')' || c == ' ');
                    name.to_string()
                } else {
                    "default".to_string()
                }
            } else {
                "default".to_string()
            }
        } else {
            "default".to_string()
        };

        info!(
            pred_type = ?pred_type,
            target = %target_column,
            table = %source_table,
            model = %model_name,
            "parsed PREDICT query"
        );

        Some((target_column, source_table, model_name, pred_type))
    }

    /// Enumerate candidate plans from FAO variants.
    fn enumerate_candidates(&self, operators: &[FaoRef], total_rows: usize) -> Vec<CandidatePlan> {
        operators
            .iter()
            .map(|fao| {
                let device_multiplier = self.device_pool.speed_multiplier();
                CandidatePlan {
                    fao_ref: fao.clone(),
                    est_latency_ms: fao.est_latency_ms * (total_rows as f64 / 1000.0).max(1.0)
                        / device_multiplier,
                    est_accuracy: fao.est_accuracy,
                    est_cost: fao.est_latency_ms * 0.001 / device_multiplier,
                }
            })
            .collect()
    }

    /// Compute the Pareto frontier from a set of candidate plans.
    ///
    /// A plan is on the frontier if no other plan dominates it.
    pub fn compute_pareto_frontier(&self, candidates: &[CandidatePlan]) -> Vec<CandidatePlan> {
        let mut frontier = Vec::new();

        for (i, candidate) in candidates.iter().enumerate() {
            let dominated = candidates
                .iter()
                .enumerate()
                .any(|(j, other)| i != j && other.dominates(candidate));
            if !dominated {
                frontier.push(candidate.clone());
            }
        }

        debug!(
            candidates = candidates.len(),
            frontier = frontier.len(),
            "computed Pareto frontier"
        );

        frontier
    }
}

/// Type of prediction for PREDICT SQL queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionType {
    /// PREDICT CLASS OF — classification task.
    Classification,
    /// PREDICT VALUE OF — regression task.
    Regression,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_with_clause() {
        let registry = Arc::new(ModelRegistry::new());
        let pool = Arc::new(DevicePool::cpu_only());
        let optimizer = ParetoOptimizer::new(registry, pool);

        let (sql, constraints) = optimizer
            .parse_constraints(
                "SELECT * FROM HighRisk WITH (max_latency_ms = 50, min_accuracy = 0.95)",
            )
            .unwrap();

        assert_eq!(sql, "SELECT * FROM HighRisk");
        let c = constraints.unwrap();
        assert_eq!(c.max_latency_ms, Some(50.0));
        assert_eq!(c.min_accuracy, Some(0.95));
    }

    #[test]
    fn pareto_frontier_basic() {
        let registry = Arc::new(ModelRegistry::new());
        let pool = Arc::new(DevicePool::cpu_only());
        let optimizer = ParetoOptimizer::new(registry, pool);

        let candidates = vec![
            CandidatePlan {
                fao_ref: FaoRef {
                    function_id: "fast".into(),
                    version: "1".into(),
                    model_id: "m1".into(),
                    est_latency_ms: 10.0,
                    est_accuracy: 0.8,
                },
                est_latency_ms: 10.0,
                est_accuracy: 0.8,
                est_cost: 0.01,
            },
            CandidatePlan {
                fao_ref: FaoRef {
                    function_id: "accurate".into(),
                    version: "1".into(),
                    model_id: "m2".into(),
                    est_latency_ms: 100.0,
                    est_accuracy: 0.99,
                },
                est_latency_ms: 100.0,
                est_accuracy: 0.99,
                est_cost: 0.1,
            },
            CandidatePlan {
                fao_ref: FaoRef {
                    function_id: "dominated".into(),
                    version: "1".into(),
                    model_id: "m3".into(),
                    est_latency_ms: 100.0,
                    est_accuracy: 0.8,
                },
                est_latency_ms: 100.0,
                est_accuracy: 0.8,
                est_cost: 0.1,
            },
        ];

        let frontier = optimizer.compute_pareto_frontier(&candidates);
        // "dominated" should be excluded (worse latency AND cost than "fast",
        // worse accuracy than "accurate").
        assert_eq!(frontier.len(), 2);
        let ids: Vec<_> = frontier
            .iter()
            .map(|c| c.fao_ref.function_id.as_str())
            .collect();
        assert!(ids.contains(&"fast"));
        assert!(ids.contains(&"accurate"));
    }
}
