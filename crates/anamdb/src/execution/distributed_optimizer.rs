//! Distributed Multi-Objective Optimizer.
//!
//! Extends the single-node Pareto optimizer with network-aware cost estimation.
//! The optimizer includes network routing costs and data-movement overhead in
//! its Pareto frontier calculations, and supports progressive refinement.

use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

/// A candidate plan with network-aware cost dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedPlan {
    /// Plan identifier (model or operator name).
    pub name: String,
    /// Estimated compute latency in ms (local execution).
    pub compute_latency_ms: f64,
    /// Estimated network latency in ms (data movement overhead).
    pub network_latency_ms: f64,
    /// Total estimated latency = compute + network.
    pub total_latency_ms: f64,
    /// Estimated result accuracy (0.0–1.0).
    pub accuracy: f64,
    /// Estimated cost (normalized units).
    pub cost: f64,
    /// Target node for this plan.
    pub target_node: String,
}

/// Constraints for distributed query planning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributedConstraints {
    /// Maximum total latency (compute + network) in ms.
    pub max_total_latency_ms: Option<f64>,
    /// Minimum acceptable accuracy.
    pub min_accuracy: Option<f64>,
    /// Maximum cost.
    pub max_cost: Option<f64>,
    /// Maximum network hop count.
    pub max_hops: Option<u32>,
}

/// The Distributed Multi-Objective Optimizer.
///
/// Computes network-aware Pareto frontiers that account for data-movement
/// overhead between nodes.
#[derive(Debug)]
pub struct DistributedOptimizer;

impl DistributedOptimizer {
    /// Compute the Pareto frontier for a set of distributed candidate plans.
    ///
    /// A plan is **dominated** if another plan is better in all 3 dimensions
    /// (total_latency, accuracy, cost). Non-dominated plans form the frontier.
    #[instrument(skip(plans))]
    pub fn compute_frontier(plans: &[DistributedPlan]) -> Vec<&DistributedPlan> {
        let mut frontier = Vec::new();

        for (i, plan) in plans.iter().enumerate() {
            let is_dominated = plans.iter().enumerate().any(|(j, other)| {
                i != j
                    && other.total_latency_ms <= plan.total_latency_ms
                    && other.accuracy >= plan.accuracy
                    && other.cost <= plan.cost
                    && (other.total_latency_ms < plan.total_latency_ms
                        || other.accuracy > plan.accuracy
                        || other.cost < plan.cost)
            });

            if !is_dominated {
                frontier.push(plan);
            }
        }

        info!(
            candidates = plans.len(),
            frontier = frontier.len(),
            "computed distributed Pareto frontier"
        );

        frontier
    }

    /// Filter the frontier by constraints.
    pub fn apply_constraints<'a>(
        frontier: &[&'a DistributedPlan],
        constraints: &DistributedConstraints,
    ) -> Vec<&'a DistributedPlan> {
        frontier
            .iter()
            .copied()
            .filter(|p| {
                constraints
                    .max_total_latency_ms
                    .is_none_or(|max| p.total_latency_ms <= max)
                    && constraints.min_accuracy.is_none_or(|min| p.accuracy >= min)
                    && constraints.max_cost.is_none_or(|max| p.cost <= max)
            })
            .collect()
    }

    /// Select the optimal plan from the frontier given constraints.
    ///
    /// Among feasible plans, selects the one with the highest accuracy.
    /// Ties broken by lowest total latency.
    pub fn select_optimal<'a>(
        plans: &[&'a DistributedPlan],
        constraints: &DistributedConstraints,
    ) -> Option<&'a DistributedPlan> {
        let feasible = Self::apply_constraints(plans, constraints);

        feasible.into_iter().max_by(|a, b| {
            a.accuracy
                .partial_cmp(&b.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.total_latency_ms
                        .partial_cmp(&a.total_latency_ms)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        })
    }

    /// Progressive refinement: re-evaluate a plan mid-execution.
    ///
    /// If an edge node's model fails to meet accuracy constraints, the optimizer
    /// re-routes remaining data to a more accurate core node model.
    #[instrument(skip(alternatives))]
    pub fn progressive_rewrite(
        failing_plan: &DistributedPlan,
        accuracy_threshold: f64,
        alternatives: &[DistributedPlan],
    ) -> Option<DistributedPlan> {
        if failing_plan.accuracy >= accuracy_threshold {
            debug!("plan meets accuracy threshold — no rewrite needed");
            return None;
        }

        info!(
            failing = %failing_plan.name,
            actual_accuracy = failing_plan.accuracy,
            threshold = accuracy_threshold,
            "progressive rewrite triggered"
        );

        // Find the highest-accuracy alternative that meets the threshold.
        alternatives
            .iter()
            .filter(|p| p.accuracy >= accuracy_threshold && p.name != failing_plan.name)
            .max_by(|a, b| {
                a.accuracy
                    .partial_cmp(&b.accuracy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    /// Get a formatted summary of the frontier.
    pub fn format_frontier(plans: &[&DistributedPlan]) -> String {
        let mut lines = vec!["═══ Distributed Pareto Frontier ═══".to_string()];
        for plan in plans {
            lines.push(format!(
                "  ★ {} @ {} — compute: {:.1}ms + network: {:.1}ms = {:.1}ms total, \
                 accuracy: {:.0}%, cost: {:.2}",
                plan.name,
                plan.target_node,
                plan.compute_latency_ms,
                plan.network_latency_ms,
                plan.total_latency_ms,
                plan.accuracy * 100.0,
                plan.cost
            ));
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_plans() -> Vec<DistributedPlan> {
        vec![
            DistributedPlan {
                name: "fraud_fast@edge".into(),
                compute_latency_ms: 0.5,
                network_latency_ms: 5.0,
                total_latency_ms: 5.5,
                accuracy: 0.75,
                cost: 0.1,
                target_node: "edge-0".into(),
            },
            DistributedPlan {
                name: "fraud_detector@core".into(),
                compute_latency_ms: 5.0,
                network_latency_ms: 1.0,
                total_latency_ms: 6.0,
                accuracy: 0.95,
                cost: 0.5,
                target_node: "core-0".into(),
            },
            DistributedPlan {
                name: "fraud_ensemble@hybrid".into(),
                compute_latency_ms: 10.0,
                network_latency_ms: 2.0,
                total_latency_ms: 12.0,
                accuracy: 0.99,
                cost: 1.0,
                target_node: "hybrid-0".into(),
            },
        ]
    }

    #[test]
    fn pareto_frontier() {
        let plans = test_plans();
        let frontier = DistributedOptimizer::compute_frontier(&plans);
        // All 3 are on the frontier (trade different dimensions).
        assert_eq!(frontier.len(), 3);
    }

    #[test]
    fn select_with_constraints() {
        let plans = test_plans();
        let frontier = DistributedOptimizer::compute_frontier(&plans);

        let constraints = DistributedConstraints {
            max_total_latency_ms: Some(10.0),
            min_accuracy: Some(0.90),
            ..Default::default()
        };

        let best = DistributedOptimizer::select_optimal(&frontier, &constraints);
        assert!(best.is_some());
        assert_eq!(best.unwrap().name, "fraud_detector@core");
    }

    #[test]
    fn progressive_rewrite_triggered() {
        let plans = test_plans();
        let failing = &plans[0]; // fraud_fast: accuracy 0.75

        let rewrite = DistributedOptimizer::progressive_rewrite(failing, 0.90, &plans);
        assert!(rewrite.is_some());
        assert_eq!(rewrite.unwrap().name, "fraud_ensemble@hybrid");
    }
}
