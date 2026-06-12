//! `LogicOptimizerRule` — a DataFusion logical optimizer rule that extracts
//! pushable predicates from Datalog rules and injects them as Filter nodes.
//!
//! When a session has Datalog rules containing simple column comparisons
//! (e.g., `X.amount > 10000`), this rule can extract those predicates and
//! inject them into the logical plan as native DataFusion filter nodes,
//! allowing the query engine to push them toward scan operations.
//!
//! ## How It Works
//!
//! The rule is registered as a DataFusion `OptimizerRule`. During logical plan
//! optimization, it checks if there are any registered Datalog rules with
//! pushable predicates and adds corresponding `Filter` nodes to the plan.

use std::sync::Arc;

use datafusion_common::tree_node::Transformed;
use datafusion_common::Result as DfResult;
use datafusion_expr::{col, lit, Expr, LogicalPlan};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use parking_lot::RwLock;
use tracing::{debug, info};

use crate::logic::engine::LogicEngine;

/// A logical optimizer rule that injects Datalog-derived filter predicates
/// into the DataFusion logical plan.
#[derive(Debug)]
pub struct LogicOptimizerRule {
    /// Reference to the logic engine for inspecting rule sources.
    logic_engine: Arc<RwLock<LogicEngine>>,
}

impl LogicOptimizerRule {
    /// Create a new instance of the optimizer rule.
    pub fn new(logic_engine: Arc<RwLock<LogicEngine>>) -> Self {
        Self { logic_engine }
    }

    /// Extract simple column comparison predicates from all registered Datalog rules.
    fn extract_all_predicates(&self) -> Vec<Expr> {
        let engine = self.logic_engine.read();
        let rule_names = engine.list_rules();
        let mut all_predicates = Vec::new();

        for name in &rule_names {
            if let Some(source) = engine.get_rule_body(name) {
                let predicates = extract_pushable_predicates(&source);
                if !predicates.is_empty() {
                    debug!(
                        rule = %name,
                        predicates = predicates.len(),
                        "extracted pushable predicates from Datalog rule"
                    );
                    all_predicates.extend(predicates);
                }
            }
        }

        all_predicates
    }
}

impl OptimizerRule for LogicOptimizerRule {
    fn name(&self) -> &str {
        "LogicFilterPushdown"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> DfResult<Transformed<LogicalPlan>> {
        let predicates = self.extract_all_predicates();

        if predicates.is_empty() {
            return Ok(Transformed::no(plan));
        }

        // Only inject filters on TableScan or similar leaf nodes.
        match &plan {
            LogicalPlan::TableScan(_) => {
                let schema = plan.schema().clone();

                // Verify all columns in the predicate exist in the scan schema.
                let valid_predicates: Vec<Expr> = predicates
                    .into_iter()
                    .filter(|p| {
                        // Simple check: see if the predicate references columns in the schema.
                        let cols = p.column_refs();
                        cols.iter().all(|c| schema.has_column(c))
                    })
                    .collect();

                if valid_predicates.is_empty() {
                    return Ok(Transformed::no(plan));
                }

                let combined = combine_predicates(&valid_predicates);
                info!(
                    predicates = valid_predicates.len(),
                    "injecting Datalog predicates as Filter on TableScan"
                );

                let filtered = LogicalPlan::Filter(
                    datafusion_expr::logical_plan::Filter::try_new(
                        combined,
                        Arc::new(plan),
                    )?,
                );

                Ok(Transformed::yes(filtered))
            }
            _ => Ok(Transformed::no(plan)),
        }
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        // Apply bottom-up so we inject filters at the leaf (scan) level first.
        Some(ApplyOrder::BottomUp)
    }
}

/// Extract simple column comparison predicates from a Datalog rule source.
///
/// Supported patterns (parsed from the Datalog body):
/// - `X.column > value`
/// - `X.column < value`
/// - `X.column >= value`
/// - `X.column <= value`
/// - `X.column = value`
/// - `X.column != value`
fn extract_pushable_predicates(source: &str) -> Vec<Expr> {
    let mut predicates = Vec::new();

    // Parse the body of the Datalog rule (after ":-").
    let body = match source.split_once(":-") {
        Some((_, body)) => body.trim().trim_end_matches('.'),
        None => return predicates,
    };

    // Split on commas to get individual conditions.
    for part in body.split(',') {
        let trimmed = part.trim();

        // Try to parse "X.column op value" patterns.
        // Check multi-char operators first to avoid partial matches.
        for (op_str, make_expr) in &[
            (">=", make_gte as fn(&str, &str) -> Option<Expr>),
            ("<=", make_lte as fn(&str, &str) -> Option<Expr>),
            ("!=", make_neq as fn(&str, &str) -> Option<Expr>),
            (">", make_gt as fn(&str, &str) -> Option<Expr>),
            ("<", make_lt as fn(&str, &str) -> Option<Expr>),
            ("=", make_eq as fn(&str, &str) -> Option<Expr>),
        ] {
            if let Some((lhs, rhs)) = trimmed.split_once(op_str) {
                let col_name = extract_column_name(lhs.trim());
                if let Some(col_name) = col_name {
                    if let Some(expr) = make_expr(&col_name, rhs.trim()) {
                        predicates.push(expr);
                        break;
                    }
                }
            }
        }
    }

    predicates
}

/// Extract the column name from a Datalog variable reference like `X.column_name`.
fn extract_column_name(s: &str) -> Option<String> {
    if let Some((_var, col)) = s.split_once('.') {
        let col = col.trim();
        if !col.is_empty() && col.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Some(col.to_string());
        }
    }
    None
}

/// Parse a value string into a DataFusion literal expression.
fn parse_value_lit(s: &str) -> Option<Expr> {
    let s = s.trim();

    // Try float first.
    if s.contains('.') {
        if let Ok(f) = s.parse::<f64>() {
            return Some(lit(f));
        }
    }

    // Try integer.
    if let Ok(i) = s.parse::<i64>() {
        return Some(lit(i));
    }

    // Try string literal (single-quoted).
    if s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2 {
        let inner = &s[1..s.len() - 1];
        return Some(lit(inner));
    }

    None
}

fn make_gt(col_name: &str, val: &str) -> Option<Expr> {
    Some(col(col_name).gt(parse_value_lit(val)?))
}

fn make_gte(col_name: &str, val: &str) -> Option<Expr> {
    Some(col(col_name).gt_eq(parse_value_lit(val)?))
}

fn make_lt(col_name: &str, val: &str) -> Option<Expr> {
    Some(col(col_name).lt(parse_value_lit(val)?))
}

fn make_lte(col_name: &str, val: &str) -> Option<Expr> {
    Some(col(col_name).lt_eq(parse_value_lit(val)?))
}

fn make_eq(col_name: &str, val: &str) -> Option<Expr> {
    Some(col(col_name).eq(parse_value_lit(val)?))
}

fn make_neq(col_name: &str, val: &str) -> Option<Expr> {
    Some(col(col_name).not_eq(parse_value_lit(val)?))
}

/// Combine multiple predicates into a single AND expression.
fn combine_predicates(predicates: &[Expr]) -> Expr {
    predicates
        .iter()
        .cloned()
        .reduce(|a, b| a.and(b))
        .unwrap_or_else(|| lit(true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_column_name_parses_var_dot_col() {
        assert_eq!(extract_column_name("X.amount"), Some("amount".into()));
        assert_eq!(extract_column_name("T.region_code"), Some("region_code".into()));
        assert_eq!(extract_column_name("plain_name"), None);
        assert_eq!(extract_column_name(""), None);
    }

    #[test]
    fn parse_value_lit_handles_types() {
        // Float
        assert!(parse_value_lit("3.14").is_some());
        // Integer
        assert!(parse_value_lit("42").is_some());
        // String literal
        assert!(parse_value_lit("'hello'").is_some());
        // Invalid
        assert!(parse_value_lit("no_quotes").is_none());
    }

    #[test]
    fn extract_predicates_from_datalog_rule() {
        let source = "high_risk(X) :- transactions(X), X.amount > 10000, X.region = 'EU'.";
        let predicates = extract_pushable_predicates(source);
        assert_eq!(predicates.len(), 2, "should extract 2 predicates");
    }

    #[test]
    fn extract_predicates_empty_for_no_conditions() {
        let source = "derived(X) :- base(X).";
        let predicates = extract_pushable_predicates(source);
        assert!(predicates.is_empty(), "no comparison predicates expected");
    }

    #[test]
    fn combine_predicates_works() {
        let preds = vec![col("a").gt(lit(1)), col("b").lt(lit(10))];
        let combined = combine_predicates(&preds);
        let s = format!("{combined}");
        assert!(s.contains("a"), "should reference column a");
        assert!(s.contains("b"), "should reference column b");
    }
}
