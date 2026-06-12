//! Hamming-distance-1 Datalog repair loop.
//!
//! When a Datalog rule evaluates to zero results (indicating an overly
//! restrictive or incorrectly formulated constraint), this module generates
//! repair candidates at Hamming distance 1 — swapping a single operator,
//! column, or threshold — and returns the first variant that produces results.
//!
//! Inspired by Xander's Query Test-and-Repair (QTR) module.

use tracing::{debug, info, instrument};

use crate::core::error::Result;
use crate::logic::engine::LogicEngine;

/// A repair candidate: the modified rule and what was changed.
#[derive(Debug, Clone)]
pub struct RepairCandidate {
    /// The modified Datalog source.
    pub modified_source: String,
    /// Human-readable description of what was changed.
    pub change_description: String,
    /// The type of mutation applied.
    pub mutation: MutationType,
}

/// What kind of mutation was applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationType {
    /// Swapped a comparison operator (e.g. `>` → `>=`).
    OperatorSwap,
    /// Scaled a numeric threshold (e.g. `10000` → `1000`).
    ThresholdScale,
    /// Relaxed a conjunction to disjunction (conceptual — implemented as
    /// removing one condition).
    ConditionRelax,
}

/// Report from a successful repair.
#[derive(Debug, Clone)]
pub struct RepairReport {
    /// The original rule that produced zero results.
    pub original_source: String,
    /// The repaired rule that produced results.
    pub repaired_source: String,
    /// What was changed.
    pub change_description: String,
    /// How many results the repaired rule produced.
    pub result_count: usize,
    /// The mutation type.
    pub mutation: MutationType,
}

impl RepairReport {
    /// Format a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "═══ Datalog Repair Report ═══\n\
             Original: {}\n\
             Repaired: {}\n\
             Change:   {}\n\
             Results:  {} rows",
            self.original_source, self.repaired_source, self.change_description, self.result_count,
        )
    }
}

/// Generate Hamming-distance-1 repair candidates for a Datalog rule.
///
/// Returns a list of modified rules, each differing from the original by
/// exactly one operator swap, threshold scale, or condition relaxation.
pub fn generate_candidates(source: &str) -> Vec<RepairCandidate> {
    let source = source.trim();
    let mut candidates = Vec::new();

    if let Some((head, body)) = source.split_once(":-") {
        let body = body.trim().trim_end_matches('.');
        let parts: Vec<&str> = body.split(',').map(|s| s.trim()).collect();

        for (idx, part) in parts.iter().enumerate() {
            let trimmed = *part;

            // ── Operator swaps ────────────────────────────────────────
            let op_swaps: Vec<(&str, &str, &str)> = vec![
                (">", ">=", "relaxed > to >="),
                ("<", "<=", "relaxed < to <="),
                (">=", ">", "tightened >= to >"),
                ("<=", "<", "tightened <= to <"),
                ("=", "!=", "inverted = to !="),
                ("!=", "=", "inverted != to ="),
            ];

            for (from, to, desc) in &op_swaps {
                // Avoid double-matching: only match the exact operator.
                if contains_exact_op(trimmed, from)
                    && !(*from == ">" && trimmed.contains(">="))
                    && !(*from == "<" && trimmed.contains("<="))
                {
                    let new_part = replace_exact_op(trimmed, from, to);
                    let new_body = rebuild_body(&parts, idx, &new_part);
                    candidates.push(RepairCandidate {
                        modified_source: format!("{} :- {new_body}.", head.trim()),
                        change_description: format!("Condition '{}': {desc}", trimmed),
                        mutation: MutationType::OperatorSwap,
                    });
                }
            }

            // ── Threshold scaling ─────────────────────────────────────
            if let Some(threshold_candidates) =
                generate_threshold_variants(trimmed, head.trim(), &parts, idx)
            {
                candidates.extend(threshold_candidates);
            }
        }

        // ── Condition relaxation (remove one condition at a time) ──────
        let conditions: Vec<(usize, &&str)> = parts
            .iter()
            .enumerate()
            .filter(|(_, p)| is_condition(p))
            .collect();

        if conditions.len() > 1 {
            for (idx, cond) in &conditions {
                let remaining: Vec<&str> = parts
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i != idx)
                    .map(|(_, p)| *p)
                    .collect();
                let new_body = remaining.join(", ");
                candidates.push(RepairCandidate {
                    modified_source: format!("{} :- {new_body}.", head.trim()),
                    change_description: format!("Removed condition '{}'", cond),
                    mutation: MutationType::ConditionRelax,
                });
            }
        }
    }

    debug!(
        original = source,
        candidates = candidates.len(),
        "generated repair candidates"
    );

    candidates
}

/// Attempt to repair a rule that produced zero results.
///
/// Generates candidates, evaluates each against the engine's facts, and
/// returns the first candidate that produces results.
#[instrument(skip(engine))]
pub fn attempt_repair(
    engine: &LogicEngine,
    rule_name: &str,
    original_source: &str,
) -> Result<Option<RepairReport>> {
    info!(rule = rule_name, "attempting Hamming-distance-1 repair");

    let candidates = generate_candidates(original_source);

    if candidates.is_empty() {
        debug!("no repair candidates generated");
        return Ok(None);
    }

    for candidate in &candidates {
        // Try registering and evaluating the candidate.
        // We use a temporary engine clone to avoid polluting the real engine.
        let mut temp_engine = engine.clone_for_repair()?;
        if temp_engine
            .register_rule(rule_name, &candidate.modified_source)
            .is_err()
        {
            continue;
        }

        match temp_engine.evaluate(rule_name) {
            Ok(batches) => {
                let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
                if total_rows > 0 {
                    info!(
                        rule = rule_name,
                        rows = total_rows,
                        change = %candidate.change_description,
                        "repair successful"
                    );
                    return Ok(Some(RepairReport {
                        original_source: original_source.to_string(),
                        repaired_source: candidate.modified_source.clone(),
                        change_description: candidate.change_description.clone(),
                        result_count: total_rows,
                        mutation: candidate.mutation,
                    }));
                }
            }
            Err(_) => continue,
        }
    }

    debug!(rule = rule_name, "no repair candidate produced results");
    Ok(None)
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Check if a body part is a filter condition.
fn is_condition(part: &str) -> bool {
    let operators = [">=", "<=", "!=", ">", "<", "="];
    if part.contains('(') && !operators.iter().any(|op| part.contains(op)) {
        return false;
    }
    operators.iter().any(|op| part.contains(op))
}

/// Check if a part contains an exact operator (not a prefix of another).
fn contains_exact_op(part: &str, op: &str) -> bool {
    match op {
        ">" => part.contains('>') && !part.contains(">="),
        "<" => part.contains('<') && !part.contains("<="),
        "=" => {
            part.contains('=')
                && !part.contains(">=")
                && !part.contains("<=")
                && !part.contains("!=")
        }
        _ => part.contains(op),
    }
}

/// Replace the exact operator in a condition string.
fn replace_exact_op(part: &str, from: &str, to: &str) -> String {
    // For single-char operators, we need to be careful about multi-char ones.
    match from {
        ">" => {
            // Only replace standalone `>`.
            let mut result = String::new();
            let chars: Vec<char> = part.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                if chars[i] == '>' && (i + 1 >= chars.len() || chars[i + 1] != '=') {
                    result.push_str(to);
                    i += 1;
                } else {
                    result.push(chars[i]);
                    i += 1;
                }
            }
            result
        }
        "<" => {
            let mut result = String::new();
            let chars: Vec<char> = part.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                if chars[i] == '<' && (i + 1 >= chars.len() || chars[i + 1] != '=') {
                    result.push_str(to);
                    i += 1;
                } else {
                    result.push(chars[i]);
                    i += 1;
                }
            }
            result
        }
        _ => part.replacen(from, to, 1),
    }
}

/// Rebuild the body by replacing one part.
fn rebuild_body(parts: &[&str], replace_idx: usize, new_part: &str) -> String {
    parts
        .iter()
        .enumerate()
        .map(|(i, p)| if i == replace_idx { new_part } else { p })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Generate threshold scaling variants for numeric conditions.
fn generate_threshold_variants(
    condition: &str,
    head: &str,
    parts: &[&str],
    idx: usize,
) -> Option<Vec<RepairCandidate>> {
    let operators = [">=", "<=", "!=", ">", "<", "="];

    for op in &operators {
        if let Some((lhs, rhs)) = condition.split_once(op) {
            let rhs = rhs.trim().trim_end_matches('.');
            if let Ok(value) = rhs.parse::<f64>() {
                let mut candidates = Vec::new();

                // Scale down by 10×.
                let scaled_down = value / 10.0;
                let new_part = format!("{} {op} {scaled_down}", lhs.trim());
                let new_body = rebuild_body(parts, idx, &new_part);
                candidates.push(RepairCandidate {
                    modified_source: format!("{head} :- {new_body}."),
                    change_description: format!("Scaled threshold: {value} → {scaled_down} (÷10)"),
                    mutation: MutationType::ThresholdScale,
                });

                // Scale up by 10×.
                let scaled_up = value * 10.0;
                let new_part = format!("{} {op} {scaled_up}", lhs.trim());
                let new_body = rebuild_body(parts, idx, &new_part);
                candidates.push(RepairCandidate {
                    modified_source: format!("{head} :- {new_body}."),
                    change_description: format!("Scaled threshold: {value} → {scaled_up} (×10)"),
                    mutation: MutationType::ThresholdScale,
                });

                return Some(candidates);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_operator_swap_candidates() {
        let source = "high_risk(X) :- transactions(X), X.fraud_prob > 0.9, X.amount > 10000.";
        let candidates = generate_candidates(source);

        // Should have operator swaps and threshold scales.
        assert!(!candidates.is_empty());

        // Should include a `> → >=` swap.
        let has_gte_swap = candidates
            .iter()
            .any(|c| c.mutation == MutationType::OperatorSwap && c.modified_source.contains(">="));
        assert!(has_gte_swap, "Expected >= swap candidate");
    }

    #[test]
    fn generate_threshold_scale_candidates() {
        let source = "high_risk(X) :- transactions(X), X.amount > 10000.";
        let candidates = generate_candidates(source);

        let threshold_candidates: Vec<_> = candidates
            .iter()
            .filter(|c| c.mutation == MutationType::ThresholdScale)
            .collect();

        assert!(
            threshold_candidates.len() >= 2,
            "Expected at least 2 threshold variants (÷10 and ×10)"
        );

        let has_1000 = threshold_candidates
            .iter()
            .any(|c| c.modified_source.contains("1000"));
        assert!(has_1000, "Expected 10000 ÷ 10 = 1000 variant");
    }

    #[test]
    fn generate_condition_relaxation() {
        let source = "strict(X) :- transactions(X), X.fraud_prob > 0.99, X.amount > 100000, X.region = 'EU'.";
        let candidates = generate_candidates(source);

        let relax_candidates: Vec<_> = candidates
            .iter()
            .filter(|c| c.mutation == MutationType::ConditionRelax)
            .collect();

        // 3 conditions, so 3 relaxation candidates.
        assert_eq!(
            relax_candidates.len(),
            3,
            "Expected 3 condition relaxation candidates"
        );
    }

    #[test]
    fn no_candidates_for_atom_only_rule() {
        // A rule with no comparison conditions should still generate candidates
        // (but only relaxation, and only if there are multiple atoms).
        let source = "derived(X) :- facts(X).";
        let candidates = generate_candidates(source);
        // No conditions to swap, no relaxation possible (single atom).
        assert!(candidates.is_empty());
    }
}
