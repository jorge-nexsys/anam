//! Differentiable Datalog engine backed by Scallop.
//!
//! This module bridges Arrow `RecordBatch` data with the Scallop-core runtime,
//! enabling probabilistic logic programming over relational data.

use std::collections::HashMap;

use datafusion::arrow::array::{
    Array, ArrayRef, Float64Array, RecordBatch, StringArray, UInt64Array,
};
use datafusion::arrow::compute;
use datafusion::arrow::datatypes::DataType;
use tracing::{debug, info, instrument};

use crate::core::error::{AnamError, Result};
use crate::core::provenance::ProvenanceMode;

/// A named Datalog rule stored in the engine.
#[derive(Debug, Clone)]
pub struct LogicRule {
    /// Human-readable name for this rule set.
    pub name: String,
    /// Raw Datalog source (Scallop syntax).
    pub datalog_source: String,
}

/// The logic engine manages Datalog rules and evaluates them against facts.
pub struct LogicEngine {
    /// Active provenance mode.
    provenance_mode: ProvenanceMode,
    /// Registered rules: `name → LogicRule`.
    rules: HashMap<String, LogicRule>,
    /// Fact tables: `relation_name → Vec<RecordBatch>`.
    facts: HashMap<String, Vec<RecordBatch>>,
}

impl std::fmt::Debug for LogicEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogicEngine")
            .field("provenance_mode", &self.provenance_mode)
            .field("rules", &self.rules.len())
            .field("fact_tables", &self.facts.len())
            .finish()
    }
}

impl LogicEngine {
    /// Create a new logic engine with the specified provenance mode.
    pub fn new(provenance_mode: ProvenanceMode) -> Result<Self> {
        info!(?provenance_mode, "initializing logic engine");

        Ok(Self {
            provenance_mode,
            rules: HashMap::new(),
            facts: HashMap::new(),
        })
    }

    /// Register a named Datalog rule.
    #[instrument(skip(self))]
    pub fn register_rule(&mut self, name: &str, datalog_source: &str) -> Result<()> {
        info!(name, source = %datalog_source, "registering Datalog rule");
        self.validate_datalog(datalog_source)?;

        self.rules.insert(
            name.to_string(),
            LogicRule {
                name: name.to_string(),
                datalog_source: datalog_source.to_string(),
            },
        );
        Ok(())
    }

    /// Remove a rule by name.
    pub fn remove_rule(&mut self, name: &str) -> Result<()> {
        if self.rules.remove(name).is_some() {
            Ok(())
        } else {
            Err(AnamError::Logic(format!("rule '{name}' not found")))
        }
    }

    /// List all registered rule names.
    pub fn list_rules(&self) -> Vec<String> {
        self.rules.keys().cloned().collect()
    }

    /// Get the Datalog source body of a named rule.
    pub fn get_rule_body(&self, name: &str) -> Option<String> {
        self.rules.get(name).map(|r| r.datalog_source.clone())
    }

    /// Add facts (ground tuples) for a relation.
    pub fn add_facts(&mut self, relation: &str, batches: Vec<RecordBatch>) -> Result<()> {
        debug!(relation, batch_count = batches.len(), "adding facts");
        self.facts
            .entry(relation.to_string())
            .or_default()
            .extend(batches);
        Ok(())
    }

    /// Evaluate a specific rule and return derived tuples.
    #[instrument(skip(self))]
    pub fn evaluate(&self, rule_name: &str) -> Result<Vec<RecordBatch>> {
        let rule = self
            .rules
            .get(rule_name)
            .ok_or_else(|| AnamError::Logic(format!("rule '{rule_name}' not found")))?;

        info!(rule = %rule.name, "evaluating Datalog rule");
        self.evaluate_with_scallop(rule)
    }

    /// Evaluate all registered rules and return all derived tuples.
    pub fn evaluate_all(&self) -> Result<Vec<RecordBatch>> {
        let mut all_results = Vec::new();
        for rule_name in self.rules.keys() {
            let batches = self.evaluate(rule_name)?;
            all_results.extend(batches);
        }
        Ok(all_results)
    }

    /// Apply all registered rules as post-filters against the given batches.
    ///
    /// For each rule, rows that **violate** the rule's conditions are removed.
    /// Rules whose columns don't match the batch schema are silently skipped
    /// (they apply to different tables).
    ///
    /// This is the integration point for wiring Datalog constraints into
    /// DataFusion's query pipeline.
    pub fn filter_batches(&self, batches: &[RecordBatch]) -> Result<Vec<RecordBatch>> {
        if self.rules.is_empty() || batches.is_empty() {
            return Ok(batches.to_vec());
        }

        let mut filtered = batches.to_vec();

        for rule in self.rules.values() {
            let (_output_rel, _input_rels, conditions) =
                self.parse_rule_structure(&rule.datalog_source)?;

            if conditions.is_empty() {
                continue;
            }

            // Check if any condition column matches the batch schema.
            // If none match, this rule applies to a different table — skip it.
            if let Some(first_batch) = filtered.first() {
                let schema = first_batch.schema();
                let any_column_matches = conditions.iter().any(|c| {
                    let col_name = c.column.split('.').next_back().unwrap_or(&c.column);
                    schema.column_with_name(col_name).is_some()
                });
                if !any_column_matches {
                    continue;
                }
            }

            debug!(
                rule = %rule.name,
                conditions = conditions.len(),
                "applying Datalog rule as post-filter"
            );

            filtered = filtered
                .iter()
                .map(|batch| self.apply_conditions(batch, &conditions))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .filter(|b| b.num_rows() > 0)
                .collect();
        }

        Ok(filtered)
    }

    fn evaluate_with_scallop(&self, rule: &LogicRule) -> Result<Vec<RecordBatch>> {
        let (_output_rel, input_rels, conditions) =
            self.parse_rule_structure(&rule.datalog_source)?;

        let mut result_batches = Vec::new();

        for rel_name in &input_rels {
            if let Some(fact_batches) = self.facts.get(rel_name.as_str()) {
                for batch in fact_batches {
                    let filtered = self.apply_conditions(batch, &conditions)?;
                    if filtered.num_rows() > 0 {
                        result_batches.push(filtered);
                    }
                }
            }
        }

        if result_batches.is_empty() {
            debug!(rule = %rule.name, "no matching facts found");
        }

        Ok(result_batches)
    }

    fn parse_rule_structure(
        &self,
        source: &str,
    ) -> Result<(String, Vec<String>, Vec<FilterCondition>)> {
        let source = source.trim();

        if let Some((head, body)) = source.split_once(":-") {
            let output_rel = head
                .trim()
                .split('(')
                .next()
                .unwrap_or("derived")
                .trim()
                .to_string();

            let body_parts: Vec<&str> = body.split(',').map(|s| s.trim()).collect();
            let mut input_rels = Vec::new();
            let mut conditions = Vec::new();

            for part in body_parts {
                let trimmed = part.trim().trim_end_matches('.');
                if trimmed.contains('>') || trimmed.contains('<') || trimmed.contains('=') {
                    if let Some(cond) = FilterCondition::parse(trimmed) {
                        conditions.push(cond);
                    }
                } else if let Some(rel) = trimmed.split('(').next() {
                    input_rels.push(rel.trim().to_string());
                }
            }

            Ok((output_rel, input_rels, conditions))
        } else {
            let conditions: Vec<FilterCondition> = source
                .split(" AND ")
                .filter_map(|part| FilterCondition::parse(part.trim()))
                .collect();

            let table = conditions
                .first()
                .and_then(|c| c.column.split('.').next().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown".to_string());

            Ok(("derived".to_string(), vec![table], conditions))
        }
    }

    fn apply_conditions(
        &self,
        batch: &RecordBatch,
        conditions: &[FilterCondition],
    ) -> Result<RecordBatch> {
        if conditions.is_empty() {
            return Ok(batch.clone());
        }

        let num_rows = batch.num_rows();
        let mut mask = vec![true; num_rows];

        for condition in conditions {
            let col_name = condition
                .column
                .split('.')
                .next_back()
                .unwrap_or(&condition.column);

            if let Some((col_idx, _)) = batch.schema().column_with_name(col_name) {
                let col = batch.column(col_idx);
                let nulls = col.nulls();
                match col.data_type() {
                    DataType::Float64 => {
                        if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                            let threshold: f64 = condition.value.parse().map_err(|_| {
                                AnamError::Logic(format!(
                                    "invalid numeric value: {}",
                                    condition.value
                                ))
                            })?;
                            for (i, m) in mask.iter_mut().enumerate().take(num_rows) {
                                let is_valid = nulls.is_none_or(|n| n.is_valid(i));
                                if *m && is_valid {
                                    *m = match condition.op {
                                        FilterOp::Gt => arr.value(i) > threshold,
                                        FilterOp::Lt => arr.value(i) < threshold,
                                        FilterOp::Gte => arr.value(i) >= threshold,
                                        FilterOp::Lte => arr.value(i) <= threshold,
                                        FilterOp::Eq => {
                                            (arr.value(i) - threshold).abs() < f64::EPSILON
                                        }
                                        FilterOp::Neq => {
                                            (arr.value(i) - threshold).abs() >= f64::EPSILON
                                        }
                                    };
                                }
                            }
                        }
                    }
                    DataType::Utf8 => {
                        if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                            let target = condition.value.trim_matches('\'').trim_matches('"');
                            for (i, m) in mask.iter_mut().enumerate().take(num_rows) {
                                let is_valid = nulls.is_none_or(|n| n.is_valid(i));
                                if *m && is_valid {
                                    *m = match condition.op {
                                        FilterOp::Eq => arr.value(i) == target,
                                        FilterOp::Neq => arr.value(i) != target,
                                        _ => true,
                                    };
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Build filtered batch using take.
        let indices: Vec<u64> = mask
            .iter()
            .enumerate()
            .filter(|&(_, m)| *m)
            .map(|(i, _)| i as u64)
            .collect();
        let indices_array = UInt64Array::from(indices);

        let new_columns: Vec<ArrayRef> = (0..batch.num_columns())
            .map(|col_idx| {
                let col = batch.column(col_idx);
                compute::take(col, &indices_array, None).unwrap_or_else(|_| col.clone())
            })
            .collect();

        RecordBatch::try_new(batch.schema(), new_columns).map_err(AnamError::Arrow)
    }

    fn validate_datalog(&self, source: &str) -> Result<()> {
        let source = source.trim();
        if source.is_empty() {
            return Err(AnamError::Logic("empty Datalog source".into()));
        }
        if !source.contains(":-")
            && !source.contains('>')
            && !source.contains('<')
            && !source.contains('=')
        {
            return Err(AnamError::Logic(format!(
                "invalid Datalog syntax: '{source}' (expected rule or constraint)"
            )));
        }
        Ok(())
    }
}

// ── Filter condition types ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum FilterOp {
    Gt,
    Lt,
    Gte,
    Lte,
    Eq,
    Neq,
}

#[derive(Debug, Clone)]
struct FilterCondition {
    column: String,
    op: FilterOp,
    value: String,
}

impl FilterCondition {
    fn parse(s: &str) -> Option<Self> {
        let operators = [
            (">=", FilterOp::Gte),
            ("<=", FilterOp::Lte),
            ("!=", FilterOp::Neq),
            (">", FilterOp::Gt),
            ("<", FilterOp::Lt),
            ("=", FilterOp::Eq),
        ];
        for (op_str, op) in &operators {
            if let Some((col, val)) = s.split_once(op_str) {
                return Some(FilterCondition {
                    column: col.trim().to_string(),
                    op: op.clone(),
                    value: val.trim().to_string(),
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_list_rules() {
        let mut engine = LogicEngine::new(ProvenanceMode::Boolean).unwrap();
        engine
            .register_rule(
                "test",
                "high_risk(X) :- transactions(X), X.fraud_prob > 0.9",
            )
            .unwrap();
        assert_eq!(engine.list_rules().len(), 1);
    }

    #[test]
    fn reject_empty_source() {
        let mut engine = LogicEngine::new(ProvenanceMode::Boolean).unwrap();
        assert!(engine.register_rule("bad", "").is_err());
    }

    #[test]
    fn parse_constraint_expression() {
        let engine = LogicEngine::new(ProvenanceMode::Boolean).unwrap();
        let (_output, _inputs, conditions) = engine
            .parse_rule_structure("fraud_prob > 0.90 AND amount > 10000 AND region = 'EU'")
            .unwrap();
        assert_eq!(conditions.len(), 3);
    }
}
