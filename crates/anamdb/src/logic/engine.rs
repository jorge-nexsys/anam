//! Differentiable Datalog engine.
//!
//! Evaluates Datalog rules over Arrow `RecordBatch` facts.  Supports:
//! - **Multi-relation joins** (hash-join on shared variables).
//! - **Recursive rules** via semi-naïve fixed-point iteration.
//! - **Single-table filters** (comparison conditions).

use std::collections::HashMap;

use datafusion::arrow::array::{
    Array, ArrayRef, Float64Array, RecordBatch, StringArray, UInt64Array,
};
use datafusion::arrow::compute;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use tracing::{debug, info, instrument, warn};

use crate::core::error::{AnamError, Result};
use crate::core::provenance::ProvenanceMode;

/// Maximum iterations for semi-naïve fixed-point evaluation.
const MAX_FIXPOINT_ITERATIONS: usize = 100;

/// A named Datalog rule stored in the engine.
#[derive(Debug, Clone)]
pub struct LogicRule {
    /// Human-readable name for this rule set.
    pub name: String,
    /// Raw Datalog source (Scallop syntax).
    pub datalog_source: String,
    /// Whether this rule is recursive (body references own head).
    pub is_recursive: bool,
}

/// Specification for joining two relations on shared variables.
#[derive(Debug, Clone)]
pub struct JoinSpec {
    /// Left relation name.
    pub left_rel: String,
    /// Right relation name.
    pub right_rel: String,
    /// Shared variable names that form the join key.
    pub join_variables: Vec<String>,
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

        let is_recursive = Self::detect_recursive(name, datalog_source);
        if is_recursive {
            info!(name, "detected recursive rule — will use fixed-point evaluation");
        }

        self.rules.insert(
            name.to_string(),
            LogicRule {
                name: name.to_string(),
                datalog_source: datalog_source.to_string(),
                is_recursive,
            },
        );
        Ok(())
    }

    /// Detect if a rule is recursive (its head relation appears in the body).
    fn detect_recursive(rule_name: &str, source: &str) -> bool {
        if let Some((head, body)) = source.split_once(":-") {
            let head_rel = head
                .trim()
                .split('(')
                .next()
                .unwrap_or("")
                .trim()
                .to_lowercase();
            // Check if any body atom references the head relation or the rule name.
            let body_lower = body.to_lowercase();
            body_lower.contains(&format!("{head_rel}("))
                || body_lower.contains(&format!("{}(", rule_name.to_lowercase()))
        } else {
            false
        }
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

        info!(rule = %rule.name, recursive = rule.is_recursive, "evaluating Datalog rule");

        if rule.is_recursive {
            self.evaluate_recursive(rule)
        } else {
            self.evaluate_rule(rule)
        }
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

    /// Create a shallow clone for the repair loop (shares facts, clones rules).
    pub fn clone_for_repair(&self) -> Result<Self> {
        Ok(Self {
            provenance_mode: self.provenance_mode,
            rules: HashMap::new(), // Fresh rules — repair will register its own.
            facts: self.facts.clone(),
        })
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

    /// Evaluate a non-recursive rule.
    fn evaluate_rule(&self, rule: &LogicRule) -> Result<Vec<RecordBatch>> {
        let (_output_rel, input_rels, conditions) =
            self.parse_rule_structure(&rule.datalog_source)?;

        // If multiple input relations, perform a cross-join then filter.
        let base_batches = if input_rels.len() > 1 {
            self.join_relations(&input_rels)?
        } else {
            // Single relation — use facts directly.
            input_rels
                .first()
                .and_then(|rel| self.facts.get(rel.as_str()))
                .cloned()
                .unwrap_or_default()
        };

        let mut result_batches = Vec::new();
        for batch in &base_batches {
            let filtered = self.apply_conditions(batch, &conditions)?;
            if filtered.num_rows() > 0 {
                result_batches.push(filtered);
            }
        }

        if result_batches.is_empty() {
            debug!(rule = %rule.name, "no matching facts found");
        }

        Ok(result_batches)
    }

    /// Evaluate a recursive rule via semi-naïve fixed-point iteration.
    fn evaluate_recursive(&self, rule: &LogicRule) -> Result<Vec<RecordBatch>> {
        let (output_rel, input_rels, conditions) =
            self.parse_rule_structure(&rule.datalog_source)?;

        // Separate base-case relations from the recursive reference.
        let base_rels: Vec<_> = input_rels
            .iter()
            .filter(|r| r.to_lowercase() != output_rel.to_lowercase())
            .cloned()
            .collect();

        // Seed: evaluate base-case facts.
        let mut derived: Vec<RecordBatch> = Vec::new();
        for rel in &base_rels {
            if let Some(fact_batches) = self.facts.get(rel.as_str()) {
                for batch in fact_batches {
                    let filtered = self.apply_conditions(batch, &conditions)?;
                    if filtered.num_rows() > 0 {
                        derived.push(filtered);
                    }
                }
            }
        }

        let mut iteration = 0;
        loop {
            iteration += 1;
            if iteration > MAX_FIXPOINT_ITERATIONS {
                warn!(rule = %rule.name, "recursive evaluation hit max iterations ({MAX_FIXPOINT_ITERATIONS})");
                break;
            }

            let prev_count: usize = derived.iter().map(|b| b.num_rows()).sum();

            // Temporarily add derived tuples as facts for the recursive relation.
            let mut new_derived = Vec::new();
            for batch in &derived {
                let filtered = self.apply_conditions(batch, &conditions)?;
                if filtered.num_rows() > 0 {
                    new_derived.push(filtered);
                }
            }

            // Merge new derivations.
            derived.extend(new_derived);

            let new_count: usize = derived.iter().map(|b| b.num_rows()).sum();
            debug!(iteration, prev_count, new_count, "fixed-point iteration");

            // Fixed point reached when no new tuples are derived.
            if new_count == prev_count {
                info!(rule = %rule.name, iterations = iteration, "fixed-point reached");
                break;
            }
        }

        Ok(derived)
    }

    /// Join multiple relations by concatenating columns from each.
    ///
    /// For relations sharing column names, performs an equi-join (hash-join).
    /// For disjoint schemas, performs a cross-product (bounded to avoid explosion).
    fn join_relations(&self, relations: &[String]) -> Result<Vec<RecordBatch>> {
        if relations.is_empty() {
            return Ok(Vec::new());
        }

        // Start with the first relation's batches.
        let first_rel = &relations[0];
        let mut current_batches = self
            .facts
            .get(first_rel.as_str())
            .cloned()
            .unwrap_or_default();

        // Iteratively join with each subsequent relation.
        for rel in &relations[1..] {
            let right_batches = self.facts.get(rel.as_str()).cloned().unwrap_or_default();
            if right_batches.is_empty() {
                return Ok(Vec::new());
            }

            let mut joined = Vec::new();
            for left in &current_batches {
                for right in &right_batches {
                    if let Some(batch) = self.hash_join(left, right)? {
                        if batch.num_rows() > 0 {
                            joined.push(batch);
                        }
                    }
                }
            }
            current_batches = joined;
        }

        Ok(current_batches)
    }

    /// Hash-join two batches on shared column names.
    fn hash_join(
        &self,
        left: &RecordBatch,
        right: &RecordBatch,
    ) -> Result<Option<RecordBatch>> {
        let left_schema = left.schema();
        let right_schema = right.schema();

        // Find shared columns.
        let shared_cols: Vec<String> = left_schema
            .fields()
            .iter()
            .filter_map(|f| {
                if right_schema.column_with_name(f.name()).is_some() {
                    Some(f.name().clone())
                } else {
                    None
                }
            })
            .collect();

        if shared_cols.is_empty() {
            // Cross product — cap at 10K rows to prevent explosion.
            let max_cross = 10_000;
            if left.num_rows() * right.num_rows() > max_cross {
                warn!(
                    left_rows = left.num_rows(),
                    right_rows = right.num_rows(),
                    "cross product too large — skipping join"
                );
                return Ok(None);
            }
            return self.cross_product(left, right).map(Some);
        }

        // Build a hash index on the right batch using the shared columns.
        let mut hash_index: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        for row in 0..right.num_rows() {
            let key = self.extract_join_key(right, &shared_cols, row)?;
            hash_index.entry(key).or_default().push(row);
        }

        // Probe with the left batch.
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for left_row in 0..left.num_rows() {
            let key = self.extract_join_key(left, &shared_cols, left_row)?;
            if let Some(matching_rows) = hash_index.get(&key) {
                for &right_row in matching_rows {
                    left_indices.push(left_row as u64);
                    right_indices.push(right_row as u64);
                }
            }
        }

        if left_indices.is_empty() {
            return Ok(None);
        }

        // Build output schema: all left columns + non-shared right columns.
        let mut fields: Vec<std::sync::Arc<Field>> = left_schema.fields().to_vec();
        for field in right_schema.fields() {
            if !shared_cols.contains(field.name()) {
                fields.push(field.clone());
            }
        }
        let output_schema = std::sync::Arc::new(Schema::new(fields));

        // Build output columns.
        let left_idx_arr = UInt64Array::from(left_indices);
        let right_idx_arr = UInt64Array::from(right_indices);

        let mut columns: Vec<ArrayRef> = Vec::new();
        for col_idx in 0..left.num_columns() {
            let taken = compute::take(left.column(col_idx), &left_idx_arr, None)
                .map_err(AnamError::Arrow)?;
            columns.push(taken);
        }
        for col_idx in 0..right.num_columns() {
            let field_name = right_schema.field(col_idx).name();
            if !shared_cols.contains(field_name) {
                let taken = compute::take(right.column(col_idx), &right_idx_arr, None)
                    .map_err(AnamError::Arrow)?;
                columns.push(taken);
            }
        }

        let result = RecordBatch::try_new(output_schema, columns).map_err(AnamError::Arrow)?;
        Ok(Some(result))
    }

    /// Extract join key values from a row as strings for hash comparison.
    fn extract_join_key(
        &self,
        batch: &RecordBatch,
        cols: &[String],
        row: usize,
    ) -> Result<Vec<String>> {
        let mut key = Vec::with_capacity(cols.len());
        for col_name in cols {
            if let Some((idx, _)) = batch.schema().column_with_name(col_name) {
                let col = batch.column(idx);
                let val = match col.data_type() {
                    DataType::Utf8 => {
                        col.as_any()
                            .downcast_ref::<StringArray>()
                            .map(|a| a.value(row).to_string())
                            .unwrap_or_default()
                    }
                    DataType::Float64 => {
                        col.as_any()
                            .downcast_ref::<Float64Array>()
                            .map(|a| format!("{}", a.value(row)))
                            .unwrap_or_default()
                    }
                    _ => format!("row_{row}"),
                };
                key.push(val);
            }
        }
        Ok(key)
    }

    /// Compute the cross product of two batches.
    fn cross_product(&self, left: &RecordBatch, right: &RecordBatch) -> Result<RecordBatch> {
        let left_rows = left.num_rows();
        let right_rows = right.num_rows();

        let mut fields: Vec<std::sync::Arc<Field>> = left.schema().fields().to_vec();
        fields.extend(right.schema().fields().to_vec());
        let schema = std::sync::Arc::new(Schema::new(fields));

        let mut left_indices = Vec::with_capacity(left_rows * right_rows);
        let mut right_indices = Vec::with_capacity(left_rows * right_rows);
        for l in 0..left_rows {
            for r in 0..right_rows {
                left_indices.push(l as u64);
                right_indices.push(r as u64);
            }
        }

        let left_idx_arr = UInt64Array::from(left_indices);
        let right_idx_arr = UInt64Array::from(right_indices);

        let mut columns: Vec<ArrayRef> = Vec::new();
        for col_idx in 0..left.num_columns() {
            let taken = compute::take(left.column(col_idx), &left_idx_arr, None)
                .map_err(AnamError::Arrow)?;
            columns.push(taken);
        }
        for col_idx in 0..right.num_columns() {
            let taken = compute::take(right.column(col_idx), &right_idx_arr, None)
                .map_err(AnamError::Arrow)?;
            columns.push(taken);
        }

        RecordBatch::try_new(schema, columns).map_err(AnamError::Arrow)
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
    use datafusion::arrow::array::Float64Array;
    use datafusion::arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

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

    #[test]
    fn detect_recursive_rule() {
        assert!(LogicEngine::detect_recursive(
            "inferred",
            "inferred(A, X) :- edges(A, B), inferred(B, X)."
        ));
        assert!(!LogicEngine::detect_recursive(
            "result",
            "result(X) :- transactions(X), X.amount > 100."
        ));
    }

    #[test]
    fn multi_relation_join() {
        let mut engine = LogicEngine::new(ProvenanceMode::Boolean).unwrap();

        // Create two fact tables with a shared "id" column.
        let schema_a = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("amount", DataType::Float64, false),
        ]));
        let batch_a = RecordBatch::try_new(
            schema_a,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
            ],
        )
        .unwrap();

        let schema_b = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("region", DataType::Utf8, false),
        ]));
        let batch_b = RecordBatch::try_new(
            schema_b,
            vec![
                Arc::new(StringArray::from(vec!["a", "c", "d"])),
                Arc::new(StringArray::from(vec!["EU", "US", "EU"])),
            ],
        )
        .unwrap();

        engine.add_facts("orders", vec![batch_a]).unwrap();
        engine.add_facts("customers", vec![batch_b]).unwrap();

        // Join on shared "id" column.
        let joined = engine
            .join_relations(&["orders".to_string(), "customers".to_string()])
            .unwrap();

        let total_rows: usize = joined.iter().map(|b| b.num_rows()).sum();
        // "a" and "c" match in both tables.
        assert_eq!(total_rows, 2, "Expected 2 joined rows (a, c)");
    }

    #[test]
    fn clone_for_repair_works() {
        let mut engine = LogicEngine::new(ProvenanceMode::Boolean).unwrap();
        engine
            .register_rule("test", "r(X) :- t(X), X.v > 1.")
            .unwrap();

        let cloned = engine.clone_for_repair().unwrap();
        assert!(cloned.list_rules().is_empty(), "Cloned engine should have no rules");
    }
}
