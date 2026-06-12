//! Datalog Partial-Query Checker (PQC).
//!
//! Validates LLM-generated Datalog rules against the live database schema,
//! catching errors *before* registration and execution. Inspired by Xander's
//! partial-query checking approach (vocabulary, scope, and type checks).

use std::collections::{HashMap, HashSet};

use datafusion::arrow::datatypes::{DataType, Schema};
use tracing::{debug, info};

/// A single validation error found by the PQC.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Human-readable description of the problem.
    pub message: String,
    /// The offending token or fragment.
    pub offending_fragment: String,
    /// Category of the error.
    pub category: ValidationCategory,
    /// Suggested fix (if any).
    pub suggestion: Option<String>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.category, self.message)?;
        if let Some(ref suggestion) = self.suggestion {
            write!(f, " (suggestion: {suggestion})")?;
        }
        Ok(())
    }
}

/// Category of validation error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationCategory {
    /// Relation name not found in schema.
    Vocabulary,
    /// Variable referenced in condition but not bound in a body atom.
    Scope,
    /// Type mismatch between column and comparison value.
    TypeMismatch,
    /// Structural syntax error.
    Syntax,
}

impl std::fmt::Display for ValidationCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationCategory::Vocabulary => write!(f, "VOCABULARY"),
            ValidationCategory::Scope => write!(f, "SCOPE"),
            ValidationCategory::TypeMismatch => write!(f, "TYPE"),
            ValidationCategory::Syntax => write!(f, "SYNTAX"),
        }
    }
}

/// Result of a PQC validation run.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the rule passed all checks.
    pub is_valid: bool,
    /// Errors found (empty if valid).
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    /// Create a passing result.
    fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
        }
    }

    /// Format all errors into a single diagnostic string suitable for
    /// feeding back to an LLM for correction.
    pub fn diagnostic_string(&self) -> String {
        if self.is_valid {
            return "Rule is valid.".to_string();
        }
        let mut lines = vec!["Datalog validation errors:".to_string()];
        for (i, err) in self.errors.iter().enumerate() {
            lines.push(format!("  {}: {err}", i + 1));
        }
        lines.join("\n")
    }
}

/// The Datalog Partial-Query Checker.
///
/// Validates rules against a schema catalog mapping relation names to
/// their Arrow schemas.
#[derive(Debug)]
pub struct DatalogChecker {
    /// Known relations: `relation_name → Schema`.
    schemas: HashMap<String, Schema>,
}

impl DatalogChecker {
    /// Create a new checker with no schemas registered.
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
        }
    }

    /// Register a relation schema for validation.
    pub fn register_schema(&mut self, relation: &str, schema: Schema) {
        self.schemas.insert(relation.to_lowercase(), schema);
    }

    /// Register schemas from a list of `(name, schema)` pairs.
    pub fn register_schemas(&mut self, schemas: impl IntoIterator<Item = (String, Schema)>) {
        for (name, schema) in schemas {
            self.schemas.insert(name.to_lowercase(), schema);
        }
    }

    /// Validate a Datalog rule or constraint expression.
    pub fn validate(&self, source: &str) -> ValidationResult {
        let source = source.trim();
        if source.is_empty() {
            return ValidationResult {
                is_valid: false,
                errors: vec![ValidationError {
                    message: "Empty Datalog source.".into(),
                    offending_fragment: String::new(),
                    category: ValidationCategory::Syntax,
                    suggestion: Some("Provide a valid Datalog rule.".into()),
                }],
            };
        }

        info!(source, "PQC: validating Datalog rule");

        let mut errors = Vec::new();

        if let Some((_head, body)) = source.split_once(":-") {
            let body = body.trim().trim_end_matches('.');
            let parts: Vec<&str> = body.split(',').map(|s| s.trim()).collect();

            let mut bound_variables: HashSet<String> = HashSet::new();
            let mut referenced_relations: Vec<String> = Vec::new();

            // First pass: extract atoms and bound variables.
            for part in &parts {
                let trimmed = part.trim();
                if !self.is_condition(trimmed)
                    && let Some((rel, args)) = trimmed.split_once('(')
                {
                    let rel = rel.trim().to_lowercase();
                    referenced_relations.push(rel.clone());

                    // Extract bound variables from atom arguments.
                    let args = args.trim_end_matches(')');
                    for arg in args.split(',') {
                        let arg = arg.trim();
                        if !arg.starts_with('\'')
                            && !arg.starts_with('"')
                            && arg.parse::<f64>().is_err()
                        {
                            bound_variables.insert(arg.to_uppercase());
                        }
                    }
                }
            }

            // Vocabulary check: all relation names must be known.
            for rel in &referenced_relations {
                // Skip self-references (recursive rules reference their own head).
                let head_rel = source
                    .split(":-")
                    .next()
                    .and_then(|h| h.split('(').next())
                    .map(|h| h.trim().to_lowercase())
                    .unwrap_or_default();

                if rel != &head_rel && !self.schemas.contains_key(rel) {
                    let similar = self.find_similar_relation(rel);
                    errors.push(ValidationError {
                        message: format!("Relation '{rel}' not found in schema catalog."),
                        offending_fragment: rel.clone(),
                        category: ValidationCategory::Vocabulary,
                        suggestion: similar.map(|s| format!("Did you mean '{s}'?")),
                    });
                }
            }

            // Second pass: check conditions.
            for part in &parts {
                let trimmed = part.trim();
                if self.is_condition(trimmed) {
                    // Scope check: variable in condition must be bound.
                    if let Some((lhs, _rhs)) = self.split_condition(trimmed)
                        && let Some((var, col)) = lhs.split_once('.')
                    {
                        let var_upper = var.trim().to_uppercase();
                        if !bound_variables.contains(&var_upper) {
                            errors.push(ValidationError {
                                message: format!(
                                    "Variable '{var_upper}' used in condition but not bound in any body atom."
                                ),
                                offending_fragment: lhs.to_string(),
                                category: ValidationCategory::Scope,
                                suggestion: Some(format!(
                                    "Add a body atom that binds '{var_upper}', e.g. some_relation({var_upper})."
                                )),
                            });
                        }

                        // Type check: if we know the relation, check
                        // the column type against the comparison value.
                        let col_name = col.trim().to_lowercase();
                        self.type_check_condition(
                            &var_upper,
                            &col_name,
                            trimmed,
                            &referenced_relations,
                            &mut errors,
                        );
                    }
                }
            }
        } else {
            // Plain constraint expression (no `:-`).
            // Just validate conditions in `AND`-separated form.
            let parts: Vec<&str> = source.split(" AND ").collect();
            if parts.is_empty() {
                errors.push(ValidationError {
                    message: "Could not parse constraint expression.".into(),
                    offending_fragment: source.to_string(),
                    category: ValidationCategory::Syntax,
                    suggestion: None,
                });
            }
        }

        debug!(error_count = errors.len(), "PQC validation complete");

        if errors.is_empty() {
            ValidationResult::valid()
        } else {
            ValidationResult {
                is_valid: false,
                errors,
            }
        }
    }

    /// Check if a body part is a condition (vs. an atom).
    fn is_condition(&self, part: &str) -> bool {
        let operators = [">=", "<=", "!=", ">", "<"];
        // An atom has `relation(...)` structure.
        // A condition has a comparison operator.
        if part.contains('(') && part.contains(')') && !operators.iter().any(|op| part.contains(op))
        {
            return false;
        }
        // Check for `=` but not inside parentheses.
        if !part.contains('(') && part.contains('=') {
            return true;
        }
        operators.iter().any(|op| part.contains(op))
    }

    /// Split a condition into LHS and RHS around the operator.
    fn split_condition<'a>(&self, cond: &'a str) -> Option<(&'a str, &'a str)> {
        let operators = [">=", "<=", "!=", ">", "<", "="];
        for op in &operators {
            if let Some((lhs, rhs)) = cond.split_once(op) {
                return Some((lhs.trim(), rhs.trim()));
            }
        }
        None
    }

    /// Type-check a condition's RHS against the column's Arrow DataType.
    fn type_check_condition(
        &self,
        _variable: &str,
        col_name: &str,
        condition: &str,
        referenced_relations: &[String],
        errors: &mut Vec<ValidationError>,
    ) {
        // Find the column in any referenced relation's schema.
        let col_type = referenced_relations.iter().find_map(|rel| {
            self.schemas.get(rel).and_then(|schema| {
                schema
                    .column_with_name(col_name)
                    .map(|(_, field)| field.data_type().clone())
            })
        });

        if let Some(data_type) = col_type {
            // Extract the RHS value from the condition.
            if let Some((_lhs, rhs)) = self.split_condition(condition) {
                let rhs = rhs.trim().trim_end_matches('.');
                match &data_type {
                    DataType::Float64 | DataType::Float32 | DataType::Int64 | DataType::Int32 => {
                        // Quoted string compared against numeric column.
                        if rhs.starts_with('\'') || rhs.starts_with('"') {
                            errors.push(ValidationError {
                                message: format!(
                                    "Column '{col_name}' is numeric ({data_type}) but compared against string literal '{rhs}'."
                                ),
                                offending_fragment: condition.to_string(),
                                category: ValidationCategory::TypeMismatch,
                                suggestion: Some(format!(
                                    "Use a numeric literal, e.g. {col_name} > 0.9"
                                )),
                            });
                        } else if rhs.parse::<f64>().is_err()
                            && !rhs.contains('.')
                            && !rhs.chars().all(|c| c.is_uppercase())
                        {
                            // Non-numeric, non-variable, non-qualified ref.
                            errors.push(ValidationError {
                                message: format!(
                                    "Column '{col_name}' is numeric ({data_type}) but compared against non-numeric value '{rhs}'."
                                ),
                                offending_fragment: condition.to_string(),
                                category: ValidationCategory::TypeMismatch,
                                suggestion: Some(format!(
                                    "Use a numeric literal, e.g. {col_name} > 0.9"
                                )),
                            });
                        }
                    }
                    DataType::Utf8 | DataType::LargeUtf8 => {
                        if !rhs.starts_with('\'') && !rhs.starts_with('"') {
                            // Allow qualified variable references (X.col) or
                            // single-char uppercase variables (X, Y, Z).
                            let is_variable = rhs.contains('.')
                                || (rhs.len() == 1 && rhs.chars().all(|c| c.is_uppercase()));
                            if !is_variable {
                                errors.push(ValidationError {
                                    message: format!(
                                        "Column '{col_name}' is string ({data_type}) but compared against unquoted value '{rhs}'."
                                    ),
                                    offending_fragment: condition.to_string(),
                                    category: ValidationCategory::TypeMismatch,
                                    suggestion: Some(format!(
                                        "Quote the value: {col_name} = '{rhs}'"
                                    )),
                                });
                            }
                        }
                    }
                    _ => {} // Other types: skip for now.
                }
            }
        }
    }

    /// Find the most similar relation name in the catalog (Levenshtein ≤ 2).
    fn find_similar_relation(&self, name: &str) -> Option<String> {
        self.schemas
            .keys()
            .filter(|k| levenshtein(name, k) <= 2)
            .min_by_key(|k| levenshtein(name, k))
            .cloned()
    }
}

impl Default for DatalogChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple Levenshtein distance for relation name suggestions.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = i;
    }
    for (j, val) in dp[0].iter_mut().enumerate().take(n + 1) {
        *val = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::Field;

    fn test_checker() -> DatalogChecker {
        let mut checker = DatalogChecker::new();
        checker.register_schema(
            "transactions",
            Schema::new(vec![
                Field::new("amount", DataType::Float64, false),
                Field::new("fraud_prob", DataType::Float64, false),
                Field::new("region", DataType::Utf8, false),
                Field::new("merchant_type", DataType::Utf8, false),
            ]),
        );
        checker.register_schema(
            "customers",
            Schema::new(vec![
                Field::new("name", DataType::Utf8, false),
                Field::new("order_count", DataType::Int64, false),
                Field::new("region", DataType::Utf8, false),
            ]),
        );
        checker
    }

    #[test]
    fn valid_rule_passes() {
        let checker = test_checker();
        let result = checker
            .validate("high_risk(X) :- transactions(X), X.fraud_prob > 0.9, X.amount > 10000.");
        assert!(
            result.is_valid,
            "Expected valid, got: {}",
            result.diagnostic_string()
        );
    }

    #[test]
    fn unknown_relation_fails_vocabulary() {
        let checker = test_checker();
        let result = checker.validate("bad(X) :- nonexistent_table(X), X.foo > 1.");
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.category == ValidationCategory::Vocabulary)
        );
    }

    #[test]
    fn typo_relation_suggests_fix() {
        let checker = test_checker();
        let result = checker.validate("bad(X) :- transctions(X), X.amount > 1.");
        assert!(!result.is_valid);
        let vocab_err = result
            .errors
            .iter()
            .find(|e| e.category == ValidationCategory::Vocabulary)
            .unwrap();
        assert!(
            vocab_err
                .suggestion
                .as_ref()
                .unwrap()
                .contains("transactions")
        );
    }

    #[test]
    fn type_mismatch_string_vs_numeric() {
        let checker = test_checker();
        let result = checker.validate("bad(X) :- transactions(X), X.amount > 'high'.");
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.category == ValidationCategory::TypeMismatch)
        );
    }

    #[test]
    fn type_mismatch_unquoted_string() {
        let checker = test_checker();
        let result = checker.validate("bad(X) :- transactions(X), X.region = EU.");
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.category == ValidationCategory::TypeMismatch)
        );
    }

    #[test]
    fn empty_source_fails() {
        let checker = test_checker();
        let result = checker.validate("");
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.category == ValidationCategory::Syntax)
        );
    }

    #[test]
    fn plain_constraint_passes() {
        let checker = test_checker();
        let result = checker.validate("fraud_prob > 0.90 AND amount > 10000");
        // Plain constraints don't have atoms to validate against.
        assert!(result.is_valid);
    }

    #[test]
    fn recursive_rule_passes() {
        let checker = test_checker();
        // Recursive rules reference their own head — this should not be flagged
        // as a vocabulary error.
        let result = checker.validate("inferred(A, X) :- transactions(A), inferred(A, X).");
        assert!(
            result.is_valid,
            "Recursive self-reference should not fail vocabulary check: {}",
            result.diagnostic_string()
        );
    }
}
