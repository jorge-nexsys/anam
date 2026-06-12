//! Canonical Datalog normalization.
//!
//! Standardises Datalog rules into a canonical form so that semantically
//! equivalent rules always produce the same string.  This enables:
//! - Deduplication in the rule cache.
//! - Deterministic rule ordering for indexes and provenance traces.
//!
//! **Canonical form**
//! 1. Variables are UPPERCASED (`x` → `X`).
//! 2. Relation names are lowercased (`Transactions` → `transactions`).
//! 3. Body atoms are sorted alphabetically by relation name.
//! 4. Conditions are sorted alphabetically by column name.
//! 5. Trailing whitespace and redundant spaces are collapsed.

/// Normalize a Datalog rule (or plain constraint expression) into canonical form.
///
/// # Examples
/// ```
/// use anamdb::logic::canonical::normalize;
///
/// let raw = "High_Risk(x) :- Transactions(x), x.amount > 10000, x.fraud_prob > 0.9.";
/// let canonical = normalize(raw);
/// assert_eq!(canonical, "high_risk(X) :- transactions(X), X.amount > 10000, X.fraud_prob > 0.9.");
/// ```
pub fn normalize(source: &str) -> String {
    let source = source.trim();
    if source.is_empty() {
        return String::new();
    }

    // Determine if this is a rule (contains `:-`) or a plain constraint expression.
    if let Some((head, body)) = source.split_once(":-") {
        let head = normalize_head(head.trim());
        let body = normalize_body(body.trim());
        format!("{head} :- {body}")
    } else {
        // Plain constraint expression: normalize conditions only.
        normalize_conditions(source)
    }
}

/// Normalize the head atom: lowercase relation, uppercase variables.
fn normalize_head(head: &str) -> String {
    let head = head.trim().trim_end_matches('.');
    if let Some((rel, args)) = head.split_once('(') {
        let rel = rel.trim().to_lowercase();
        let args = args.trim_end_matches(')');
        let normalized_args: Vec<String> =
            args.split(',').map(|a| normalize_term(a.trim())).collect();
        format!("{rel}({})", normalized_args.join(", "))
    } else {
        head.to_lowercase()
    }
}

/// Normalize the body: separate atoms from conditions, sort each group,
/// then recombine.
fn normalize_body(body: &str) -> String {
    let body = body.trim().trim_end_matches('.');

    // Split on commas at the top level only (not inside parentheses).
    let parts = split_top_level_commas(body);

    let mut atoms: Vec<String> = Vec::new();
    let mut conditions: Vec<String> = Vec::new();

    for part in parts {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        if is_condition(trimmed) {
            conditions.push(normalize_single_condition(trimmed));
        } else {
            atoms.push(normalize_atom(trimmed));
        }
    }

    // Sort atoms alphabetically by relation name.
    atoms.sort();

    // Sort conditions alphabetically by column name (the LHS of the operator).
    conditions.sort_by(|a, b| {
        let a_col = extract_condition_column(a);
        let b_col = extract_condition_column(b);
        a_col.cmp(&b_col)
    });

    let mut result_parts: Vec<String> = Vec::new();
    result_parts.extend(atoms);
    result_parts.extend(conditions);

    format!("{}.", result_parts.join(", "))
}

/// Split a body string on commas, but only at the top level (depth 0).
/// Commas inside parenthesized argument lists are not split on.
fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in s.chars() {
        match ch {
            '(' => {
                depth += 1;
                current.push(ch);
            }
            ')' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                parts.push(current.clone());
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// Check if a body part is a filter condition (contains comparison operators).
fn is_condition(part: &str) -> bool {
    // Conditions contain comparison operators outside of parenthesised atoms.
    let operators = [">=", "<=", "!=", ">", "<", "="];
    // Exclude atoms like `relation(X)` — these contain `(` before any operator.
    if part.contains('(') && !part.contains('>') && !part.contains('<') && !part.contains("!=") {
        return false;
    }
    operators.iter().any(|op| part.contains(op))
}

/// Normalize a single atom: lowercase relation, uppercase variables.
fn normalize_atom(atom: &str) -> String {
    let atom = atom.trim().trim_end_matches('.');
    if let Some((rel, args)) = atom.split_once('(') {
        let rel = rel.trim().to_lowercase();
        let args = args.trim_end_matches(')');
        let normalized_args: Vec<String> =
            args.split(',').map(|a| normalize_term(a.trim())).collect();
        format!("{rel}({})", normalized_args.join(", "))
    } else {
        atom.to_lowercase()
    }
}

/// Normalize a term: uppercase bare variables, leave qualified refs and literals.
fn normalize_term(term: &str) -> String {
    let term = term.trim();

    // String literal — leave as-is.
    if term.starts_with('\'') || term.starts_with('"') {
        return term.to_string();
    }

    // Numeric literal — leave as-is.
    if term.parse::<f64>().is_ok() {
        return term.to_string();
    }

    // Qualified reference like `X.column_name` — uppercase the variable part.
    if let Some((var, col)) = term.split_once('.') {
        return format!("{}.{}", var.to_uppercase(), col.to_lowercase());
    }

    // Bare variable — uppercase.
    term.to_uppercase()
}

/// Normalize a single condition expression.
fn normalize_single_condition(cond: &str) -> String {
    let operators = [">=", "<=", "!=", ">", "<", "="];
    for op in &operators {
        if let Some((lhs, rhs)) = cond.split_once(op) {
            let lhs = normalize_term(lhs.trim());
            let rhs = normalize_term(rhs.trim());
            return format!("{lhs} {op} {rhs}");
        }
    }
    cond.to_string()
}

/// Normalize a plain constraint expression (no head/body — just conditions
/// joined by `AND`).
fn normalize_conditions(expr: &str) -> String {
    let parts: Vec<&str> = expr.split(" AND ").collect();
    let mut normalized: Vec<String> = parts
        .iter()
        .map(|p| normalize_single_condition(p.trim()))
        .collect();

    // Sort by column name.
    normalized.sort_by(|a, b| {
        let a_col = extract_condition_column(a);
        let b_col = extract_condition_column(b);
        a_col.cmp(&b_col)
    });

    normalized.join(" AND ")
}

/// Extract the column name (LHS) from a condition for sorting.
fn extract_condition_column(cond: &str) -> String {
    let operators = [">=", "<=", "!=", ">", "<", "="];
    for op in &operators {
        if let Some((lhs, _)) = cond.split_once(op) {
            let lhs = lhs.trim();
            // For qualified refs like `X.col`, extract `col`.
            if let Some((_, col)) = lhs.split_once('.') {
                return col.to_lowercase();
            }
            return lhs.to_lowercase();
        }
    }
    cond.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_basic_rule() {
        let raw = "High_Risk(x) :- Transactions(x), x.fraud_prob > 0.9, x.amount > 10000.";
        let result = normalize(raw);
        assert_eq!(
            result,
            "high_risk(X) :- transactions(X), X.amount > 10000, X.fraud_prob > 0.9."
        );
    }

    #[test]
    fn normalize_plain_constraint() {
        let raw = "fraud_prob > 0.90 AND amount > 10000 AND region = 'EU'";
        let result = normalize(raw);
        assert_eq!(
            result,
            "AMOUNT > 10000 AND FRAUD_PROB > 0.90 AND REGION = 'EU'"
        );
    }

    #[test]
    fn normalize_multi_atom_rule() {
        let raw = "result(X) :- orders(X), customers(X), X.amount > 100.";
        let result = normalize(raw);
        // Atoms sorted alphabetically: customers before orders.
        assert_eq!(
            result,
            "result(X) :- customers(X), orders(X), X.amount > 100."
        );
    }

    #[test]
    fn normalize_empty() {
        assert_eq!(normalize(""), "");
        assert_eq!(normalize("   "), "");
    }

    #[test]
    fn normalize_preserves_string_literals() {
        let raw = "eu_cust(X) :- customers(X), X.region = 'EU'.";
        let result = normalize(raw);
        assert!(result.contains("'EU'"));
    }

    #[test]
    fn normalize_recursive_rule() {
        let raw = "inferred_interaction(A, X) :- InteractsWith(A, B), inferred_interaction(B, X).";
        let result = normalize(raw);
        assert_eq!(
            result,
            "inferred_interaction(A, X) :- inferred_interaction(B, X), interactswith(A, B)."
        );
    }
}
