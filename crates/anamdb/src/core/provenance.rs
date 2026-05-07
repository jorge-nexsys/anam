//! Semiring provenance framework.
//!
//! Every tuple in AnamDB carries a provenance tag from a commutative semiring.
//! This module defines the [`Semiring`] trait plus the three concrete semirings
//! from the SPECS:
//!
//! | Semiring | Purpose |
//! |---|---|
//! | `BoolSemiring` | Standard SQL (exists / not-exists). |
//! | `ProbSemiring` | Neural confidence propagation. |
//! | `PolynomialSemiring` | Fine-grained lineage tracking (model+function+source). |

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::core::error::{AnamError, Result};

// ── Trait ──────────────────────────────────────────────────────────────────

/// A commutative semiring `(K, ⊕, ⊗, 0, 1)`.
pub trait Semiring: Clone + fmt::Debug + Send + Sync + 'static {
    /// Additive identity.
    fn zero() -> Self;
    /// Multiplicative identity.
    fn one() -> Self;
    /// Semiring addition (⊕) — used when *alternative derivations* merge.
    fn add(&self, other: &Self) -> Self;
    /// Semiring multiplication (⊗) — used when derivations are *composed*.
    fn mul(&self, other: &Self) -> Self;

    /// Serialize to bytes for Arrow `BinaryArray` storage.
    fn to_bytes(&self) -> Result<Vec<u8>>;
    /// Deserialize from bytes.
    fn from_bytes(bytes: &[u8]) -> Result<Self>;
}

// ── Boolean ────────────────────────────────────────────────────────────────

/// Standard SQL provenance: a tuple either exists or it does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoolSemiring(pub bool);

impl Semiring for BoolSemiring {
    fn zero() -> Self {
        BoolSemiring(false)
    }
    fn one() -> Self {
        BoolSemiring(true)
    }
    fn add(&self, other: &Self) -> Self {
        BoolSemiring(self.0 || other.0)
    }
    fn mul(&self, other: &Self) -> Self {
        BoolSemiring(self.0 && other.0)
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(vec![self.0 as u8])
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bytes
            .first()
            .map(|&b| BoolSemiring(b != 0))
            .ok_or_else(|| AnamError::Serde("empty bytes for BoolSemiring".into()))
    }
}

// ── Probability ────────────────────────────────────────────────────────────

/// Neural confidence propagation: probabilities in `[0, 1]`.
///
/// ⊕ = probabilistic-or (`1 - (1-a)(1-b)`)
/// ⊗ = product
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProbSemiring(pub f64);

impl Semiring for ProbSemiring {
    fn zero() -> Self {
        ProbSemiring(0.0)
    }
    fn one() -> Self {
        ProbSemiring(1.0)
    }

    fn add(&self, other: &Self) -> Self {
        // Independent-OR: P(A ∨ B) = 1 − (1−P(A))(1−P(B))
        ProbSemiring(1.0 - (1.0 - self.0) * (1.0 - other.0))
    }

    fn mul(&self, other: &Self) -> Self {
        ProbSemiring(self.0 * other.0)
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(self.0.to_le_bytes().to_vec())
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let arr: [u8; 8] = bytes
            .try_into()
            .map_err(|_| AnamError::Serde("expected 8 bytes for ProbSemiring".into()))?;
        Ok(ProbSemiring(f64::from_le_bytes(arr)))
    }
}

// ── Polynomial (Lineage) ───────────────────────────────────────────────────

/// A provenance token identifying the exact origin of a derived tuple.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProvenanceToken {
    /// Model version that produced this derivation step.
    pub model_ver_id: String,
    /// The specific FAO function invoked.
    pub func_id: String,
    /// The source record(s) consumed.
    pub source_record_ids: Vec<String>,
}

impl fmt::Display for ProvenanceToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:[{}]",
            self.model_ver_id,
            self.func_id,
            self.source_record_ids.join(",")
        )
    }
}

/// Fine-grained lineage: `ℕ[X]` polynomials over [`ProvenanceToken`]s.
///
/// Each monomial tracks *how many times* a particular derivation path was
/// exercised. The coefficients are natural numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolynomialSemiring {
    /// `token → coefficient`
    pub terms: HashMap<String, usize>,
}

impl PolynomialSemiring {
    /// Create a singleton monomial with coefficient 1.
    pub fn singleton(token: ProvenanceToken) -> Self {
        let mut terms = HashMap::new();
        terms.insert(token.to_string(), 1);
        Self { terms }
    }

    /// Human-readable lineage dump.
    pub fn explain(&self) -> String {
        self.terms
            .iter()
            .map(|(tok, coeff)| {
                if *coeff == 1 {
                    tok.clone()
                } else {
                    format!("{coeff}·{tok}")
                }
            })
            .collect::<Vec<_>>()
            .join(" ⊕ ")
    }
}

impl Semiring for PolynomialSemiring {
    fn zero() -> Self {
        Self {
            terms: HashMap::new(),
        }
    }

    fn one() -> Self {
        let mut terms = HashMap::new();
        terms.insert("1".to_string(), 1);
        Self { terms }
    }

    fn add(&self, other: &Self) -> Self {
        let mut merged = self.terms.clone();
        for (tok, coeff) in &other.terms {
            *merged.entry(tok.clone()).or_insert(0) += coeff;
        }
        Self { terms: merged }
    }

    fn mul(&self, other: &Self) -> Self {
        let mut product = HashMap::new();
        for (tok_a, c_a) in &self.terms {
            for (tok_b, c_b) in &other.terms {
                let combined_key = if tok_a == "1" {
                    tok_b.clone()
                } else if tok_b == "1" {
                    tok_a.clone()
                } else {
                    format!("{tok_a}⊗{tok_b}")
                };
                *product.entry(combined_key).or_insert(0) += c_a * c_b;
            }
        }
        Self { terms: product }
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| AnamError::Serde(e.to_string()))
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|e| AnamError::Serde(e.to_string()))
    }
}

// ── Utilities ──────────────────────────────────────────────────────────────

/// Which provenance semiring a session is configured to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ProvenanceMode {
    /// Standard boolean (SQL) semantics.
    Boolean,
    /// Probabilistic confidence propagation.
    Probability,
    /// Full polynomial lineage.
    #[default]
    Polynomial,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_semiring_laws() {
        let zero = BoolSemiring::zero();
        let one = BoolSemiring::one();
        assert_eq!(zero.add(&zero), zero);
        assert_eq!(zero.add(&one), one);
        assert_eq!(one.mul(&one), one);
        assert_eq!(one.mul(&zero), zero);
    }

    #[test]
    fn prob_semiring_independent_or() {
        let a = ProbSemiring(0.5);
        let b = ProbSemiring(0.5);
        let result = a.add(&b);
        assert!((result.0 - 0.75).abs() < 1e-10);
    }

    #[test]
    fn polynomial_merge() {
        let tok = ProvenanceToken {
            model_ver_id: "resnet50_v2".into(),
            func_id: "detect_objects".into(),
            source_record_ids: vec!["img_001".into()],
        };
        let a = PolynomialSemiring::singleton(tok.clone());
        let b = PolynomialSemiring::singleton(tok);
        let merged = a.add(&b);
        let key = merged.terms.keys().next().unwrap();
        assert_eq!(merged.terms[key], 2);
    }

    #[test]
    fn prob_round_trip() {
        let original = ProbSemiring(0.42);
        let bytes = original.to_bytes().unwrap();
        let decoded = ProbSemiring::from_bytes(&bytes).unwrap();
        assert_eq!(original, decoded);
    }
}
