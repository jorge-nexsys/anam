//! Logic Pack: a self-contained, distributable bundle of neurosymbolic rules
//! and model references for a specific domain.
//!
//! # Example
//!
//! ```no_run
//! use anamdb::sdk::LogicPackBuilder;
//!
//! let pack = LogicPackBuilder::new("financial_compliance", "1.0.0")
//!     .description("EU AML/KYC compliance rules for transaction monitoring")
//!     .rule("high_risk", "fraud_prob > 0.90 AND amount > 10000")
//!     .rule("wire_alert", "merchant_type = 'wire_transfer' AND amount > 50000")
//!     .model_ref("fraud_detector", "demo/models/fraud_detector.onnx", 3, 5.0, 0.95)
//!     .model_ref("fraud_fast", "demo/models/fraud_detector_fast.onnx", 3, 0.5, 0.75)
//!     .build();
//! ```

use serde::{Deserialize, Serialize};

use crate::core::error::{AnamError, Result};

/// A bundled Datalog rule within a Logic Pack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackRule {
    /// Rule name (used as the Datalog head).
    pub name: String,
    /// The raw Datalog body (e.g., `fraud_prob > 0.90 AND amount > 10000`).
    pub datalog: String,
    /// Optional natural-language description of what this rule enforces.
    pub description: Option<String>,
}

/// A model reference within a Logic Pack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackModelRef {
    /// Model name (matches the AI-Tables registry key).
    pub name: String,
    /// Path to the ONNX artifact.
    pub artifact_path: String,
    /// Number of input features.
    pub num_features: usize,
    /// Expected average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Expected accuracy (0.0–1.0).
    pub accuracy: f64,
    /// Optional natural-language description for the optimizer.
    pub description: Option<String>,
}

/// A Logic Pack — a distributable bundle of domain-specific rules and models.
///
/// Logic Packs decouple domain expertise from the core engine, allowing
/// third-party developers to create reusable rulesets (e.g., Financial
/// Compliance, Healthcare Advising) without touching engine internals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicPack {
    /// Pack name (e.g., `financial_compliance`).
    pub name: String,
    /// Semantic version (e.g., `1.0.0`).
    pub version: String,
    /// Human-readable description of what this pack provides.
    pub description: Option<String>,
    /// Author or organization.
    pub author: Option<String>,
    /// Datalog rules bundled in this pack.
    pub rules: Vec<PackRule>,
    /// Model references bundled in this pack.
    pub models: Vec<PackModelRef>,
}

impl LogicPack {
    /// Load a Logic Pack from a JSON file.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| AnamError::Logic(format!("failed to parse Logic Pack JSON: {e}")))
    }

    /// Serialize this pack to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| AnamError::Logic(format!("failed to serialize Logic Pack: {e}")))
    }

    /// Load a Logic Pack from a JSON file on disk.
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            AnamError::Logic(format!("failed to read Logic Pack file '{path}': {e}"))
        })?;
        Self::from_json(&content)
    }

    /// Number of rules in this pack.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Number of model references in this pack.
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Get a formatted summary of this pack.
    pub fn summary(&self) -> String {
        let mut lines = vec![format!("Logic Pack: {} v{}", self.name, self.version)];
        if let Some(desc) = &self.description {
            lines.push(format!("  {desc}"));
        }
        if let Some(author) = &self.author {
            lines.push(format!("  Author: {author}"));
        }
        lines.push(format!(
            "  {} rule(s), {} model(s)",
            self.rules.len(),
            self.models.len()
        ));
        for rule in &self.rules {
            lines.push(format!("    • {} ← {}", rule.name, rule.datalog));
        }
        for model in &self.models {
            lines.push(format!(
                "    ◆ {} [{}] — {:.1}ms, {:.0}% accuracy",
                model.name,
                model.artifact_path,
                model.avg_latency_ms,
                model.accuracy * 100.0
            ));
        }
        lines.join("\n")
    }
}

/// Builder for constructing Logic Packs programmatically.
pub struct LogicPackBuilder {
    name: String,
    version: String,
    description: Option<String>,
    author: Option<String>,
    rules: Vec<PackRule>,
    models: Vec<PackModelRef>,
}

impl LogicPackBuilder {
    /// Start building a new Logic Pack.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            description: None,
            author: None,
            rules: Vec::new(),
            models: Vec::new(),
        }
    }

    /// Set the pack description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the author.
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Add a Datalog rule to the pack.
    pub fn rule(mut self, name: impl Into<String>, datalog: impl Into<String>) -> Self {
        self.rules.push(PackRule {
            name: name.into(),
            datalog: datalog.into(),
            description: None,
        });
        self
    }

    /// Add a Datalog rule with a description.
    pub fn rule_with_desc(
        mut self,
        name: impl Into<String>,
        datalog: impl Into<String>,
        desc: impl Into<String>,
    ) -> Self {
        self.rules.push(PackRule {
            name: name.into(),
            datalog: datalog.into(),
            description: Some(desc.into()),
        });
        self
    }

    /// Add a model reference to the pack.
    pub fn model_ref(
        mut self,
        name: impl Into<String>,
        path: impl Into<String>,
        num_features: usize,
        avg_latency_ms: f64,
        accuracy: f64,
    ) -> Self {
        self.models.push(PackModelRef {
            name: name.into(),
            artifact_path: path.into(),
            num_features,
            avg_latency_ms,
            accuracy,
            description: None,
        });
        self
    }

    /// Build the Logic Pack.
    pub fn build(self) -> LogicPack {
        LogicPack {
            name: self.name,
            version: self.version,
            description: self.description,
            author: self.author,
            rules: self.rules,
            models: self.models,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_logic_pack() {
        let pack = LogicPackBuilder::new("financial_compliance", "1.0.0")
            .description("AML/KYC transaction rules")
            .author("NSC")
            .rule("high_risk", "fraud_prob > 0.90 AND amount > 10000")
            .rule(
                "wire_alert",
                "merchant_type = 'wire_transfer' AND amount > 50000",
            )
            .model_ref(
                "fraud_detector",
                "demo/models/fraud_detector.onnx",
                3,
                5.0,
                0.95,
            )
            .build();

        assert_eq!(pack.name, "financial_compliance");
        assert_eq!(pack.rule_count(), 2);
        assert_eq!(pack.model_count(), 1);
    }

    #[test]
    fn serde_roundtrip() {
        let pack = LogicPackBuilder::new("test_pack", "0.1.0")
            .rule("r1", "x > 10")
            .build();

        let json = pack.to_json().unwrap();
        let restored = LogicPack::from_json(&json).unwrap();
        assert_eq!(restored.name, "test_pack");
        assert_eq!(restored.rules.len(), 1);
    }
}
