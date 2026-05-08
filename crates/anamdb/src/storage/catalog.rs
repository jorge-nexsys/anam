//! Persistent catalog: SQLite-backed metadata storage for tables, rules,
//! model metadata, and session configuration.
//!
//! This ensures that registered tables, Datalog rules, and model entries
//! survive across process restarts.

use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info, instrument};

use crate::core::error::{AnamError, Result};

/// A single catalog entry representing a registered table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableEntry {
    /// Logical table name used in SQL queries.
    pub name: String,
    /// Filesystem path to the Lance dataset.
    pub lance_path: String,
}

/// A single catalog entry representing a Datalog rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleEntry {
    /// Rule name.
    pub name: String,
    /// Datalog source expression.
    pub datalog: String,
}

/// A single catalog entry representing a registered model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Model name.
    pub name: String,
    /// Model version.
    pub version: String,
    /// Path to the model artifact.
    pub artifact_path: String,
    /// FAO function ID.
    pub function_id: String,
    /// Number of input features.
    pub num_features: usize,
    /// Average latency in ms.
    pub avg_latency_ms: f64,
    /// Accuracy score.
    pub accuracy: f64,
}

/// Persistent catalog backed by a JSON file.
///
/// Uses a simple JSON file for storage to avoid adding a SQLite dependency.
/// The catalog is loaded into memory on open, mutated in-place, and flushed
/// to disk on every write operation for crash safety.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Catalog {
    /// Registered tables.
    pub tables: Vec<TableEntry>,
    /// Registered Datalog rules.
    pub rules: Vec<RuleEntry>,
    /// Registered models.
    pub models: Vec<ModelEntry>,
}

/// Handle to a persistent catalog on disk.
#[derive(Debug)]
pub struct CatalogStore {
    /// Path to the catalog file.
    path: String,
    /// In-memory catalog state.
    catalog: Catalog,
}

impl CatalogStore {
    /// Open or create a catalog at the given path.
    #[instrument]
    pub fn open(path: &str) -> Result<Self> {
        let catalog = if Path::new(path).exists() {
            info!(path, "loading existing catalog");
            let data = std::fs::read_to_string(path).map_err(AnamError::Io)?;
            serde_json::from_str(&data)
                .map_err(|e| AnamError::Serde(format!("failed to parse catalog: {e}")))?
        } else {
            info!(path, "creating new catalog");
            let catalog = Catalog::default();
            // Ensure parent directory exists.
            if let Some(parent) = Path::new(path).parent() {
                std::fs::create_dir_all(parent).map_err(AnamError::Io)?;
            }
            catalog
        };

        let store = Self {
            path: path.to_string(),
            catalog,
        };
        store.flush()?;
        Ok(store)
    }

    /// Flush the catalog to disk.
    fn flush(&self) -> Result<()> {
        let data = serde_json::to_string_pretty(&self.catalog)
            .map_err(|e| AnamError::Serde(format!("failed to serialize catalog: {e}")))?;
        std::fs::write(&self.path, data).map_err(AnamError::Io)?;
        debug!(path = %self.path, "catalog flushed to disk");
        Ok(())
    }

    // ── Tables ────────────────────────────────────────────────────────

    /// Register a table in the catalog.
    pub fn register_table(&mut self, name: &str, lance_path: &str) -> Result<()> {
        // Upsert: replace if exists.
        self.catalog.tables.retain(|t| t.name != name);
        self.catalog.tables.push(TableEntry {
            name: name.to_string(),
            lance_path: lance_path.to_string(),
        });
        self.flush()
    }

    /// List all registered tables.
    pub fn list_tables(&self) -> &[TableEntry] {
        &self.catalog.tables
    }

    /// Remove a table from the catalog.
    pub fn remove_table(&mut self, name: &str) -> Result<()> {
        self.catalog.tables.retain(|t| t.name != name);
        self.flush()
    }

    // ── Rules ─────────────────────────────────────────────────────────

    /// Register a Datalog rule in the catalog.
    pub fn register_rule(&mut self, name: &str, datalog: &str) -> Result<()> {
        self.catalog.rules.retain(|r| r.name != name);
        self.catalog.rules.push(RuleEntry {
            name: name.to_string(),
            datalog: datalog.to_string(),
        });
        self.flush()
    }

    /// List all registered rules.
    pub fn list_rules(&self) -> &[RuleEntry] {
        &self.catalog.rules
    }

    /// Remove a rule from the catalog.
    pub fn remove_rule(&mut self, name: &str) -> Result<()> {
        self.catalog.rules.retain(|r| r.name != name);
        self.flush()
    }

    // ── Models ────────────────────────────────────────────────────────

    /// Register a model in the catalog.
    pub fn register_model(&mut self, entry: ModelEntry) -> Result<()> {
        self.catalog.models.retain(|m| m.name != entry.name);
        self.catalog.models.push(entry);
        self.flush()
    }

    /// List all registered models.
    pub fn list_models(&self) -> &[ModelEntry] {
        &self.catalog.models
    }

    /// Remove a model from the catalog.
    pub fn remove_model(&mut self, name: &str) -> Result<()> {
        self.catalog.models.retain(|m| m.name != name);
        self.flush()
    }

    /// Get the in-memory catalog snapshot.
    pub fn snapshot(&self) -> &Catalog {
        &self.catalog
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.json");
        let path_str = path.to_str().unwrap();

        // Create catalog and add entries.
        let mut store = CatalogStore::open(path_str).unwrap();
        store.register_table("txns", "/data/txns.lance").unwrap();
        store
            .register_rule("high_risk", "high_risk(X) :- txns(X), fraud_prob > 0.80")
            .unwrap();
        store
            .register_model(ModelEntry {
                name: "fraud_detector".to_string(),
                version: "1.0.0".to_string(),
                artifact_path: "models/fraud.onnx".to_string(),
                function_id: "fraud_detector".to_string(),
                num_features: 3,
                avg_latency_ms: 5.0,
                accuracy: 0.95,
            })
            .unwrap();

        // Re-open and verify persistence.
        let store2 = CatalogStore::open(path_str).unwrap();
        assert_eq!(store2.list_tables().len(), 1);
        assert_eq!(store2.list_tables()[0].name, "txns");
        assert_eq!(store2.list_rules().len(), 1);
        assert_eq!(store2.list_rules()[0].name, "high_risk");
        assert_eq!(store2.list_models().len(), 1);
        assert_eq!(store2.list_models()[0].name, "fraud_detector");
    }

    #[test]
    fn catalog_upsert() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.json");
        let path_str = path.to_str().unwrap();

        let mut store = CatalogStore::open(path_str).unwrap();
        store.register_table("txns", "/old/path.lance").unwrap();
        store.register_table("txns", "/new/path.lance").unwrap();
        assert_eq!(store.list_tables().len(), 1);
        assert_eq!(store.list_tables()[0].lance_path, "/new/path.lance");
    }

    #[test]
    fn catalog_remove() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.json");
        let path_str = path.to_str().unwrap();

        let mut store = CatalogStore::open(path_str).unwrap();
        store.register_table("a", "/a.lance").unwrap();
        store.register_table("b", "/b.lance").unwrap();
        assert_eq!(store.list_tables().len(), 2);

        store.remove_table("a").unwrap();
        assert_eq!(store.list_tables().len(), 1);
        assert_eq!(store.list_tables()[0].name, "b");
    }
}
