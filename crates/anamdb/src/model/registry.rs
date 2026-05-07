//! Model registry: CRUD operations over AI-Tables + FAO operator lookup.

use dashmap::DashMap;
use std::sync::Arc;
use tracing::info;

use crate::core::error::{AnamError, Result};
use crate::model::ai_tables::AiModelEntry;
use crate::model::fao::{FaoOperator, FaoRef};

/// Composite key: `(function_id, version)`.
type FaoKey = (String, String);

/// Central registry for AI models and their corresponding FAO operators.
///
/// Thread-safe via [`DashMap`].
pub struct ModelRegistry {
    /// AI-Tables catalog: `model_id → AiModelEntry`.
    catalog: DashMap<String, AiModelEntry>,
    /// FAO operator registry: `(function_id, version) → Arc<dyn FaoOperator>`.
    operators: DashMap<FaoKey, Arc<dyn FaoOperator>>,
}

impl std::fmt::Debug for ModelRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelRegistry")
            .field("catalog_size", &self.catalog.len())
            .field("operator_count", &self.operators.len())
            .finish()
    }
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            catalog: DashMap::new(),
            operators: DashMap::new(),
        }
    }

    // ── AI-Tables CRUD ─────────────────────────────────────────────────

    /// Register a new model in the AI-Tables catalog.
    pub fn register_model(&self, entry: AiModelEntry) -> Result<()> {
        info!(
            model_id = %entry.model_id,
            name = %entry.name,
            version = %entry.version,
            "registering model in AI-Tables"
        );
        self.catalog.insert(entry.model_id.clone(), entry);
        Ok(())
    }

    /// Remove a model from the catalog.
    pub fn unregister_model(&self, model_id: &str) -> Result<()> {
        if self.catalog.remove(model_id).is_some() {
            info!(model_id, "unregistered model");
            // Also remove any operators backed by this model.
            self.operators.retain(|_, op| op.model_id() != model_id);
            Ok(())
        } else {
            Err(AnamError::ModelNotFound(model_id.to_string()))
        }
    }

    /// Retrieve a model entry by ID.
    pub fn get_model(&self, model_id: &str) -> Result<AiModelEntry> {
        self.catalog
            .get(model_id)
            .map(|e| e.clone())
            .ok_or_else(|| AnamError::ModelNotFound(model_id.to_string()))
    }

    /// List all registered models.
    pub fn list_models(&self) -> Vec<AiModelEntry> {
        self.catalog.iter().map(|e| e.value().clone()).collect()
    }

    // ── FAO operator management ────────────────────────────────────────

    /// Register an FAO operator.
    pub fn register_operator(&self, operator: Arc<dyn FaoOperator>) -> Result<()> {
        let key = (
            operator.function_id().to_string(),
            operator.version().to_string(),
        );
        info!(
            function_id = %key.0,
            version = %key.1,
            model_id = %operator.model_id(),
            "registering FAO operator"
        );
        self.operators.insert(key, operator);
        Ok(())
    }

    /// Look up an FAO operator by function ID and version.
    pub fn get_operator(&self, function_id: &str, version: &str) -> Result<Arc<dyn FaoOperator>> {
        let key = (function_id.to_string(), version.to_string());
        self.operators
            .get(&key)
            .map(|op| Arc::clone(op.value()))
            .ok_or_else(|| {
                AnamError::ModelNotFound(format!("{function_id}@{version}"))
            })
    }

    /// Look up the latest version of an FAO operator by function ID.
    pub fn get_latest_operator(&self, function_id: &str) -> Result<Arc<dyn FaoOperator>> {
        let mut candidates: Vec<_> = self
            .operators
            .iter()
            .filter(|e| e.key().0 == function_id)
            .map(|e| (e.key().1.clone(), Arc::clone(e.value())))
            .collect();

        candidates.sort_by(|a, b| a.0.cmp(&b.0));

        candidates
            .pop()
            .map(|(_, op)| op)
            .ok_or_else(|| AnamError::ModelNotFound(format!("{function_id}@latest")))
    }

    /// Get all FAO references for a given function ID (all versions),
    /// used by the Pareto optimizer to enumerate candidate plans.
    pub fn get_operator_variants(&self, function_id: &str) -> Vec<FaoRef> {
        self.operators
            .iter()
            .filter(|e| e.key().0 == function_id)
            .map(|e| FaoRef::from_operator(e.value().as_ref()))
            .collect()
    }

    /// List all registered FAO references.
    pub fn list_operators(&self) -> Vec<FaoRef> {
        self.operators
            .iter()
            .map(|e| FaoRef::from_operator(e.value().as_ref()))
            .collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ai_tables::{AiModelEntry, ModelFormat};

    #[test]
    fn register_and_retrieve_model() {
        let registry = ModelRegistry::new();
        let entry = AiModelEntry::builder("test-model", "1.0.0")
            .format(ModelFormat::Onnx)
            .accuracy(0.95)
            .avg_latency_ms(10.0)
            .build();

        let model_id = entry.model_id.clone();
        registry.register_model(entry).unwrap();

        let retrieved = registry.get_model(&model_id).unwrap();
        assert_eq!(retrieved.name, "test-model");
        assert_eq!(retrieved.accuracy, 0.95);
    }

    #[test]
    fn unregister_missing_model_errors() {
        let registry = ModelRegistry::new();
        assert!(registry.unregister_model("nonexistent").is_err());
    }
}
