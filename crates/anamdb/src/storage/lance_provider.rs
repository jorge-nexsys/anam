//! Lance table provider: wraps Lance datasets as DataFusion `TableProvider`s.
//!
//! Opens datasets lazily on first `.load`, reads into DataFusion MemTable.
//! Large datasets will use Lance's streaming scan with batch-level schema
//! reconciliation in a future iteration.

use std::path::PathBuf;
use std::sync::Arc;

use dashmap::DashMap;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::TableProvider;
use datafusion::datasource::memory::MemTable;
use futures::TryStreamExt;
use lance::Dataset;
use tracing::{info, instrument};

use crate::core::error::{AnamError, Result};

/// Manages Lance datasets and exposes them as DataFusion table providers.
#[derive(Debug)]
pub struct LanceTableManager {
    /// Open datasets: `name → dataset_path`.
    datasets: DashMap<String, PathBuf>,
}

impl LanceTableManager {
    /// Create a new table manager.
    pub fn new() -> Self {
        Self {
            datasets: DashMap::new(),
        }
    }

    /// Open a Lance dataset and return a TableProvider.
    ///
    /// Data is scanned into memory on open. For large datasets, consider
    /// using `open_table_projected` with a column subset.
    #[instrument(skip(self))]
    pub async fn open_table(&self, path: &str) -> Result<Arc<dyn TableProvider>> {
        info!(path, "opening Lance dataset");

        let dataset = Dataset::open(path)
            .await
            .map_err(|e| AnamError::Lance(format!("failed to open dataset at '{path}': {e}")))?;

        let provider = self.scan_to_memtable(&dataset).await?;

        self.datasets.insert(path.to_string(), PathBuf::from(path));

        Ok(provider)
    }

    /// Open a Lance dataset at a specific version (snapshot).
    #[instrument(skip(self))]
    pub async fn open_table_as_of(
        &self,
        path: &str,
        version: u64,
    ) -> Result<Arc<dyn TableProvider>> {
        info!(path, version, "opening Lance dataset at version");

        let dataset = Dataset::open(path)
            .await
            .map_err(|e| AnamError::Lance(format!("failed to open dataset at '{path}': {e}")))?;
        dataset
            .checkout_version(version)
            .await
            .map_err(|e| AnamError::Lance(format!("failed to checkout version {version}: {e}")))?;

        let provider = self.scan_to_memtable(&dataset).await?;
        Ok(provider)
    }

    /// Scan a Lance dataset into a DataFusion MemTable.
    ///
    /// This reads all batches eagerly. Schema is derived from the first batch
    /// to ensure exact type consistency with Lance's output.
    async fn scan_to_memtable(&self, dataset: &Dataset) -> Result<Arc<dyn TableProvider>> {
        let mut scanner = dataset.scan();

        // Project all user columns explicitly to avoid internal lance columns.
        let lance_schema = dataset.schema();
        let field_names: Vec<String> = lance_schema.fields.iter().map(|f| f.name.clone()).collect();

        if !field_names.is_empty() {
            scanner
                .project(&field_names)
                .map_err(|e| AnamError::Lance(format!("projection failed: {e}")))?;
        }

        let batches: Vec<datafusion::arrow::array::RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| AnamError::Lance(format!("scan failed: {e}")))?
            .try_collect()
            .await
            .map_err(|e: lance::Error| AnamError::Lance(format!("collect failed: {e}")))?;

        if batches.is_empty() {
            // Empty dataset — derive schema from Lance.
            let arrow_schema: arrow_schema::Schema = arrow_schema::Schema::from(lance_schema);
            let schema: SchemaRef = Arc::new(arrow_schema);
            let provider =
                Arc::new(MemTable::try_new(schema, vec![vec![]]).map_err(AnamError::DataFusion)?);
            return Ok(provider);
        }

        // Use the actual batch schema for perfect type consistency.
        let schema = batches[0].schema();

        info!(
            rows = batches.iter().map(|b| b.num_rows()).sum::<usize>(),
            columns = schema.fields().len(),
            "loaded Lance dataset into memory"
        );

        let provider =
            Arc::new(MemTable::try_new(schema, vec![batches]).map_err(AnamError::DataFusion)?);

        Ok(provider)
    }

    /// List all opened datasets.
    pub fn list_datasets(&self) -> Vec<(String, PathBuf)> {
        self.datasets
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect()
    }
}

impl Default for LanceTableManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manager_default() {
        let mgr = LanceTableManager::new();
        assert!(mgr.list_datasets().is_empty());
    }
}
