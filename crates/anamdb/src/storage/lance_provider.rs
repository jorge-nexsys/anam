//! Lance table provider: wraps Lance datasets as DataFusion `TableProvider`s.
//!
//! Supports `AS OF 'timestamp'` via Lance's built-in snapshot versioning.

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use datafusion::datasource::TableType;
use datafusion::execution::context::SessionState;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_physical_plan::memory::MemoryExec;
use dashmap::DashMap;
use lance::dataset::Dataset;
use tracing::{debug, info, instrument};

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
    #[instrument(skip(self))]
    pub async fn open_table(&self, path: &str) -> Result<Arc<dyn TableProvider>> {
        info!(path, "opening Lance dataset");

        let dataset = Dataset::open(path)
            .await
            .map_err(|e| AnamError::Lance(format!("failed to open dataset at '{path}': {e}")))?;

        let schema = Arc::new(dataset.schema().into());
        let provider = Arc::new(LanceTableProvider {
            path: PathBuf::from(path),
            dataset: Arc::new(tokio::sync::RwLock::new(dataset)),
            schema,
        });

        self.datasets
            .insert(path.to_string(), PathBuf::from(path));

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

        let dataset = Dataset::checkout_version(path, version)
            .await
            .map_err(|e| {
                AnamError::Lance(format!(
                    "failed to open dataset at '{path}' version {version}: {e}"
                ))
            })?;

        let schema = Arc::new(dataset.schema().into());
        let provider = Arc::new(LanceTableProvider {
            path: PathBuf::from(path),
            dataset: Arc::new(tokio::sync::RwLock::new(dataset)),
            schema,
        });

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

/// A DataFusion `TableProvider` backed by a Lance dataset.
struct LanceTableProvider {
    path: PathBuf,
    dataset: Arc<tokio::sync::RwLock<Dataset>>,
    schema: SchemaRef,
}

impl std::fmt::Debug for LanceTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceTableProvider")
            .field("path", &self.path)
            .finish()
    }
}

#[async_trait]
impl TableProvider for LanceTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn datafusion::catalog::Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        let dataset = self.dataset.read().await;

        // Build a Lance scanner.
        let mut scanner = dataset.scan();

        // Apply projection.
        if let Some(proj) = projection {
            let field_names: Vec<String> = proj
                .iter()
                .filter_map(|&i| self.schema.field(i).name().to_string().into())
                .collect();
            if !field_names.is_empty() {
                scanner.project(&field_names)
                    .map_err(|e| datafusion_common::DataFusionError::External(
                        Box::new(AnamError::Lance(e.to_string()))
                    ))?;
            }
        }

        // Apply limit.
        if let Some(lim) = limit {
            scanner.limit(lim, None)
                .map_err(|e| datafusion_common::DataFusionError::External(
                    Box::new(AnamError::Lance(e.to_string()))
                ))?;
        }

        // Execute the scan and collect batches.
        let record_batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| datafusion_common::DataFusionError::External(
                Box::new(AnamError::Lance(e.to_string()))
            ))?
            .try_collect()
            .await
            .map_err(|e| datafusion_common::DataFusionError::External(
                Box::new(AnamError::Lance(e.to_string()))
            ))?;

        // Wrap in a MemoryExec for DataFusion.
        let exec = MemoryExec::try_new(
            &[record_batches],
            self.schema.clone(),
            projection.cloned(),
        )?;

        Ok(Arc::new(exec))
    }
}

use futures::TryStreamExt;
