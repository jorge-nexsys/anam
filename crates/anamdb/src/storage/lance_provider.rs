//! Lance table provider: wraps Lance datasets as DataFusion `TableProvider`s.
//!
//! Supports `AS OF 'timestamp'` via Lance's built-in snapshot versioning.

use std::any::Any;
use std::path::PathBuf;
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use datafusion::datasource::memory::MemTable;
use datafusion::datasource::TableType;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use dashmap::DashMap;
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
    #[instrument(skip(self))]
    pub async fn open_table(&self, path: &str) -> Result<Arc<dyn TableProvider>> {
        info!(path, "opening Lance dataset");

        let dataset = Dataset::open(path)
            .await
            .map_err(|e| AnamError::Lance(format!("failed to open dataset at '{path}': {e}")))?;

        let lance_schema = dataset.schema();
        let arrow_fields: Vec<datafusion::arrow::datatypes::Field> = lance_schema
            .fields
            .iter()
            .map(|f| {
                datafusion::arrow::datatypes::Field::new(
                    &f.name,
                    lance_to_arrow_dtype(&f.data_type()),
                    f.nullable,
                )
            })
            .collect();
        let schema: SchemaRef = Arc::new(datafusion::arrow::datatypes::Schema::new(arrow_fields));

        // Read all data into memory for the MVP.
        let mut scanner = dataset.scan();
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| AnamError::Lance(format!("scan failed: {e}")))?
            .try_collect()
            .await
            .map_err(|e: lance::Error| AnamError::Lance(format!("collect failed: {e}")))?;

        let provider = Arc::new(
            MemTable::try_new(schema, vec![batches])
                .map_err(|e| AnamError::DataFusion(e))?
        );

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

        let lance_schema = dataset.schema();
        let arrow_fields: Vec<datafusion::arrow::datatypes::Field> = lance_schema
            .fields
            .iter()
            .map(|f| {
                datafusion::arrow::datatypes::Field::new(
                    &f.name,
                    lance_to_arrow_dtype(&f.data_type()),
                    f.nullable,
                )
            })
            .collect();
        let schema: SchemaRef = Arc::new(datafusion::arrow::datatypes::Schema::new(arrow_fields));

        let mut scanner = dataset.scan();
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| AnamError::Lance(format!("scan failed: {e}")))?
            .try_collect()
            .await
            .map_err(|e: lance::Error| AnamError::Lance(format!("collect failed: {e}")))?;

        let provider = Arc::new(
            MemTable::try_new(schema, vec![batches])
                .map_err(|e| AnamError::DataFusion(e))?
        );

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

/// Convert a Lance data type to an Arrow data type.
fn lance_to_arrow_dtype(dt: &lance::datatypes::DataType) -> datafusion::arrow::datatypes::DataType {
    use datafusion::arrow::datatypes::DataType;
    // Lance data types map to arrow fairly directly.
    // For the MVP we handle the most common types.
    match dt {
        lance::datatypes::DataType::Boolean => DataType::Boolean,
        lance::datatypes::DataType::Int8 => DataType::Int8,
        lance::datatypes::DataType::Int16 => DataType::Int16,
        lance::datatypes::DataType::Int32 => DataType::Int32,
        lance::datatypes::DataType::Int64 => DataType::Int64,
        lance::datatypes::DataType::UInt8 => DataType::UInt8,
        lance::datatypes::DataType::UInt16 => DataType::UInt16,
        lance::datatypes::DataType::UInt32 => DataType::UInt32,
        lance::datatypes::DataType::UInt64 => DataType::UInt64,
        lance::datatypes::DataType::Float16 => DataType::Float16,
        lance::datatypes::DataType::Float32 => DataType::Float32,
        lance::datatypes::DataType::Float64 => DataType::Float64,
        lance::datatypes::DataType::Utf8 => DataType::Utf8,
        lance::datatypes::DataType::Binary => DataType::Binary,
        lance::datatypes::DataType::Date32 => DataType::Date32,
        lance::datatypes::DataType::Date64 => DataType::Date64,
        _ => DataType::Binary, // fallback for unsupported types
    }
}
