//! Streaming Lance TableProvider — wraps Lance datasets as proper DataFusion
//! `TableProvider`s with push-down projection and filter support.
//!
//! This replaces the eager `scan_to_memtable` approach with Lance's native
//! streaming scanner, which avoids loading entire datasets into memory.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::TableProvider;
use datafusion::datasource::TableType;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use lance::Dataset;
use tracing::{debug, info};

use crate::core::error::{AnamError, Result};

/// A streaming TableProvider that wraps a Lance dataset.
///
/// Unlike MemTable, this provider does NOT load all data into memory.
/// It uses Lance's native scanner with push-down projection and
/// filter support for efficient query execution.
#[derive(Debug)]
pub struct LanceStreamingProvider {
    dataset: Arc<Dataset>,
    schema: SchemaRef,
}

impl LanceStreamingProvider {
    /// Open a Lance dataset and create a streaming provider.
    pub async fn open(path: &str) -> Result<Self> {
        info!(path, "opening Lance dataset (streaming)");

        let dataset = Dataset::open(path)
            .await
            .map_err(|e| AnamError::Lance(format!("failed to open dataset at '{path}': {e}")))?;

        let lance_schema = dataset.schema();
        let arrow_schema: arrow_schema::Schema = arrow_schema::Schema::from(lance_schema);
        let schema: SchemaRef = Arc::new(arrow_schema);

        debug!(
            columns = schema.fields().len(),
            "opened Lance dataset with streaming provider"
        );

        Ok(Self {
            dataset: Arc::new(dataset),
            schema,
        })
    }

    /// Access the underlying Lance dataset (for mutations).
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }
}

#[async_trait]
impl TableProvider for LanceStreamingProvider {
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
        _state: &dyn datafusion::catalog::Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> std::result::Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        // Use Lance's native LanceTableProvider for the actual scan.
        let lance_provider =
            lance::datafusion::LanceTableProvider::new(self.dataset.clone(), false, false);

        lance_provider
            .scan(_state, projection, filters, limit)
            .await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> std::result::Result<Vec<datafusion::logical_expr::TableProviderFilterPushDown>, DataFusionError>
    {
        // Lance supports filter push-down for simple predicates.
        Ok(filters
            .iter()
            .map(|_| datafusion::logical_expr::TableProviderFilterPushDown::Inexact)
            .collect())
    }
}
