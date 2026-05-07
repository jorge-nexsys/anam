//! Write path: INSERT, UPDATE (via merge), and DELETE support for Lance tables.
//!
//! These operations map SQL mutations onto Lance's native append, merge, and
//! delete APIs. Each mutation creates a new dataset version (Lance is append-only
//! with tombstone-based deletes).

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchIterator;
use arrow_schema::Schema;
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use tracing::{info, instrument};

use crate::core::error::{AnamError, Result};

/// Result of a write operation.
#[derive(Debug)]
pub struct WriteResult {
    /// Number of rows affected.
    pub rows_affected: usize,
    /// New dataset version after the mutation.
    pub new_version: u64,
}

/// Append new rows to a Lance dataset (INSERT).
#[instrument(skip(batches))]
pub async fn insert_rows(
    path: &str,
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
) -> Result<WriteResult> {
    info!(path, batch_count = batches.len(), "INSERT into Lance");

    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();

    if batches.is_empty() {
        return Err(AnamError::Lance("no rows to insert".into()));
    }

    let batch_reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

    let params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };

    let dataset = Dataset::write(batch_reader, path, Some(params))
        .await
        .map_err(|e| AnamError::Lance(format!("INSERT failed: {e}")))?;

    let version = dataset.version().version;
    info!(rows, version, "INSERT complete");

    Ok(WriteResult {
        rows_affected: rows,
        new_version: version,
    })
}

/// Delete rows matching a SQL predicate from a Lance dataset (DELETE).
///
/// The predicate uses Lance's filter syntax (e.g. `"amount > 1000 AND region = 'EU'"`).
#[instrument]
pub async fn delete_rows(path: &str, predicate: &str) -> Result<WriteResult> {
    info!(path, predicate, "DELETE from Lance");

    let mut dataset = Dataset::open(path)
        .await
        .map_err(|e| AnamError::Lance(format!("failed to open dataset: {e}")))?;

    let result = dataset
        .delete(predicate)
        .await
        .map_err(|e| AnamError::Lance(format!("DELETE failed: {e}")))?;

    let version = dataset.version().version;
    let rows_deleted = result.num_deleted_rows as usize;
    info!(rows_deleted, version, "DELETE complete");

    Ok(WriteResult {
        rows_affected: rows_deleted,
        new_version: version,
    })
}

/// Overwrite a Lance dataset with new data (REPLACE / bulk update).
///
/// This is a full overwrite — all existing data is replaced.
/// For partial updates, use `merge_rows` once Lance's merge API supports it.
#[instrument(skip(batches))]
pub async fn overwrite_rows(
    path: &str,
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
) -> Result<WriteResult> {
    info!(path, batch_count = batches.len(), "OVERWRITE Lance dataset");

    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();

    if batches.is_empty() {
        return Err(AnamError::Lance("no rows for overwrite".into()));
    }

    let batch_reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

    let params = WriteParams {
        mode: WriteMode::Overwrite,
        ..Default::default()
    };

    let dataset = Dataset::write(batch_reader, path, Some(params))
        .await
        .map_err(|e| AnamError::Lance(format!("OVERWRITE failed: {e}")))?;

    let version = dataset.version().version;
    info!(rows, version, "OVERWRITE complete");

    Ok(WriteResult {
        rows_affected: rows,
        new_version: version,
    })
}
