//! Data ingestion pipelines: convert CSV, Parquet, and JSON into Lance datasets.

use std::path::Path;
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::csv::ReaderBuilder as CsvReaderBuilder;
use datafusion::arrow::datatypes::Schema;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use tracing::{info, instrument};

use crate::core::error::{AnamError, Result};

/// Ingest a CSV file into a new Lance dataset.
#[instrument]
pub async fn ingest_csv(csv_path: &str, lance_path: &str) -> Result<()> {
    info!(csv_path, lance_path, "ingesting CSV → Lance");

    let file = std::fs::File::open(csv_path).map_err(AnamError::Io)?;

    // Infer schema from the first 100 rows.
    let (schema, _) = arrow::csv::reader::Format::default()
        .with_header(true)
        .infer_schema(&file, Some(100))
        .map_err(AnamError::Arrow)?;

    let schema = Arc::new(schema);

    // Re-open file and build reader.
    let file = std::fs::File::open(csv_path).map_err(AnamError::Io)?;
    let reader = CsvReaderBuilder::new(schema.clone())
        .with_header(true)
        .build(file)
        .map_err(AnamError::Arrow)?;

    let batches: Vec<RecordBatch> = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(AnamError::Arrow)?;

    if batches.is_empty() {
        return Err(AnamError::Lance("CSV file produced no record batches".into()));
    }

    // Write to Lance.
    let reader = lance::arrow::RecordBatchIterator::new(
        batches.into_iter().map(Ok),
        schema,
    );

    let params = WriteParams {
        mode: WriteMode::Create,
        ..Default::default()
    };

    Dataset::write(reader, lance_path, Some(params))
        .await
        .map_err(|e| AnamError::Lance(format!("failed to write Lance dataset: {e}")))?;

    info!(lance_path, "CSV ingestion complete");
    Ok(())
}

/// Ingest Arrow RecordBatches directly into a Lance dataset.
#[instrument(skip(batches))]
pub async fn ingest_batches(
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
    lance_path: &str,
    mode: WriteMode,
) -> Result<()> {
    info!(
        lance_path,
        batch_count = batches.len(),
        "ingesting batches → Lance"
    );

    if batches.is_empty() {
        return Err(AnamError::Lance("no record batches to ingest".into()));
    }

    let reader = lance::arrow::RecordBatchIterator::new(
        batches.into_iter().map(Ok),
        schema,
    );

    let params = WriteParams {
        mode,
        ..Default::default()
    };

    Dataset::write(reader, lance_path, Some(params))
        .await
        .map_err(|e| AnamError::Lance(format!("failed to write Lance dataset: {e}")))?;

    info!(lance_path, "batch ingestion complete");
    Ok(())
}
