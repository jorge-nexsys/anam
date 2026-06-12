//! Knowledge Graph Embedding (KGE) scoring FAO operators.
//!
//! Implements geometric embedding scoring functions as native FAO operators:
//! - **TransE**: Score = -‖h + r - t‖₂
//! - **TransH**: Projects entities onto relation-specific hyperplanes before TransE scoring.
//!
//! These operators accept input batches with `head_embedding`, `relation_embedding`,
//! and `tail_embedding` Float32 columns, and produce a `score` Float64 column.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, Float32Array, Float64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use tracing::debug;

use crate::core::error::{AnamError, Result};
use crate::model::fao::{DeviceAffinity, FaoOperator};

/// Embedding dimension (number of Float32 values per embedding).
///
/// We detect this dynamically from the input array length.
fn infer_embedding_dim(batch: &RecordBatch) -> usize {
    // Check for explicit metadata first.
    if let Some(dim_str) = batch.schema().metadata().get("embedding_dim") {
        if let Ok(dim) = dim_str.parse::<usize>() {
            return dim;
        }
    }

    // Fallback: infer from array length / num_rows.
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return 0;
    }

    if let Some((idx, _)) = batch.schema().column_with_name("head_embedding") {
        let col = batch.column(idx);
        if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
            return arr.len() / num_rows;
        }
    }
    128 // default embedding dimension
}

/// Extract a flat Float32 embedding column as `Vec<Vec<f32>>` of shape `[rows, dim]`.
fn extract_embeddings(batch: &RecordBatch, col_name: &str, dim: usize) -> Result<Vec<Vec<f32>>> {
    let (idx, _) = batch
        .schema()
        .column_with_name(col_name)
        .ok_or_else(|| AnamError::Inference(format!("missing column '{col_name}'")))?;

    let col = batch.column(idx);
    let arr = col
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| AnamError::Inference(format!("column '{col_name}' is not Float32")))?;

    // When embeddings are stored flat, the logical row count is arr.len() / dim.
    let num_rows = if dim > 0 { arr.len() / dim } else { 0 };
    let mut embeddings = Vec::with_capacity(num_rows);

    for row in 0..num_rows {
        let start = row * dim;
        let end = start + dim;
        if end > arr.len() {
            return Err(AnamError::Inference(format!(
                "embedding column '{col_name}' has insufficient values: expected {} got {}",
                end,
                arr.len()
            )));
        }
        let embedding: Vec<f32> = (start..end).map(|i| arr.value(i)).collect();
        embeddings.push(embedding);
    }

    Ok(embeddings)
}

/// L2 norm of a vector.
fn l2_norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt()
}

// ── TransE Scorer ──────────────────────────────────────────────────────────

/// TransE scoring: Score = -‖h + r - t‖₂.
///
/// Higher score (closer to 0) means the triple (h, r, t) is more plausible.
pub struct TransEScorer {
    function_id: String,
    version: String,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl TransEScorer {
    /// Create a new TransE scorer.
    pub fn new(_embedding_dim: usize) -> Self {
        let input_fields = vec![
            Field::new("head_embedding", DataType::Float32, false),
            Field::new("relation_embedding", DataType::Float32, false),
            Field::new("tail_embedding", DataType::Float32, false),
        ];
        let output_fields = vec![Field::new("score", DataType::Float64, false)];

        Self {
            function_id: "transe_score".to_string(),
            version: "1.0.0".to_string(),
            input_schema: Arc::new(Schema::new(input_fields)),
            output_schema: Arc::new(Schema::new(output_fields)),
        }
    }

    /// Score a single triple: -‖h + r - t‖₂.
    fn score_triple(head: &[f32], relation: &[f32], tail: &[f32]) -> f64 {
        let diff: Vec<f32> = head
            .iter()
            .zip(relation.iter())
            .zip(tail.iter())
            .map(|((h, r), t)| h + r - t)
            .collect();
        -l2_norm(&diff)
    }
}

impl std::fmt::Debug for TransEScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransEScorer")
            .field("function_id", &self.function_id)
            .finish()
    }
}

#[async_trait]
impl FaoOperator for TransEScorer {
    fn function_id(&self) -> &str {
        &self.function_id
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn model_id(&self) -> &str {
        "kge_transe"
    }

    fn input_schema(&self) -> &Arc<Schema> {
        &self.input_schema
    }

    fn output_schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn estimated_latency_ms(&self, _batch_size: usize) -> f64 {
        0.1 // Very fast — pure math.
    }

    fn estimated_accuracy(&self) -> f64 {
        0.80 // Depends on trained embeddings.
    }

    fn device_affinity(&self) -> Option<DeviceAffinity> {
        Some(DeviceAffinity::Cpu) // Pure math, no GPU needed.
    }

    async fn execute(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let dim = infer_embedding_dim(&batch);
        debug!(dim, rows = batch.num_rows(), "TransE scoring");

        let heads = extract_embeddings(&batch, "head_embedding", dim)?;
        let relations = extract_embeddings(&batch, "relation_embedding", dim)?;
        let tails = extract_embeddings(&batch, "tail_embedding", dim)?;

        let scores: Vec<f64> = heads
            .iter()
            .zip(relations.iter())
            .zip(tails.iter())
            .map(|((h, r), t)| Self::score_triple(h, r, t))
            .collect();

        let score_array = Float64Array::from(scores);
        let result = RecordBatch::try_new(self.output_schema.clone(), vec![Arc::new(score_array)])
            .map_err(AnamError::Arrow)?;

        Ok(result)
    }
}

// ── TransH Scorer ──────────────────────────────────────────────────────────

/// TransH scoring: Projects entities onto relation-specific hyperplanes, then
/// applies TransE scoring in the projected space.
///
/// Each relation defines a hyperplane normal `w_r`. Entities are projected:
///   h_⊥ = h - w_r^T h · w_r
///   t_⊥ = t - w_r^T t · w_r
///
/// Score = -‖h_⊥ + d_r - t_⊥‖₂
///
/// The input batch must include an additional `hyperplane_normal` Float32 column
/// alongside the standard `relation_embedding` (which serves as `d_r`).
pub struct TransHScorer {
    function_id: String,
    version: String,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl TransHScorer {
    /// Create a new TransH scorer.
    pub fn new(_embedding_dim: usize) -> Self {
        let input_fields = vec![
            Field::new("head_embedding", DataType::Float32, false),
            Field::new("relation_embedding", DataType::Float32, false),
            Field::new("tail_embedding", DataType::Float32, false),
            Field::new("hyperplane_normal", DataType::Float32, false),
        ];
        let output_fields = vec![Field::new("score", DataType::Float64, false)];

        Self {
            function_id: "transh_score".to_string(),
            version: "1.0.0".to_string(),
            input_schema: Arc::new(Schema::new(input_fields)),
            output_schema: Arc::new(Schema::new(output_fields)),
        }
    }

    /// Project an entity onto the hyperplane defined by normal `w`.
    /// e_⊥ = e - (w^T e) * w
    fn project_to_hyperplane(entity: &[f32], normal: &[f32]) -> Vec<f32> {
        let dot: f32 = entity.iter().zip(normal.iter()).map(|(e, w)| e * w).sum();
        entity
            .iter()
            .zip(normal.iter())
            .map(|(e, w)| e - dot * w)
            .collect()
    }
}

impl std::fmt::Debug for TransHScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransHScorer")
            .field("function_id", &self.function_id)
            .finish()
    }
}

#[async_trait]
impl FaoOperator for TransHScorer {
    fn function_id(&self) -> &str {
        &self.function_id
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn model_id(&self) -> &str {
        "kge_transh"
    }

    fn input_schema(&self) -> &Arc<Schema> {
        &self.input_schema
    }

    fn output_schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn estimated_latency_ms(&self, _batch_size: usize) -> f64 {
        0.15 // Slightly more expensive than TransE due to projection.
    }

    fn estimated_accuracy(&self) -> f64 {
        0.85 // Generally more accurate than TransE.
    }

    fn device_affinity(&self) -> Option<DeviceAffinity> {
        Some(DeviceAffinity::Cpu)
    }

    async fn execute(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let dim = infer_embedding_dim(&batch);
        debug!(dim, rows = batch.num_rows(), "TransH scoring");

        let heads = extract_embeddings(&batch, "head_embedding", dim)?;
        let relations = extract_embeddings(&batch, "relation_embedding", dim)?;
        let tails = extract_embeddings(&batch, "tail_embedding", dim)?;
        let normals = extract_embeddings(&batch, "hyperplane_normal", dim)?;

        let scores: Vec<f64> = heads
            .iter()
            .zip(relations.iter())
            .zip(tails.iter())
            .zip(normals.iter())
            .map(|(((h, r), t), w)| {
                let h_proj = Self::project_to_hyperplane(h, w);
                let t_proj = Self::project_to_hyperplane(t, w);
                TransEScorer::score_triple(&h_proj, r, &t_proj)
            })
            .collect();

        let score_array = Float64Array::from(scores);
        let result = RecordBatch::try_new(self.output_schema.clone(), vec![Arc::new(score_array)])
            .map_err(AnamError::Arrow)?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding_batch(
        heads: Vec<Vec<f32>>,
        relations: Vec<Vec<f32>>,
        tails: Vec<Vec<f32>>,
    ) -> RecordBatch {
        let dim = heads[0].len();

        let head_flat: Vec<f32> = heads.into_iter().flatten().collect();
        let rel_flat: Vec<f32> = relations.into_iter().flatten().collect();
        let tail_flat: Vec<f32> = tails.into_iter().flatten().collect();

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("embedding_dim".to_string(), dim.to_string());

        let schema = Arc::new(
            Schema::new(vec![
                Field::new("head_embedding", DataType::Float32, false),
                Field::new("relation_embedding", DataType::Float32, false),
                Field::new("tail_embedding", DataType::Float32, false),
            ])
            .with_metadata(metadata),
        );

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float32Array::from(head_flat)),
                Arc::new(Float32Array::from(rel_flat)),
                Arc::new(Float32Array::from(tail_flat)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn kge_transe_exact_match() {
        let scorer = TransEScorer::new(3);

        // h + r == t  →  score should be 0.0 (perfect).
        let batch = make_embedding_batch(
            vec![vec![1.0, 0.0, 0.0]],
            vec![vec![0.0, 1.0, 0.0]],
            vec![vec![1.0, 1.0, 0.0]], // = h + r
        );

        let result = scorer.execute(batch).await.unwrap();
        let scores = result
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert!(
            (scores.value(0) - 0.0).abs() < 1e-6,
            "Expected score ≈ 0.0 for exact match, got {}",
            scores.value(0)
        );
    }

    #[tokio::test]
    async fn kge_transe_imperfect_match() {
        let scorer = TransEScorer::new(3);

        let batch = make_embedding_batch(
            vec![vec![1.0, 0.0, 0.0]],
            vec![vec![0.0, 1.0, 0.0]],
            vec![vec![0.0, 0.0, 1.0]], // ≠ h + r
        );

        let result = scorer.execute(batch).await.unwrap();
        let scores = result
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert!(
            scores.value(0) < -0.5,
            "Expected negative score for mismatch, got {}",
            scores.value(0)
        );
    }

    #[tokio::test]
    async fn kge_transh_projection() {
        let scorer = TransHScorer::new(3);

        // Normal pointing along x-axis → projection removes x component.
        let head_flat: Vec<f32> = vec![1.0, 2.0, 0.0];
        let rel_flat: Vec<f32> = vec![0.0, 1.0, 0.0];
        let tail_flat: Vec<f32> = vec![0.0, 3.0, 0.0]; // h_proj + r = (0, 2, 0) + (0, 1, 0) = (0, 3, 0) = t_proj
        let normal_flat: Vec<f32> = vec![1.0, 0.0, 0.0]; // Normalize along x.

        let schema = Arc::new(Schema::new(vec![
            Field::new("head_embedding", DataType::Float32, false),
            Field::new("relation_embedding", DataType::Float32, false),
            Field::new("tail_embedding", DataType::Float32, false),
            Field::new("hyperplane_normal", DataType::Float32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float32Array::from(head_flat)),
                Arc::new(Float32Array::from(rel_flat)),
                Arc::new(Float32Array::from(tail_flat)),
                Arc::new(Float32Array::from(normal_flat)),
            ],
        )
        .unwrap();

        let result = scorer.execute(batch).await.unwrap();
        let scores = result
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert!(
            (scores.value(0) - 0.0).abs() < 1e-5,
            "Expected score ≈ 0.0 after hyperplane projection, got {}",
            scores.value(0)
        );
    }

    #[tokio::test]
    async fn kge_multiple_triples() {
        let scorer = TransEScorer::new(2);

        let batch = make_embedding_batch(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]], // Both are exact matches.
        );

        let result = scorer.execute(batch).await.unwrap();
        assert_eq!(result.num_rows(), 2);

        let scores = result
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        for i in 0..2 {
            assert!(
                (scores.value(i) - 0.0).abs() < 1e-6,
                "Row {i}: expected score ≈ 0.0, got {}",
                scores.value(i)
            );
        }
    }
}
