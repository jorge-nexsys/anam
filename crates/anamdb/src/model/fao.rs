//! Function-as-Operator (FAO) abstraction.
//!
//! Query steps that involve neural inference are compiled into explicit,
//! version-stamped functions. This allows the system to swap models at
//! plan-time and track exactly which implementation derived each tuple.

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::Schema;
use async_trait::async_trait;
use std::sync::Arc;

use crate::core::error::Result;

/// A single inference function that can be wired into a DataFusion physical plan.
///
/// Each operator is version-stamped so provenance traces can record the exact
/// function + version that produced a derived tuple.
#[async_trait]
pub trait FaoOperator: Send + Sync + std::fmt::Debug {
    /// Unique function identifier (e.g. `"classify_fraud"`).
    fn function_id(&self) -> &str;

    /// Semantic version of this operator implementation.
    fn version(&self) -> &str;

    /// The model ID from AI-Tables that backs this operator.
    fn model_id(&self) -> &str;

    /// Expected input schema.
    fn input_schema(&self) -> &Arc<Schema>;

    /// Produced output schema.
    fn output_schema(&self) -> &Arc<Schema>;

    /// Run inference on a batch and return the augmented output batch.
    ///
    /// The implementation is responsible for:
    /// 1. Projecting relevant columns from the input.
    /// 2. Running inference (ONNX / Burn / custom).
    /// 3. Attaching confidence and provenance columns.
    async fn execute(&self, input: RecordBatch) -> Result<RecordBatch>;

    /// Estimated latency for a batch of the given size (milliseconds).
    fn estimated_latency_ms(&self, batch_size: usize) -> f64;

    /// Estimated accuracy.
    fn estimated_accuracy(&self) -> f64;
}

/// A versioned reference to an FAO operator, used by the optimizer to
/// enumerate candidate plans.
#[derive(Debug, Clone)]
pub struct FaoRef {
    /// Function identifier.
    pub function_id: String,
    /// Version string.
    pub version: String,
    /// Backing model ID.
    pub model_id: String,
    /// Estimated latency for a 1 000-row batch.
    pub est_latency_ms: f64,
    /// Estimated accuracy.
    pub est_accuracy: f64,
}

impl FaoRef {
    /// Create from a concrete operator.
    pub fn from_operator(op: &dyn FaoOperator) -> Self {
        Self {
            function_id: op.function_id().to_string(),
            version: op.version().to_string(),
            model_id: op.model_id().to_string(),
            est_latency_ms: op.estimated_latency_ms(1_000),
            est_accuracy: op.estimated_accuracy(),
        }
    }
}
