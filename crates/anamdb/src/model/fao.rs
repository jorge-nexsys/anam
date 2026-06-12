//! Function-as-Operator (FAO) abstraction.
//!
//! Query steps that involve neural inference are compiled into explicit,
//! version-stamped functions. This allows the system to swap models at
//! plan-time and track exactly which implementation derived each tuple.

use async_trait::async_trait;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::core::error::Result;
use crate::execution::dispatcher::DeviceType;

/// Device affinity for hardware-dispatched inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceAffinity {
    /// No preference — use any available device.
    Any,
    /// Prefer CPU execution.
    Cpu,
    /// Prefer GPU execution.
    Gpu,
    /// Prefer NPU/accelerator execution.
    Npu,
}

impl DeviceAffinity {
    /// Map this affinity to the dispatcher's [`DeviceType`].
    ///
    /// Returns `None` for `Any` (no preference), which lets the dispatcher
    /// pick the slot with the lowest load.
    pub fn to_device_type(self) -> Option<DeviceType> {
        match self {
            DeviceAffinity::Any => None,
            DeviceAffinity::Cpu => Some(DeviceType::Cpu),
            DeviceAffinity::Gpu => Some(DeviceType::CudaGpu), // default GPU backend
            DeviceAffinity::Npu => Some(DeviceType::Npu),
        }
    }
}

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
    fn output_schema(&self) -> SchemaRef;

    /// Run inference on a batch and return the augmented output batch.
    async fn execute(&self, input: RecordBatch) -> Result<RecordBatch>;

    /// Estimated latency for a batch of the given size (milliseconds).
    fn estimated_latency_ms(&self, batch_size: usize) -> f64;

    /// Estimated accuracy.
    fn estimated_accuracy(&self) -> f64;

    /// Preferred device affinity for hardware dispatch.
    /// Returns `None` for no preference (equivalent to `DeviceAffinity::Any`).
    fn device_affinity(&self) -> Option<DeviceAffinity> {
        None
    }
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
