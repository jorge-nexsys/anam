//! ONNX Runtime inference adapter.
//!
//! Wraps `ort` to implement [`FaoOperator`] for ONNX models, with support for
//! CUDA and CoreML execution providers. Actively integrates with the
//! [`DevicePool`] for heterogeneous hardware dispatch and load management.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, ArrayRef, Float32Array, Float64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Schema, SchemaRef};
use ort::session::Session as OrtSession;
use parking_lot::Mutex;
use tracing::{debug, info, instrument};

use crate::core::error::{AnamError, Result};
use crate::execution::dispatcher::{DevicePool, ExecutionJob};
use crate::model::fao::{DeviceAffinity, FaoOperator};

/// An [`FaoOperator`] backed by an ONNX Runtime session.
///
/// When a [`DevicePool`] is attached, inference execution will actively
/// request a device slot before running and release it afterwards,
/// enabling load-balanced multi-device inference.
#[derive(Debug)]
pub struct OnnxFaoOperator {
    function_id: String,
    version: String,
    model_id: String,
    session: Mutex<OrtSession>,
    input_schema: Arc<Schema>,
    output_schema: Arc<Schema>,
    avg_latency_ms: f64,
    accuracy: f64,
    /// Device affinity preference for dispatch.
    affinity: DeviceAffinity,
    /// Optional device pool for hardware-aware dispatch.
    device_pool: Option<Arc<DevicePool>>,
}

impl OnnxFaoOperator {
    /// Load an ONNX model from disk and wrap it as an FAO operator.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all, fields(model_path = %model_path.as_ref().display()))]
    pub fn load(
        model_path: impl AsRef<std::path::Path>,
        function_id: impl Into<String>,
        version: impl Into<String>,
        model_id: impl Into<String>,
        input_schema: Arc<Schema>,
        output_schema: Arc<Schema>,
        avg_latency_ms: f64,
        accuracy: f64,
    ) -> Result<Self> {
        let session = OrtSession::builder()
            .map_err(|e| AnamError::Inference(format!("failed to create ORT builder: {e}")))?
            .commit_from_file(model_path.as_ref())
            .map_err(|e| AnamError::Inference(format!("failed to load ONNX model: {e}")))?;

        debug!("loaded ONNX model");

        Ok(Self {
            function_id: function_id.into(),
            version: version.into(),
            model_id: model_id.into(),
            session: Mutex::new(session),
            input_schema,
            output_schema,
            avg_latency_ms,
            accuracy,
            affinity: DeviceAffinity::Any,
            device_pool: None,
        })
    }

    /// Attach a device pool for hardware-aware inference dispatch.
    ///
    /// When set, every call to [`execute()`](FaoOperator::execute) will:
    /// 1. Request a device slot from the pool (respecting device affinity).
    /// 2. Run inference.
    /// 3. Release the slot (even on error).
    pub fn with_device_pool(mut self, pool: Arc<DevicePool>) -> Self {
        self.device_pool = Some(pool);
        self
    }

    /// Set the device affinity preference.
    pub fn with_affinity(mut self, affinity: DeviceAffinity) -> Self {
        self.affinity = affinity;
        self
    }
}

#[async_trait]
impl FaoOperator for OnnxFaoOperator {
    fn function_id(&self) -> &str {
        &self.function_id
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn input_schema(&self) -> &Arc<Schema> {
        &self.input_schema
    }

    fn output_schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    async fn execute(&self, input: RecordBatch) -> Result<RecordBatch> {
        let num_rows = input.num_rows();

        // ── Device Pool dispatch ──────────────────────────────────────
        let assignment = if let Some(pool) = &self.device_pool {
            let job = ExecutionJob {
                job_id: format!(
                    "{}@{}:{}",
                    self.function_id,
                    self.version,
                    uuid::Uuid::new_v4()
                ),
                preferred_device: self.affinity.to_device_type(),
                est_cpu_time_ms: self.avg_latency_ms * (num_rows as f64 / 1000.0).max(1.0),
                est_memory_bytes: (num_rows * 4 * self.input_schema.fields().len()) as u64,
            };

            match pool.dispatch(job) {
                Ok(a) => {
                    info!(
                        fao = %self.function_id,
                        device = %a.slot.name,
                        est_time_ms = a.est_time_ms,
                        "dispatched inference to device slot"
                    );
                    Some(a)
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "device pool dispatch failed — running on default device"
                    );
                    None
                }
            }
        } else {
            None
        };

        // ── Run inference ─────────────────────────────────────────────
        let result = self.run_inference(input, num_rows);

        // ── Release device slot ───────────────────────────────────────
        if let (Some(pool), Some(a)) = (&self.device_pool, &assignment) {
            pool.complete_job(a);
            debug!(
                fao = %self.function_id,
                device = %a.slot.name,
                "released device slot"
            );
        }

        result
    }

    fn estimated_latency_ms(&self, batch_size: usize) -> f64 {
        self.avg_latency_ms * (batch_size as f64 / 1000.0).max(1.0)
    }

    fn estimated_accuracy(&self) -> f64 {
        self.accuracy
    }

    fn device_affinity(&self) -> Option<DeviceAffinity> {
        Some(self.affinity)
    }
}

impl OnnxFaoOperator {
    /// Core inference logic, separated so device pool dispatch wraps it cleanly.
    fn run_inference(&self, input: RecordBatch, num_rows: usize) -> Result<RecordBatch> {
        // Collect numeric columns into a flat f32 tensor.
        let mut flat_data: Vec<f32> = Vec::new();
        let mut num_features = 0usize;

        for col_idx in 0..input.num_columns() {
            let col = input.column(col_idx);
            match col.data_type() {
                DataType::Float32 => {
                    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
                        flat_data.extend(arr.values().iter());
                        num_features += 1;
                    }
                }
                DataType::Float64 => {
                    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        flat_data.extend(arr.values().iter().map(|v| *v as f32));
                        num_features += 1;
                    }
                }
                _ => continue,
            }
        }

        if num_features == 0 {
            return Err(AnamError::Inference(
                "no numeric columns found in input batch".into(),
            ));
        }

        // Create ONNX tensor [num_rows, num_features] using (shape, Vec<T>).
        let shape = vec![num_rows, num_features];
        let input_tensor = ort::value::Tensor::from_array((shape, flat_data))
            .map_err(|e| AnamError::Inference(format!("tensor creation failed: {e}")))?;

        // Run inference.
        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| AnamError::Inference(format!("inference failed: {e}")))?;

        // Extract output (assume single output).
        let output_values: Vec<f64> = if let Some((_name, output)) = outputs.iter().next() {
            let (_shape, data) = output
                .try_extract_tensor::<f32>()
                .map_err(|e| AnamError::Inference(format!("output extraction failed: {e}")))?;
            data.iter().map(|v| *v as f64).collect()
        } else {
            return Err(AnamError::Inference("model produced no outputs".into()));
        };

        // Build output RecordBatch with the score column.
        let score_array: ArrayRef = Arc::new(Float64Array::from(output_values));
        let output_batch = RecordBatch::try_new(self.output_schema.clone(), vec![score_array])
            .map_err(AnamError::Arrow)?;

        Ok(output_batch)
    }
}
