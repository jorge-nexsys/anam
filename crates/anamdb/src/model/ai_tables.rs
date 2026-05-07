//! AI-Tables: system catalog for managing AI models as first-class citizens.
//!
//! The `__ai_models` table stores metadata for every registered model so the
//! Pareto optimizer can make informed decisions about latency/accuracy trade-offs.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Supported model serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    /// ONNX Runtime model.
    Onnx,
    /// Burn native model.
    Burn,
    /// User-supplied custom operator.
    Custom,
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelFormat::Onnx => write!(f, "onnx"),
            ModelFormat::Burn => write!(f, "burn"),
            ModelFormat::Custom => write!(f, "custom"),
        }
    }
}

/// Device affinity hint for the heterogeneous dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceAffinity {
    /// Runs on any available CPU core.
    Cpu,
    /// Prefers GPU acceleration (CUDA / Metal).
    Gpu,
    /// Prefers Neural Processing Unit.
    Npu,
    /// No preference — dispatcher decides.
    #[default]
    Any,
}

/// A single entry in the `__ai_models` system catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiModelEntry {
    /// Unique model identifier (UUID).
    pub model_id: String,
    /// Human-readable model name.
    pub name: String,
    /// Semantic version string.
    pub version: String,
    /// Serialization format.
    pub format: ModelFormat,
    /// Path to the model artifact on disk.
    pub artifact_path: String,
    /// JSON-serialized Arrow schema for inputs.
    pub input_schema_json: Option<String>,
    /// JSON-serialized Arrow schema for outputs.
    pub output_schema_json: Option<String>,
    /// Measured or estimated average inference latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Measured or estimated accuracy (0.0–1.0).
    pub accuracy: f64,
    /// Model file size in bytes.
    pub size_bytes: u64,
    /// Device affinity hint.
    pub device_affinity: DeviceAffinity,
    /// When this entry was first created.
    pub created_at: DateTime<Utc>,
    /// Last update timestamp.
    pub updated_at: DateTime<Utc>,
}

impl AiModelEntry {
    /// Create a builder for a new model entry.
    pub fn builder(name: impl Into<String>, version: impl Into<String>) -> AiModelEntryBuilder {
        AiModelEntryBuilder {
            name: name.into(),
            version: version.into(),
            format: ModelFormat::Onnx,
            artifact_path: String::new(),
            input_schema_json: None,
            output_schema_json: None,
            avg_latency_ms: 0.0,
            accuracy: 0.0,
            size_bytes: 0,
            device_affinity: DeviceAffinity::Any,
        }
    }
}

/// Builder for [`AiModelEntry`].
pub struct AiModelEntryBuilder {
    name: String,
    version: String,
    format: ModelFormat,
    artifact_path: String,
    input_schema_json: Option<String>,
    output_schema_json: Option<String>,
    avg_latency_ms: f64,
    accuracy: f64,
    size_bytes: u64,
    device_affinity: DeviceAffinity,
}

impl AiModelEntryBuilder {
    /// Set the model format.
    pub fn format(mut self, format: ModelFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the path to the model artifact.
    pub fn artifact_path(mut self, path: impl Into<String>) -> Self {
        self.artifact_path = path.into();
        self
    }

    /// Set the average latency in milliseconds.
    pub fn avg_latency_ms(mut self, ms: f64) -> Self {
        self.avg_latency_ms = ms;
        self
    }

    /// Set the accuracy.
    pub fn accuracy(mut self, acc: f64) -> Self {
        self.accuracy = acc;
        self
    }

    /// Set the model size in bytes.
    pub fn size_bytes(mut self, bytes: u64) -> Self {
        self.size_bytes = bytes;
        self
    }

    /// Set the device affinity.
    pub fn device_affinity(mut self, affinity: DeviceAffinity) -> Self {
        self.device_affinity = affinity;
        self
    }

    /// Set the input schema (as JSON).
    pub fn input_schema(mut self, json: impl Into<String>) -> Self {
        self.input_schema_json = Some(json.into());
        self
    }

    /// Set the output schema (as JSON).
    pub fn output_schema(mut self, json: impl Into<String>) -> Self {
        self.output_schema_json = Some(json.into());
        self
    }

    /// Finalize the builder into an [`AiModelEntry`].
    pub fn build(self) -> AiModelEntry {
        let now = Utc::now();
        AiModelEntry {
            model_id: Uuid::new_v4().to_string(),
            name: self.name,
            version: self.version,
            format: self.format,
            artifact_path: self.artifact_path,
            input_schema_json: self.input_schema_json,
            output_schema_json: self.output_schema_json,
            avg_latency_ms: self.avg_latency_ms,
            accuracy: self.accuracy,
            size_bytes: self.size_bytes,
            device_affinity: self.device_affinity,
            created_at: now,
            updated_at: now,
        }
    }
}
