//! Unified error types for AnamDB.

use thiserror::Error;

/// Canonical error type for all AnamDB operations.
#[derive(Error, Debug)]
pub enum AnamError {
    // ── Infra ──────────────────────────────────────────────────────────
    /// An I/O error from the storage layer.
    #[error("storage I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Arrow / DataFusion interop failure.
    #[error("arrow error: {0}")]
    Arrow(#[from] datafusion::arrow::error::ArrowError),

    /// DataFusion planning or execution error.
    #[error("datafusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),

    /// Lance storage error.
    #[error("lance error: {0}")]
    Lance(String),

    // ── Model Manager ──────────────────────────────────────────────────
    /// A requested model was not found in the AI-Tables catalog.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// Inference runtime failure.
    #[error("inference error: {0}")]
    Inference(String),

    // ── Logic Layer ────────────────────────────────────────────────────
    /// Datalog compilation or evaluation error.
    #[error("logic error: {0}")]
    Logic(String),

    /// NL-to-Datalog translation failure.
    #[error("NL compilation error: {0}")]
    NlCompilation(String),

    // ── HITL ───────────────────────────────────────────────────────────
    /// The query requires human clarification before it can continue.
    #[error("clarification required: {0}")]
    ClarificationRequired(String),

    /// A semantic anomaly was detected in intermediate results.
    #[error("semantic anomaly: {0}")]
    SemanticAnomaly(String),

    // ── Query ──────────────────────────────────────────────────────────
    /// Malformed or unsupported SQL/query syntax.
    #[error("query parse error: {0}")]
    QueryParse(String),

    /// Multi-objective constraint violation (no feasible Pareto plan).
    #[error("no feasible plan satisfying constraints: {0}")]
    NoFeasiblePlan(String),

    // ── Dispatcher ─────────────────────────────────────────────────────
    /// Hardware dispatch failure.
    #[error("dispatch error: {0}")]
    Dispatch(String),

    // ── Generic ────────────────────────────────────────────────────────
    /// Catch-all for unexpected errors.
    #[error("{0}")]
    Internal(String),

    /// Serialization / deserialization error.
    #[error("serde error: {0}")]
    Serde(String),

    /// HTTP / network error.
    #[error("http error: {0}")]
    Http(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, AnamError>;

impl From<reqwest::Error> for AnamError {
    fn from(e: reqwest::Error) -> Self {
        AnamError::Http(e.to_string())
    }
}

impl From<serde_json::Error> for AnamError {
    fn from(e: serde_json::Error) -> Self {
        AnamError::Serde(e.to_string())
    }
}

impl From<bincode::Error> for AnamError {
    fn from(e: bincode::Error) -> Self {
        AnamError::Serde(e.to_string())
    }
}
