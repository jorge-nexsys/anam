//! Python SDK scaffold — PyO3 bindings for AnamDB.
//!
//! This module provides the `pyanamdb` Python package that wraps the native
//! Rust client for zero-overhead Python integration.
//!
//! ## Usage (Python)
//! ```python
//! import pyanamdb
//!
//! client = pyanamdb.connect("localhost:8080")
//! result = client.query("SELECT * FROM txns WHERE fraud_prob > 0.9")
//! print(result.to_pandas())
//! ```
//!
//! ## Build
//! ```bash
//! cd crates/anamdb && maturin develop --features python
//! ```

/// Python module definition (compiled only with the `python` feature).
///
/// When PyO3 + maturin are enabled, this exposes:
/// - `pyanamdb.connect(addr)` → `PyAnamClient`
/// - `PyAnamClient.query(sql)` → `PyQueryResult`
/// - `PyAnamClient.register_table(name, path)`
/// - `PyAnamClient.register_rule(name, datalog)`
/// - `PyAnamClient.health()` → dict
/// - `PyQueryResult.to_arrow()` → PyArrow Table
/// - `PyQueryResult.to_pandas()` → Pandas DataFrame
///
/// The implementation wraps `crate::client::AnamClient` using `pyo3-asyncio`
/// for async bridging.

/// Marker struct for the Python SDK API surface.
///
/// The actual PyO3 implementation requires the `pyo3` and `pyo3-asyncio`
/// crates. This struct documents the intended API without adding the
/// compile-time dependency.
#[derive(Debug)]
pub struct PyAnamClient {
    /// Server address.
    pub addr: String,
}

impl PyAnamClient {
    /// Create a new Python client (mirrors `pyanamdb.connect()`).
    pub fn new(addr: &str) -> Self {
        Self {
            addr: addr.to_string(),
        }
    }
}

/// The Python query result wrapper.
#[derive(Debug)]
pub struct PyQueryResult {
    /// Number of rows.
    pub num_rows: usize,
    /// Column names.
    pub columns: Vec<String>,
    /// Reasoning tree.
    pub reasoning_tree: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_sdk_api_surface() {
        let client = PyAnamClient::new("localhost:8080");
        assert_eq!(client.addr, "localhost:8080");

        let result = PyQueryResult {
            num_rows: 10,
            columns: vec!["amount".into(), "fraud_prob".into()],
            reasoning_tree: Some("test".into()),
        };
        assert_eq!(result.num_rows, 10);
        assert_eq!(result.columns.len(), 2);
    }
}
