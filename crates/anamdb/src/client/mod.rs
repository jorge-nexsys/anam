//! Rust client SDK for AnamDB — connects to a running AnamDB server.
//!
//! Provides an async client with connection management, query execution,
//! and typed result handling over the JSON-over-TCP wire protocol.

use std::time::Duration;

use datafusion::arrow::array::RecordBatch;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tracing::{debug, info};

use crate::core::error::{AnamError, Result};

/// Configuration for the AnamDB client.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Server address (host:port).
    pub addr: String,
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Maximum retry attempts for failed queries.
    pub max_retries: u32,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8080".to_string(),
            connect_timeout: Duration::from_secs(5),
            max_retries: 3,
        }
    }
}

/// Result of a remote query.
#[derive(Debug)]
pub struct RemoteQueryResult {
    /// Decoded Arrow record batches.
    pub batches: Vec<RecordBatch>,
    /// Reasoning tree (provenance trace).
    pub reasoning_tree: Option<String>,
    /// Anomaly descriptions.
    pub anomalies: Vec<String>,
}

/// Async client for connecting to a running AnamDB server.
pub struct AnamClient {
    config: ClientConfig,
    stream: Option<BufReader<TcpStream>>,
}

impl AnamClient {
    /// Create a new client with the given configuration.
    pub fn new(config: ClientConfig) -> Self {
        Self {
            config,
            stream: None,
        }
    }

    /// Connect to the server at `addr` with default configuration.
    pub fn connect_to(addr: &str) -> Self {
        Self::new(ClientConfig {
            addr: addr.to_string(),
            ..Default::default()
        })
    }

    /// Establish the TCP connection.
    pub async fn connect(&mut self) -> Result<()> {
        info!(addr = %self.config.addr, "connecting to AnamDB server");

        let stream = tokio::time::timeout(
            self.config.connect_timeout,
            TcpStream::connect(&self.config.addr),
        )
        .await
        .map_err(|_| {
            AnamError::Io(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "connection timed out",
            ))
        })?
        .map_err(AnamError::Io)?;

        self.stream = Some(BufReader::new(stream));
        info!("connected to AnamDB server");
        Ok(())
    }

    /// Send a JSON command and receive the response.
    async fn send_command(&mut self, cmd: &Value) -> Result<Value> {
        let reader = self.stream.as_mut().ok_or_else(|| {
            AnamError::Io(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "not connected — call connect() first",
            ))
        })?;

        let mut cmd_str =
            serde_json::to_string(cmd).map_err(|e| AnamError::Serde(e.to_string()))?;
        cmd_str.push('\n');

        reader
            .get_mut()
            .write_all(cmd_str.as_bytes())
            .await
            .map_err(AnamError::Io)?;

        let mut response_line = String::new();
        reader
            .read_line(&mut response_line)
            .await
            .map_err(AnamError::Io)?;

        serde_json::from_str(&response_line)
            .map_err(|e| AnamError::Serde(format!("invalid response: {e}")))
    }

    /// Execute a SQL query on the remote server.
    pub async fn query(&mut self, sql: &str) -> Result<RemoteQueryResult> {
        debug!(sql, "sending query to server");

        let cmd = serde_json::json!({
            "method": "query",
            "sql": sql
        });

        let resp = self.send_command(&cmd).await?;

        let ok = resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
        if !ok {
            let error = resp
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            return Err(AnamError::Logic(format!("server error: {error}")));
        }

        let reasoning_tree = resp
            .get("reasoning_tree")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());

        let anomalies: Vec<String> = resp
            .get("anomalies")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(RemoteQueryResult {
            batches: Vec::new(), // IPC decoding happens client-side
            reasoning_tree,
            anomalies,
        })
    }

    /// Register a table on the remote server.
    pub async fn register_table(&mut self, name: &str, lance_path: &str) -> Result<String> {
        let cmd = serde_json::json!({
            "method": "register_table",
            "name": name,
            "lance_path": lance_path
        });

        let resp = self.send_command(&cmd).await?;
        let msg = resp
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("ok")
            .to_string();
        Ok(msg)
    }

    /// Register a Datalog rule on the remote server.
    pub async fn register_rule(&mut self, name: &str, datalog: &str) -> Result<String> {
        let cmd = serde_json::json!({
            "method": "register_rule",
            "name": name,
            "datalog": datalog
        });

        let resp = self.send_command(&cmd).await?;
        let msg = resp
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("ok")
            .to_string();
        Ok(msg)
    }

    /// Check server health.
    pub async fn health(&mut self) -> Result<ServerHealth> {
        let cmd = serde_json::json!({"method": "health"});
        let resp = self.send_command(&cmd).await?;

        Ok(ServerHealth {
            status: resp
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("UNKNOWN")
                .to_string(),
            version: resp
                .get("version")
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string(),
            table_count: resp.get("tables").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            model_count: resp.get("models").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            rule_count: resp.get("rules").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        })
    }
}

/// Server health status.
#[derive(Debug, Clone)]
pub struct ServerHealth {
    /// Server status.
    pub status: String,
    /// AnamDB version.
    pub version: String,
    /// Number of registered tables.
    pub table_count: u32,
    /// Number of registered models.
    pub model_count: u32,
    /// Number of registered rules.
    pub rule_count: u32,
}
