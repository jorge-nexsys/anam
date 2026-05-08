//! AnamDB server — production wire protocol.
//!
//! Provides streaming SQL query execution, table/rule/model registration,
//! and health checks over a JSON-over-TCP protocol. Designed for upgrade
//! to full gRPC/tonic once tonic-build is integrated.

pub mod auth;
pub mod rate_limit;
pub mod metering;


use std::sync::Arc;

use datafusion::arrow::ipc::writer::StreamWriter;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::core::error::AnamError;
use crate::core::session::{Session, SessionConfig};

/// Protocol message types for the wire protocol.
pub mod proto {
    /// Query request.
    #[derive(Clone, Debug)]
    pub struct QueryRequest {
        /// SQL query string.
        pub sql: String,
    }

    /// A single streaming query response chunk.
    #[derive(Clone, Debug)]
    pub struct QueryResponse {
        /// Arrow IPC-encoded RecordBatch bytes.
        pub arrow_ipc_batch: Vec<u8>,
        /// Arrow IPC-encoded schema bytes.
        pub arrow_ipc_schema: Vec<u8>,
        /// Provenance reasoning tree.
        pub reasoning_tree: String,
        /// Anomaly descriptions.
        pub anomalies: Vec<String>,
    }

    /// Register table request.
    #[derive(Clone, Debug)]
    pub struct RegisterTableRequest {
        /// Logical table name.
        pub name: String,
        /// Filesystem path to Lance dataset.
        pub lance_path: String,
    }

    /// Register table response.
    #[derive(Clone, Debug)]
    pub struct RegisterTableResponse {
        /// Whether the operation succeeded.
        pub success: bool,
        /// Status message.
        pub message: String,
    }

    /// Register rule request.
    #[derive(Clone, Debug)]
    pub struct RegisterRuleRequest {
        /// Rule name.
        pub name: String,
        /// Datalog expression.
        pub datalog: String,
    }

    /// Register rule response.
    #[derive(Clone, Debug)]
    pub struct RegisterRuleResponse {
        /// Whether the operation succeeded.
        pub success: bool,
        /// Status message.
        pub message: String,
    }

    /// Load model request.
    #[derive(Clone, Debug)]
    pub struct LoadModelRequest {
        /// Model name (becomes the SQL function name).
        pub name: String,
        /// Model version string.
        pub version: String,
        /// Path to the ONNX model file.
        pub model_path: String,
        /// SQL function ID.
        pub function_id: String,
        /// Number of input features.
        pub num_features: u32,
        /// Average latency in milliseconds.
        pub avg_latency_ms: f64,
        /// Model accuracy.
        pub accuracy: f64,
    }

    /// Load model response.
    #[derive(Clone, Debug)]
    pub struct LoadModelResponse {
        /// Whether the operation succeeded.
        pub success: bool,
        /// Unique model ID assigned by the registry.
        pub model_id: String,
        /// Status message.
        pub message: String,
    }

    /// Health request.
    #[derive(Clone, Debug)]
    pub struct HealthRequest;

    /// Health response.
    #[derive(Clone, Debug)]
    pub struct HealthResponse {
        /// "SERVING" or "NOT_SERVING".
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
}

/// The AnamDB service implementation.
pub struct AnamGrpcService {
    session: Arc<RwLock<Session>>,
    pub authenticator: Arc<dyn auth::Authenticator>,
    pub rate_limiter: Arc<rate_limit::RateLimiter>,
    pub metering: Arc<metering::MeteringSystem>,
}

impl AnamGrpcService {
    /// Create a new service wrapping a session.
    pub fn new(session: Session) -> Self {
        Self {
            session: Arc::new(RwLock::new(session)),
            authenticator: Arc::new(auth::DummyAuthenticator),
            rate_limiter: Arc::new(rate_limit::RateLimiter::new()),
            metering: Arc::new(metering::MeteringSystem::new()),
        }
    }

    /// Create a new service with a shared session.
    pub fn with_shared_session(session: Arc<RwLock<Session>>) -> Self {
        Self { 
            session,
            authenticator: Arc::new(auth::DummyAuthenticator),
            rate_limiter: Arc::new(rate_limit::RateLimiter::new()),
            metering: Arc::new(metering::MeteringSystem::new()),
        }
    }

    /// Execute a SQL query and return results as Arrow IPC bytes.
    pub async fn query(&self, sql: &str) -> crate::core::error::Result<proto::QueryResponse> {
        let session = self.session.read().await;
        let result = session.sql(sql).await?;

        // Serialize batches to Arrow IPC format.
        let mut ipc_bytes = Vec::new();
        let mut schema_bytes = Vec::new();

        if !result.batches.is_empty() {
            let schema = result.batches[0].schema();

            // Serialize schema.
            let mut schema_writer = StreamWriter::try_new(&mut schema_bytes, &schema)
                .map_err(AnamError::Arrow)?;
            schema_writer.finish().map_err(AnamError::Arrow)?;

            // Serialize all batches into a single IPC stream.
            let mut writer = StreamWriter::try_new(&mut ipc_bytes, &schema)
                .map_err(AnamError::Arrow)?;
            for batch in &result.batches {
                writer.write(batch).map_err(AnamError::Arrow)?;
            }
            writer.finish().map_err(AnamError::Arrow)?;
        }

        let anomalies: Vec<String> = result
            .anomalies
            .iter()
            .map(|a| format!("{:?}", a))
            .collect();

        Ok(proto::QueryResponse {
            arrow_ipc_batch: ipc_bytes,
            arrow_ipc_schema: schema_bytes,
            reasoning_tree: result.reasoning_tree.unwrap_or_default(),
            anomalies,
        })
    }

    /// Register a Lance table.
    pub async fn register_table(
        &self,
        name: &str,
        lance_path: &str,
    ) -> proto::RegisterTableResponse {
        let session = self.session.read().await;
        match session.register_table(name, lance_path).await {
            Ok(()) => proto::RegisterTableResponse {
                success: true,
                message: format!("table '{name}' registered"),
            },
            Err(e) => proto::RegisterTableResponse {
                success: false,
                message: format!("failed: {e}"),
            },
        }
    }

    /// Register a Datalog rule.
    pub async fn register_rule(
        &self,
        name: &str,
        datalog: &str,
    ) -> proto::RegisterRuleResponse {
        let session = self.session.read().await;
        match session.register_logic(name, datalog) {
            Ok(()) => proto::RegisterRuleResponse {
                success: true,
                message: format!("rule '{name}' registered"),
            },
            Err(e) => proto::RegisterRuleResponse {
                success: false,
                message: format!("failed: {e}"),
            },
        }
    }

    /// Load an ONNX model.
    pub async fn load_model(
        &self,
        req: &proto::LoadModelRequest,
    ) -> proto::LoadModelResponse {
        let session = self.session.read().await;
        match session.load_onnx_model_with_metrics(
            &req.name,
            &req.version,
            &req.model_path,
            &req.function_id,
            req.num_features as usize,
            req.avg_latency_ms,
            req.accuracy,
        ) {
            Ok(model_id) => proto::LoadModelResponse {
                success: true,
                model_id,
                message: format!("model '{}' loaded", req.name),
            },
            Err(e) => proto::LoadModelResponse {
                success: false,
                model_id: String::new(),
                message: format!("failed: {e}"),
            },
        }
    }

    /// Health check.
    pub async fn health(&self) -> proto::HealthResponse {
        let session = self.session.read().await;
        let model_count = session.models().list_models().len() as u32;
        let rule_count = session.logic_engine().read().list_rules().len() as u32;

        proto::HealthResponse {
            status: "SERVING".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            table_count: 0,
            model_count,
            rule_count,
        }
    }
}

/// Start the AnamDB server on the specified address.
///
/// Creates a TCP listener that accepts JSON-over-TCP connections.
/// Each line is a JSON command; responses are JSON lines back.
pub async fn serve(addr: &str, config: SessionConfig) -> crate::core::error::Result<()> {
    use tokio::net::TcpListener;

    info!(addr, "starting AnamDB server");

    let session = Session::with_config(config).await?;
    let service = Arc::new(AnamGrpcService::new(session));

    let listener = TcpListener::bind(addr)
        .await
        .map_err(AnamError::Io)?;

    info!(addr, "AnamDB server listening");

    loop {
        let (stream, peer) = listener.accept().await.map_err(AnamError::Io)?;
        let svc = Arc::clone(&service);

        tokio::spawn(async move {
            info!(peer = %peer, "client connected");
            
            // Peek at the first bytes to determine protocol.
            let mut buf = [0; 4];
            match stream.peek(&mut buf).await {
                Ok(n) if n >= 4 && &buf[0..4] == b"GET " => {
                    // WebSocket HTTP upgrade request.
                    if let Err(e) = handle_ws_connection(stream, svc).await {
                        warn!(error = %e, "websocket connection error");
                    }
                }
                Ok(_) => {
                    // Raw JSON-over-TCP.
                    if let Err(e) = handle_connection(stream, svc).await {
                        warn!(error = %e, "tcp connection error");
                    }
                }
                Err(e) => warn!(error = %e, "failed to peek stream"),
            }
        });
    }
}

/// Handle a single WebSocket connection using tokio-tungstenite.
async fn handle_ws_connection(
    stream: tokio::net::TcpStream,
    service: Arc<AnamGrpcService>,
) -> crate::core::error::Result<()> {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let ws_stream = tokio_tungstenite::accept_async(stream)
        .await
        .map_err(|e| AnamError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("ws error: {e}"))))?;
        
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    while let Some(msg) = ws_receiver.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                warn!("websocket error: {e}");
                break;
            }
        };

        if msg.is_close() {
            break;
        }

        if let Message::Text(text) = msg {
            let line = text.trim();
            if line.is_empty() {
                continue;
            }

            // Parse JSON command and dispatch.
            let response: serde_json::Value = match serde_json::from_str::<serde_json::Value>(line) {
                Ok(cmd) => {
                    let method = cmd.get("method").and_then(|v| v.as_str()).unwrap_or("");
                    match method {
                        "query" => {
                            let sql = cmd.get("sql").and_then(|v| v.as_str()).unwrap_or("");
                            match service.query(sql).await {
                                Ok(resp) => serde_json::json!({
                                    "ok": true,
                                    "reasoning_tree": resp.reasoning_tree,
                                    "anomalies": resp.anomalies,
                                    "ipc_bytes": resp.arrow_ipc_batch.len(),
                                }),
                                Err(e) => serde_json::json!({"ok": false, "error": format!("{e}")}),
                            }
                        }
                        "register_table" => {
                            let name = cmd.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let path = cmd.get("lance_path").and_then(|v| v.as_str()).unwrap_or("");
                            let r = service.register_table(name, path).await;
                            serde_json::json!({"ok": r.success, "message": r.message})
                        }
                        "register_rule" => {
                            let name = cmd.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let datalog = cmd.get("datalog").and_then(|v| v.as_str()).unwrap_or("");
                            let r = service.register_rule(name, datalog).await;
                            serde_json::json!({"ok": r.success, "message": r.message})
                        }
                        "health" => {
                            let h = service.health().await;
                            serde_json::json!({
                                "status": h.status,
                                "version": h.version,
                                "tables": h.table_count,
                                "models": h.model_count,
                                "rules": h.rule_count,
                            })
                        }
                        _ => serde_json::json!({"ok": false, "error": format!("unknown method: {method}")}),
                    }
                }
                Err(e) => serde_json::json!({"ok": false, "error": format!("invalid JSON: {e}")}),
            };

            let response_str = serde_json::to_string(&response).unwrap_or_default();
            if let Err(e) = ws_sender.send(Message::Text(response_str.into())).await {
                warn!("failed to send websocket message: {e}");
                break;
            }
        }
    }

    Ok(())
}

/// Handle a single client connection (JSON-over-TCP wire protocol).
async fn handle_connection(
    stream: tokio::net::TcpStream,
    service: Arc<AnamGrpcService>,
) -> crate::core::error::Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);

    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await.map_err(AnamError::Io)?;

        if n == 0 {
            break; // Client disconnected.
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Parse JSON command and dispatch.
        let response: serde_json::Value = match serde_json::from_str::<serde_json::Value>(line) {
            Ok(cmd) => {
                let method = cmd.get("method").and_then(|v| v.as_str()).unwrap_or("");
                match method {
                    "query" => {
                        let sql = cmd.get("sql").and_then(|v| v.as_str()).unwrap_or("");
                        match service.query(sql).await {
                            Ok(resp) => serde_json::json!({
                                "ok": true,
                                "reasoning_tree": resp.reasoning_tree,
                                "anomalies": resp.anomalies,
                                "ipc_bytes": resp.arrow_ipc_batch.len(),
                            }),
                            Err(e) => serde_json::json!({"ok": false, "error": format!("{e}")}),
                        }
                    }
                    "register_table" => {
                        let name = cmd.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let path = cmd.get("lance_path").and_then(|v| v.as_str()).unwrap_or("");
                        let r = service.register_table(name, path).await;
                        serde_json::json!({"ok": r.success, "message": r.message})
                    }
                    "register_rule" => {
                        let name = cmd.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let datalog = cmd.get("datalog").and_then(|v| v.as_str()).unwrap_or("");
                        let r = service.register_rule(name, datalog).await;
                        serde_json::json!({"ok": r.success, "message": r.message})
                    }
                    "health" => {
                        let h = service.health().await;
                        serde_json::json!({
                            "status": h.status,
                            "version": h.version,
                            "tables": h.table_count,
                            "models": h.model_count,
                            "rules": h.rule_count,
                        })
                    }
                    _ => serde_json::json!({"ok": false, "error": format!("unknown method: {method}")}),
                }
            }
            Err(e) => serde_json::json!({"ok": false, "error": format!("invalid JSON: {e}")}),
        };

        let mut response_str = serde_json::to_string(&response).unwrap_or_default();
        response_str.push('\n');
        writer
            .write_all(response_str.as_bytes())
            .await
            .map_err(AnamError::Io)?;
    }

    Ok(())
}
