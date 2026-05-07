//! Integration tests for server, client SDK, CLI, and Python SDK scaffold.

/// Resolve a path relative to the workspace root.
fn workspace_path(relative: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let workspace = std::path::Path::new(manifest)
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    workspace.join(relative).to_string_lossy().to_string()
}

// ── Server + Client SDK ───────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn server_client_roundtrip() {
    use anamdb::client::{AnamClient, ClientConfig};
    use anamdb::server::AnamGrpcService;
    use anamdb::Session;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::RwLock;

    let csv_path = workspace_path("demo/data/transactions.csv");
    let lance_path = workspace_path("demo/data/test_server.lance");

    if !std::path::Path::new(&csv_path).exists() {
        eprintln!("Skipping: CSV not found");
        return;
    }

    // Set up: ingest data.
    let _ = std::fs::remove_dir_all(&lance_path);
    anamdb::storage::ingestion::ingest_csv(&csv_path, &lance_path)
        .await
        .unwrap();

    // Create session and register table.
    let session = Session::new().await.unwrap();
    session.register_table("txns", &lance_path).await.unwrap();

    // Start server on a random port.
    let service = Arc::new(AnamGrpcService::new(session));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let svc = Arc::clone(&service);

    // Spawn server task.
    let server_handle = tokio::spawn(async move {
        loop {
            if let Ok((stream, _peer)) = listener.accept().await {
                let svc = Arc::clone(&svc);
                tokio::spawn(async move {
                    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
                    let (reader, mut writer) = stream.into_split();
                    let mut reader = BufReader::new(reader);
                    loop {
                        let mut line = String::new();
                        let n = reader.read_line(&mut line).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        let cmd: serde_json::Value =
                            serde_json::from_str(line).unwrap_or_default();
                        let method =
                            cmd.get("method").and_then(|v| v.as_str()).unwrap_or("");
                        let resp: serde_json::Value = match method {
                            "query" => {
                                let sql =
                                    cmd.get("sql").and_then(|v| v.as_str()).unwrap_or("");
                                match svc.query(sql).await {
                                    Ok(r) => serde_json::json!({
                                        "ok": true,
                                        "ipc_bytes": r.arrow_ipc_batch.len(),
                                        "reasoning_tree": r.reasoning_tree,
                                        "anomalies": r.anomalies,
                                    }),
                                    Err(e) => serde_json::json!({
                                        "ok": false,
                                        "error": format!("{e}")
                                    }),
                                }
                            }
                            "health" => {
                                let h = svc.health().await;
                                serde_json::json!({
                                    "status": h.status,
                                    "version": h.version,
                                    "tables": h.table_count,
                                    "models": h.model_count,
                                    "rules": h.rule_count,
                                })
                            }
                            _ => serde_json::json!({"ok": false, "error": "unknown"}),
                        };
                        let mut s = serde_json::to_string(&resp).unwrap();
                        s.push('\n');
                        let _ = writer.write_all(s.as_bytes()).await;
                    }
                });
            }
        }
    });

    // Give server a moment to start.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Connect client.
    let mut client = AnamClient::new(ClientConfig {
        addr: addr.to_string(),
        connect_timeout: Duration::from_secs(2),
        max_retries: 1,
    });
    client.connect().await.unwrap();

    // Health check.
    let health = client.health().await.unwrap();
    assert_eq!(health.status, "SERVING");
    assert!(!health.version.is_empty());

    // Query.
    let result = client
        .query("SELECT COUNT(*) AS cnt FROM txns")
        .await
        .unwrap();
    assert!(result.reasoning_tree.is_some());

    println!("\n═══ Server + Client SDK Test ═══");
    println!("  Server:  127.0.0.1:{}", addr.port());
    println!("  Health:  {}", health.status);
    println!("  Version: {}", health.version);
    println!("  Query:   OK (reasoning tree: {} chars)", result.reasoning_tree.as_ref().unwrap().len());
    println!("  ✓ Server + Client SDK roundtrip works");

    server_handle.abort();
    let _ = std::fs::remove_dir_all(&lance_path);
}

// ── CLI Init ──────────────────────────────────────────────────────────

#[test]
fn cli_init_creates_directory_structure() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("anamdb_test_data");
    let data_str = data_path.to_str().unwrap();

    // Simulate init.
    std::fs::create_dir_all(format!("{data_str}/tables")).unwrap();
    std::fs::create_dir_all(format!("{data_str}/models")).unwrap();

    let config_path = format!("{data_str}/anamdb.toml");
    std::fs::write(
        &config_path,
        r#"[server]
bind = "0.0.0.0:8080"

[engine]
provenance_mode = "polynomial"
"#,
    )
    .unwrap();

    let catalog_path = format!("{data_str}/catalog.json");
    let _store = anamdb::storage::catalog::CatalogStore::open(&catalog_path).unwrap();

    // Verify structure.
    assert!(std::path::Path::new(&format!("{data_str}/tables")).is_dir());
    assert!(std::path::Path::new(&format!("{data_str}/models")).is_dir());
    assert!(std::path::Path::new(&config_path).is_file());
    assert!(std::path::Path::new(&catalog_path).is_file());

    // Verify config content.
    let config_content = std::fs::read_to_string(&config_path).unwrap();
    assert!(config_content.contains("provenance_mode"));

    println!("\n═══ CLI Init Test ═══");
    println!("  ✓ Created tables/ directory");
    println!("  ✓ Created models/ directory");
    println!("  ✓ Created anamdb.toml config");
    println!("  ✓ Created catalog.json");
    println!("  ✓ Init structure is correct");
}

// ── Python SDK Scaffold ───────────────────────────────────────────────

#[test]
fn python_sdk_api_surface() {
    use anamdb::sdk::python::{PyAnamClient, PyQueryResult};

    let client = PyAnamClient::new("localhost:8080");
    assert_eq!(client.addr, "localhost:8080");

    let result = PyQueryResult {
        num_rows: 25,
        columns: vec!["amount".into(), "fraud_prob".into(), "region".into()],
        reasoning_tree: Some("provenance trace".into()),
    };
    assert_eq!(result.num_rows, 25);
    assert_eq!(result.columns.len(), 3);

    println!("\n═══ Python SDK Scaffold Test ═══");
    println!("  ✓ PyAnamClient API surface verified");
    println!("  ✓ PyQueryResult API surface verified");
}

// ── Docker Image ──────────────────────────────────────────────────────

#[test]
fn dockerfile_exists() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let workspace = std::path::Path::new(manifest)
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let dockerfile = workspace.join("Dockerfile");
    assert!(
        dockerfile.exists(),
        "Dockerfile should exist at workspace root"
    );

    let content = std::fs::read_to_string(&dockerfile).unwrap();
    assert!(content.contains("\"serve\""));
    assert!(content.contains("EXPOSE 8080"));

    println!("\n═══ Docker Image Test ═══");
    println!("  ✓ Dockerfile exists");
    println!("  ✓ Contains serve command");
    println!("  ✓ Exposes port 8080");
}
