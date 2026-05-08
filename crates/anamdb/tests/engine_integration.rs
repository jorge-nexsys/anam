//! Integration tests for engine integration:
//! - Provenance attachment
//! - Streaming Lance provider
//! - Write path (INSERT / DELETE)
//! - Persistent catalog

use datafusion::arrow::array::{Array, BinaryArray, Float64Array};
use datafusion::arrow::datatypes::DataType;

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

// ── Provenance Attachment ─────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn provenance_column_attached() {
    let session = anamdb::Session::new().await.unwrap();

    let csv_path = workspace_path("demo/data/transactions.csv");
    let lance_path = workspace_path("demo/data/test_prov.lance");

    if !std::path::Path::new(&csv_path).exists() {
        eprintln!("Skipping: CSV not found");
        return;
    }

    let _ = std::fs::remove_dir_all(&lance_path);
    anamdb::storage::ingestion::ingest_csv(&csv_path, &lance_path)
        .await
        .unwrap();
    session.register_table("txns", &lance_path).await.unwrap();

    let result = session
        .sql("SELECT amount, fraud_prob FROM txns LIMIT 5")
        .await
        .unwrap();

    assert!(!result.batches.is_empty());
    let batch = &result.batches[0];

    // Provenance column should be attached.
    let prov_col = batch
        .column_by_name("provenance")
        .expect("provenance column should exist");
    assert_eq!(prov_col.data_type(), &DataType::Binary);

    let prov_arr = prov_col
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("provenance should be BinaryArray");

    // Every row should have non-empty provenance bytes.
    for i in 0..prov_arr.len() {
        assert!(
            !prov_arr.value(i).is_empty(),
            "row {i} should have provenance data"
        );
    }

    // Reasoning tree should be present (default mode is Polynomial).
    assert!(
        result.reasoning_tree.is_some(),
        "reasoning tree should exist for Polynomial mode"
    );

    println!("\n═══ Provenance Attachment Test ═══");
    println!("  Rows:       {}", batch.num_rows());
    println!("  Prov bytes: {} per row (avg)", prov_arr.value(0).len());
    println!(
        "  Tree:       {} chars",
        result.reasoning_tree.as_ref().unwrap().len()
    );
    println!("  ✓ Provenance column attached to every query result");

    let _ = std::fs::remove_dir_all(&lance_path);
}

// ── Streaming Lance Provider ──────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn streaming_provider_query() {
    // Verify that the streaming provider works for basic queries.
    let session = anamdb::Session::new().await.unwrap();

    let csv_path = workspace_path("demo/data/transactions.csv");
    let lance_path = workspace_path("demo/data/test_streaming.lance");

    if !std::path::Path::new(&csv_path).exists() {
        eprintln!("Skipping: CSV not found");
        return;
    }

    let _ = std::fs::remove_dir_all(&lance_path);
    anamdb::storage::ingestion::ingest_csv(&csv_path, &lance_path)
        .await
        .unwrap();
    session.register_table("txns", &lance_path).await.unwrap();

    // Basic SELECT.
    let result = session
        .sql("SELECT amount FROM txns WHERE amount > 10000")
        .await
        .unwrap();

    let total_rows: usize = result.batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows > 0, "should have high-value transactions");

    // Verify all values satisfy the filter.
    for batch in &result.batches {
        let amounts = batch
            .column_by_name("amount")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for i in 0..amounts.len() {
            assert!(amounts.value(i) > 10000.0);
        }
    }

    // Aggregation.
    let agg = session
        .sql("SELECT COUNT(*) AS cnt, AVG(amount) AS avg_amt FROM txns")
        .await
        .unwrap();
    assert!(!agg.batches.is_empty());

    println!("\n═══ Streaming Provider Test ═══");
    println!("  Filtered rows: {total_rows} (amount > 10000)");
    println!("  Aggregation:   OK");
    println!("  ✓ Streaming Lance provider works with filters and aggregations");

    let _ = std::fs::remove_dir_all(&lance_path);
}

// ── Write Path (INSERT / DELETE) ──────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn write_path_insert_delete() {
    use datafusion::arrow::array::{
        BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
    };
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    let csv_path = workspace_path("demo/data/transactions.csv");
    let lance_path = workspace_path("demo/data/test_write.lance");

    if !std::path::Path::new(&csv_path).exists() {
        eprintln!("Skipping: CSV not found");
        return;
    }

    // Ingest initial data.
    let _ = std::fs::remove_dir_all(&lance_path);
    anamdb::storage::ingestion::ingest_csv(&csv_path, &lance_path)
        .await
        .unwrap();

    // Count initial rows.
    let session = anamdb::Session::new().await.unwrap();
    session.register_table("txns", &lance_path).await.unwrap();

    let before = session
        .sql("SELECT COUNT(*) AS cnt FROM txns")
        .await
        .unwrap();
    let before_batch = &before.batches[0];
    let initial_count = before_batch
        .column_by_name("cnt")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0) as usize;

    // INSERT new rows.
    // Build a batch matching the CSV schema.
    let schema = Arc::new(Schema::new(vec![
        Field::new("transaction_id", DataType::Utf8, false),
        Field::new("amount", DataType::Float64, false),
        Field::new("fraud_prob", DataType::Float64, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("merchant_type", DataType::Utf8, false),
        Field::new("customer_age", DataType::Int64, false),
        Field::new("is_international", DataType::Boolean, false),
        Field::new("hour_of_day", DataType::Int64, false),
    ]));

    let new_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["INS-001", "INS-002"])),
            Arc::new(Float64Array::from(vec![99999.0, 88888.0])),
            Arc::new(Float64Array::from(vec![0.99, 0.98])),
            Arc::new(StringArray::from(vec!["TEST", "TEST"])),
            Arc::new(StringArray::from(vec!["test", "test"])),
            Arc::new(Int64Array::from(vec![30, 25])),
            Arc::new(BooleanArray::from(vec![true, false])),
            Arc::new(Int64Array::from(vec![1, 2])),
        ],
    )
    .unwrap();

    let insert_result = session
        .insert("txns", &lance_path, vec![new_batch], schema)
        .await
        .unwrap();
    assert_eq!(insert_result.rows_affected, 2);

    // DELETE rows matching a predicate.
    let delete_result = session
        .delete("txns", &lance_path, "region = 'TEST'")
        .await
        .unwrap();
    assert_eq!(delete_result.rows_affected, 2);

    println!("\n═══ Write Path Test ═══");
    println!("  Initial:  {initial_count} rows");
    println!(
        "  INSERT:   {} rows (version {})",
        insert_result.rows_affected, insert_result.new_version
    );
    println!(
        "  DELETE:   {} rows (version {})",
        delete_result.rows_affected, delete_result.new_version
    );
    println!("  ✓ INSERT and DELETE work via Lance APIs");

    let _ = std::fs::remove_dir_all(&lance_path);
}

// ── Persistent Catalog ────────────────────────────────────────────────

#[test]
fn persistent_catalog_survives_restart() {
    use anamdb::storage::catalog::{CatalogStore, ModelEntry};

    let dir = tempfile::tempdir().unwrap();
    let cat_path = dir.path().join("test_catalog.json");
    let cat_str = cat_path.to_str().unwrap();

    // Session 1: populate catalog.
    {
        let mut store = CatalogStore::open(cat_str).unwrap();
        store
            .register_table("transactions", "/data/txns.lance")
            .unwrap();
        store
            .register_rule("high_risk", "high_risk(X) :- txns(X), fraud_prob > 0.80")
            .unwrap();
        store
            .register_model(ModelEntry {
                name: "fraud_detector".into(),
                version: "1.0.0".into(),
                artifact_path: "models/fraud.onnx".into(),
                function_id: "fraud_detector".into(),
                num_features: 3,
                avg_latency_ms: 5.0,
                accuracy: 0.95,
            })
            .unwrap();
    }
    // Store dropped — simulates process exit.

    // Session 2: re-open and verify everything persisted.
    {
        let store = CatalogStore::open(cat_str).unwrap();
        assert_eq!(store.list_tables().len(), 1);
        assert_eq!(store.list_tables()[0].name, "transactions");
        assert_eq!(store.list_tables()[0].lance_path, "/data/txns.lance");

        assert_eq!(store.list_rules().len(), 1);
        assert_eq!(store.list_rules()[0].name, "high_risk");

        assert_eq!(store.list_models().len(), 1);
        assert_eq!(store.list_models()[0].name, "fraud_detector");
        assert_eq!(store.list_models()[0].accuracy, 0.95);
    }

    println!("\n═══ Persistent Catalog Test ═══");
    println!("  Tables:   1 survived restart");
    println!("  Rules:    1 survived restart");
    println!("  Models:   1 survived restart");
    println!("  ✓ Catalog persists across process restarts");
}
