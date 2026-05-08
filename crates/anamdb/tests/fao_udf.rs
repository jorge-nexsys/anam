//! Integration tests for FAO operators as DataFusion scalar UDFs.
//!
//! Validates that ONNX models registered via Session::load_onnx_model_with_metrics
//! become callable SQL functions inside DataFusion queries.

use datafusion::arrow::array::Float64Array;
use datafusion::arrow::datatypes::DataType;

/// Resolve a path relative to the workspace root (two levels up from this crate).
fn workspace_path(relative: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let workspace = std::path::Path::new(manifest)
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    workspace.join(relative).to_string_lossy().to_string()
}

#[tokio::test(flavor = "multi_thread")]
async fn fao_udf_inline_inference() {
    let session = anamdb::Session::new().await.unwrap();

    // Resolve paths relative to workspace root.
    let csv_path = workspace_path("demo/data/transactions.csv");
    let lance_path = workspace_path("demo/data/transactions_test_udf.lance");
    let model_path = workspace_path("demo/models/fraud_detector.onnx");

    if !std::path::Path::new(&csv_path).exists() {
        eprintln!("Skipping test: CSV not found at {csv_path}");
        return;
    }
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping test: ONNX model not found at {model_path}");
        return;
    }

    // Ingest CSV → Lance, then register the table.
    let _ = std::fs::remove_dir_all(&lance_path);
    anamdb::storage::ingestion::ingest_csv(&csv_path, &lance_path)
        .await
        .unwrap();
    session.register_table("txns", &lance_path).await.unwrap();

    // Load the ONNX model — auto-registers a UDF named "fraud_detector".
    let model_id = session
        .load_onnx_model_with_metrics(
            "fraud_detector",
            "1.0.0",
            &model_path,
            "fraud_detector",
            3,    // 3 input features
            5.0,  // avg latency ms
            0.95, // accuracy
        )
        .unwrap();

    assert!(!model_id.is_empty());

    // Call the model inline in SQL.
    let result = session
        .sql("SELECT amount, fraud_detector(amount, fraud_prob, CAST(hour_of_day AS DOUBLE)) AS score FROM txns ORDER BY amount DESC LIMIT 5")
        .await
        .unwrap();

    assert!(!result.batches.is_empty());
    let batch = &result.batches[0];
    assert!(batch.num_rows() > 0);
    assert!(batch.num_rows() <= 5);

    // Verify the "score" column exists and is Float64.
    let score_col = batch
        .column_by_name("score")
        .expect("'score' column should exist");
    assert_eq!(score_col.data_type(), &DataType::Float64);

    let scores = score_col
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("score column should be Float64Array");

    // All scores should be between 0 and 1 (sigmoid output).
    for i in 0..scores.len() {
        let s = scores.value(i);
        assert!(
            (0.0..=1.0).contains(&s),
            "score {s} at row {i} should be in [0, 1]"
        );
    }

    println!("\n═══ FAO UDF Integration Test: Inline Inference ═══");
    println!("  Model:  fraud_detector (ONNX)");
    println!(
        "  Query:  SELECT amount, fraud_detector(amount, fraud_prob, hour) AS score FROM txns"
    );
    println!("  Rows:   {}", batch.num_rows());
    for i in 0..batch.num_rows().min(5) {
        let amount = batch
            .column_by_name("amount")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(i);
        println!(
            "    row {i}: amount={amount:>10.2}, score={:.4}",
            scores.value(i)
        );
    }
    println!("  ✓ All scores in [0, 1] — ONNX inference inline in SQL works!");

    let _ = std::fs::remove_dir_all(&lance_path);
}

#[tokio::test(flavor = "multi_thread")]
async fn fao_udf_multiple_models() {
    let session = anamdb::Session::new().await.unwrap();

    let csv_path = workspace_path("demo/data/transactions.csv");
    let lance_path = workspace_path("demo/data/transactions_test_udf_multi.lance");
    let model_path = workspace_path("demo/models/fraud_detector.onnx");
    let fast_path = workspace_path("demo/models/fraud_detector_fast.onnx");

    if !std::path::Path::new(&csv_path).exists()
        || !std::path::Path::new(&model_path).exists()
        || !std::path::Path::new(&fast_path).exists()
    {
        eprintln!("Skipping test: required files not found");
        return;
    }

    // Register two models under different function names.
    session
        .load_onnx_model_with_metrics(
            "fraud_detector",
            "1.0.0",
            &model_path,
            "fraud_detector",
            3,
            5.0,
            0.95,
        )
        .unwrap();
    session
        .load_onnx_model_with_metrics(
            "fraud_fast",
            "1.0.0",
            &fast_path,
            "fraud_fast",
            3,
            0.5,
            0.75,
        )
        .unwrap();

    // Ingest and register table.
    let _ = std::fs::remove_dir_all(&lance_path);
    anamdb::storage::ingestion::ingest_csv(&csv_path, &lance_path)
        .await
        .unwrap();
    session.register_table("t", &lance_path).await.unwrap();

    // Call both models in the same query.
    let result = session
        .sql("SELECT fraud_detector(amount, fraud_prob, CAST(hour_of_day AS DOUBLE)) AS accurate, fraud_fast(amount, fraud_prob, CAST(hour_of_day AS DOUBLE)) AS fast FROM t LIMIT 3")
        .await
        .unwrap();

    let batch = &result.batches[0];
    assert!(batch.num_rows() > 0);
    assert!(batch.column_by_name("accurate").is_some());
    assert!(batch.column_by_name("fast").is_some());

    println!("\n═══ FAO UDF Integration Test: Multiple Models ═══");
    println!("  ✓ fraud_detector() and fraud_fast() both callable in same SELECT");

    let _ = std::fs::remove_dir_all(&lance_path);
}
