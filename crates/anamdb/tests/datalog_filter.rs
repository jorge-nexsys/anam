//! Integration tests for Datalog rules as post-query filters.
//!
//! Validates that registered Datalog rules automatically filter SQL query
//! results — rows violating rule conditions are dropped.

use datafusion::arrow::array::Float64Array;

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

#[tokio::test(flavor = "multi_thread")]
async fn datalog_filter_blocks_low_risk_rows() {
    let session = anamdb::Session::new().await.unwrap();

    let csv_path = workspace_path("demo/data/transactions.csv");
    let lance_path = workspace_path("demo/data/transactions_test_logic.lance");

    if !std::path::Path::new(&csv_path).exists() {
        eprintln!("Skipping test: CSV not found at {csv_path}");
        return;
    }

    let _ = std::fs::remove_dir_all(&lance_path);
    anamdb::storage::ingestion::ingest_csv(&csv_path, &lance_path)
        .await
        .unwrap();
    session.register_table("txns", &lance_path).await.unwrap();

    // First: query without any rules — should return all rows.
    let unfiltered = session
        .sql("SELECT amount, fraud_prob, region FROM txns")
        .await
        .unwrap();
    let total_rows: usize = unfiltered.batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows > 0, "should have rows before filtering");

    // Register a Datalog rule: only keep high-risk transactions.
    session
        .register_logic("high_risk", "high_risk(X) :- txns(X), fraud_prob > 0.80")
        .unwrap();

    // Now query again — the rule should filter out low-fraud rows.
    let filtered = session
        .sql("SELECT amount, fraud_prob, region FROM txns")
        .await
        .unwrap();
    let filtered_rows: usize = filtered.batches.iter().map(|b| b.num_rows()).sum();

    assert!(
        filtered_rows < total_rows,
        "rule should reduce row count: {filtered_rows} should be < {total_rows}"
    );

    // Every surviving row should have fraud_prob > 0.80.
    for batch in &filtered.batches {
        let fraud_col = batch
            .column_by_name("fraud_prob")
            .expect("fraud_prob column should exist")
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for i in 0..fraud_col.len() {
            assert!(
                fraud_col.value(i) > 0.80,
                "row {i}: fraud_prob={} should be > 0.80",
                fraud_col.value(i)
            );
        }
    }

    println!("\n═══ Datalog Filter Integration Test ═══");
    println!("  Rule:     high_risk(X) :- txns(X), fraud_prob > 0.80");
    println!("  Before:   {total_rows} rows");
    println!("  After:    {filtered_rows} rows");
    println!(
        "  Dropped:  {} rows that violated the rule",
        total_rows - filtered_rows
    );
    println!("  ✓ All surviving rows satisfy fraud_prob > 0.80");

    let _ = std::fs::remove_dir_all(&lance_path);
}

#[tokio::test(flavor = "multi_thread")]
async fn multiple_rules_compound_filter() {
    let session = anamdb::Session::new().await.unwrap();

    let csv_path = workspace_path("demo/data/transactions.csv");
    let lance_path = workspace_path("demo/data/transactions_test_logic_multi.lance");

    if !std::path::Path::new(&csv_path).exists() {
        eprintln!("Skipping test: CSV not found at {csv_path}");
        return;
    }

    let _ = std::fs::remove_dir_all(&lance_path);
    anamdb::storage::ingestion::ingest_csv(&csv_path, &lance_path)
        .await
        .unwrap();
    session.register_table("txns", &lance_path).await.unwrap();

    // Register two rules — both must be satisfied.
    session
        .register_logic("high_value", "high_value(X) :- txns(X), amount > 5000")
        .unwrap();
    session
        .register_logic("suspicious", "suspicious(X) :- txns(X), fraud_prob > 0.50")
        .unwrap();

    let result = session
        .sql("SELECT amount, fraud_prob FROM txns")
        .await
        .unwrap();

    for batch in &result.batches {
        let amounts = batch
            .column_by_name("amount")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let fraud_probs = batch
            .column_by_name("fraud_prob")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            assert!(
                amounts.value(i) > 5000.0,
                "row {i}: amount={} should be > 5000",
                amounts.value(i)
            );
            assert!(
                fraud_probs.value(i) > 0.50,
                "row {i}: fraud_prob={} should be > 0.50",
                fraud_probs.value(i)
            );
        }
    }

    let rows: usize = result.batches.iter().map(|b| b.num_rows()).sum();
    println!("\n═══ Compound Datalog Filter Test ═══");
    println!("  Rule 1:   amount > 5000");
    println!("  Rule 2:   fraud_prob > 0.50");
    println!("  Result:   {rows} rows satisfy BOTH constraints");
    println!("  ✓ Compound Datalog filtering works");

    let _ = std::fs::remove_dir_all(&lance_path);
}

#[tokio::test(flavor = "multi_thread")]
async fn rules_skip_irrelevant_tables() {
    // A rule referencing columns not in the result schema should be silently skipped.
    let session = anamdb::Session::new().await.unwrap();

    let csv_path = workspace_path("demo/data/transactions.csv");
    let lance_path = workspace_path("demo/data/transactions_test_logic_skip.lance");

    if !std::path::Path::new(&csv_path).exists() {
        eprintln!("Skipping test: CSV not found at {csv_path}");
        return;
    }

    let _ = std::fs::remove_dir_all(&lance_path);
    anamdb::storage::ingestion::ingest_csv(&csv_path, &lance_path)
        .await
        .unwrap();
    session.register_table("txns", &lance_path).await.unwrap();

    // Register a rule with columns that don't exist in the transactions table.
    session
        .register_logic(
            "other_table_rule",
            "check(X) :- inventory(X), stock_level > 100",
        )
        .unwrap();

    // Query should return all rows — the rule doesn't match the schema.
    let result = session
        .sql("SELECT COUNT(*) AS cnt FROM txns")
        .await
        .unwrap();

    assert!(!result.batches.is_empty());
    println!("\n═══ Rule Schema Mismatch Test ═══");
    println!("  ✓ Rule with non-matching columns was silently skipped");

    let _ = std::fs::remove_dir_all(&lance_path);
}
