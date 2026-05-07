//! Quick performance test (runs in dev mode, no criterion required).
//! Run with: cargo test --lib bench_quick -- --nocapture

#[cfg(test)]
mod bench_quick {
    use std::sync::Arc;
    use std::time::Instant;

    use datafusion::arrow::array::{Float64Array, RecordBatch, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};

    use crate::core::provenance::{
        BoolSemiring, PolynomialSemiring, ProbSemiring, ProvenanceMode, ProvenanceToken, Semiring,
    };
    use crate::logic::engine::LogicEngine;

    fn generate_batch(n: usize) -> (RecordBatch, Arc<Schema>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("fraud_prob", DataType::Float64, false),
            Field::new("amount", DataType::Float64, false),
            Field::new("region", DataType::Utf8, false),
        ]));

        let fraud_probs: Vec<f64> = (0..n).map(|i| (i as f64 % 100.0) / 100.0).collect();
        let amounts: Vec<f64> = (0..n).map(|i| (i as f64) * 100.0).collect();
        let regions: Vec<&str> = (0..n)
            .map(|i| if i % 3 == 0 { "EU" } else { "US" })
            .collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(fraud_probs)),
                Arc::new(Float64Array::from(amounts)),
                Arc::new(StringArray::from(regions)),
            ],
        )
        .unwrap();

        (batch, schema)
    }

    #[test]
    fn bench_semiring_throughput() {
        println!("\n═══ Semiring Throughput Benchmark ═══");
        let n = 10_000;

        // Boolean semiring
        let start = Instant::now();
        let mut acc = BoolSemiring::zero();
        for _ in 0..n {
            acc = acc.add(&BoolSemiring::one());
        }
        let dt = start.elapsed();
        println!(
            "  BoolSemiring:   {:>7} ops in {:>6.2?}  ({:.1} M ops/sec)",
            n,
            dt,
            n as f64 / dt.as_secs_f64() / 1e6
        );

        // Probability semiring
        let start = Instant::now();
        let mut acc = ProbSemiring::zero();
        for i in 0..n {
            acc = acc.add(&ProbSemiring(0.001 * i as f64));
        }
        let dt = start.elapsed();
        println!(
            "  ProbSemiring:   {:>7} ops in {:>6.2?}  ({:.1} M ops/sec)",
            n,
            dt,
            n as f64 / dt.as_secs_f64() / 1e6
        );

        // Polynomial semiring
        let start = Instant::now();
        let mut acc = PolynomialSemiring::zero();
        for i in 0..1000 {
            let token = ProvenanceToken {
                model_ver_id: "model_v1".into(),
                func_id: "func_1".into(),
                source_record_ids: vec![format!("row_{i}")],
            };
            acc = acc.add(&PolynomialSemiring::singleton(token));
        }
        let dt = start.elapsed();
        println!(
            "  PolySemiring:   {:>7} ops in {:>6.2?}  ({:.1} K ops/sec)",
            1000,
            dt,
            1000.0 / dt.as_secs_f64() / 1e3
        );

        // Polynomial serde roundtrip
        let token = ProvenanceToken {
            model_ver_id: "model_v1".into(),
            func_id: "func_1".into(),
            source_record_ids: vec!["row_0".into(), "row_1".into()],
        };
        let poly = PolynomialSemiring::singleton(token);
        let start = Instant::now();
        for _ in 0..n {
            let bytes = poly.to_bytes().unwrap();
            let _ = PolynomialSemiring::from_bytes(&bytes).unwrap();
        }
        let dt = start.elapsed();
        println!(
            "  Poly serde:     {:>7} ops in {:>6.2?}  ({:.1} K ops/sec)",
            n,
            dt,
            n as f64 / dt.as_secs_f64() / 1e3
        );
    }

    #[test]
    fn bench_logic_engine_throughput() {
        println!("\n═══ Logic Engine Filter Benchmark ═══");

        for n in [100, 1_000, 10_000] {
            let (batch, _schema) = generate_batch(n);

            let mut engine = LogicEngine::new(ProvenanceMode::Boolean).unwrap();
            engine
                .register_rule(
                    "high_risk",
                    "fraud_prob > 0.90 AND amount > 10000 AND region = 'EU'",
                )
                .unwrap();
            engine.add_facts("fraud_prob", vec![batch]).unwrap();

            // Warmup
            let _ = engine.evaluate("high_risk").unwrap();

            // Timed run
            let iterations = 100;
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = engine.evaluate("high_risk").unwrap();
            }
            let dt = start.elapsed();
            let per_iter = dt / iterations;

            println!(
                "  Filter {n:>6} rows: {:>6.2?}/eval  ({:.1} evals/sec)",
                per_iter,
                iterations as f64 / dt.as_secs_f64()
            );
        }
    }

    #[test]
    fn bench_hitl_monitor() {
        use crate::hitl::monitor::SemanticMonitor;

        println!("\n═══ HITL Monitor Benchmark ═══");

        let monitor = SemanticMonitor::new(0.5);

        for n in [100, 1_000, 10_000] {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "fraud_prob",
                DataType::Float64,
                false,
            )]));
            let values: Vec<f64> = (0..n).map(|i| (i as f64 % 100.0) / 100.0).collect();
            let batch =
                RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap();

            let iterations = 1000;
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = monitor.inspect_batches(&[batch.clone()]).unwrap();
            }
            let dt = start.elapsed();
            let per_iter = dt / iterations;

            println!(
                "  Monitor {n:>6} rows: {:>6.2?}/scan  ({:.1} scans/sec)",
                per_iter,
                iterations as f64 / dt.as_secs_f64()
            );
        }
    }
}
