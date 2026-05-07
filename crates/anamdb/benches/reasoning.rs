//! Benchmark: Reasoning Latency, Proof Trace Overhead, and Throughput.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::sync::Arc;

use datafusion::arrow::array::{Float64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};

use anamdb::core::provenance::{
    BoolSemiring, PolynomialSemiring, ProbSemiring, ProvenanceMode, ProvenanceToken, Semiring,
};
use anamdb::logic::engine::LogicEngine;

/// Benchmark semiring operations.
fn bench_semiring(c: &mut Criterion) {
    let mut group = c.benchmark_group("semiring");

    group.bench_function("bool_add_1000", |b| {
        b.iter(|| {
            let mut acc = BoolSemiring::zero();
            for _ in 0..1000 {
                acc = acc.add(&BoolSemiring::one());
            }
            acc
        });
    });

    group.bench_function("prob_add_1000", |b| {
        b.iter(|| {
            let mut acc = ProbSemiring::zero();
            for i in 0..1000 {
                acc = acc.add(&ProbSemiring(0.001 * i as f64));
            }
            acc
        });
    });

    group.bench_function("poly_add_1000", |b| {
        b.iter(|| {
            let mut acc = PolynomialSemiring::zero();
            for i in 0..1000 {
                let token = ProvenanceToken {
                    model_ver_id: "model_v1".into(),
                    func_id: "func_1".into(),
                    source_record_ids: vec![format!("row_{i}")],
                };
                acc = acc.add(&PolynomialSemiring::singleton(token));
            }
            acc
        });
    });

    group.bench_function("poly_serde_roundtrip", |b| {
        let token = ProvenanceToken {
            model_ver_id: "model_v1".into(),
            func_id: "func_1".into(),
            source_record_ids: vec!["row_0".into(), "row_1".into()],
        };
        let poly = PolynomialSemiring::singleton(token);

        b.iter(|| {
            let bytes = poly.to_bytes().unwrap();
            PolynomialSemiring::from_bytes(&bytes).unwrap()
        });
    });

    group.finish();
}

/// Benchmark logic engine rule evaluation.
fn bench_logic_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("logic_engine");

    for num_rows in [100, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("evaluate_filter", num_rows),
            &num_rows,
            |b, &n| {
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

                let mut engine = LogicEngine::new(ProvenanceMode::Boolean).unwrap();
                engine
                    .register_rule(
                        "high_risk",
                        "fraud_prob > 0.90 AND amount > 10000 AND region = 'EU'",
                    )
                    .unwrap();
                engine.add_facts("fraud_prob", vec![batch]).unwrap();

                b.iter(|| engine.evaluate("high_risk").unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_semiring, bench_logic_engine);
criterion_main!(benches);
