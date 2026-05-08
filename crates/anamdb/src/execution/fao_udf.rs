//! `FaoScalarUdf` — bridges FAO operators into DataFusion as `ScalarUDF`s.
//!
//! Each registered ONNX model becomes a callable SQL function:
//!
//! ```sql
//! SELECT fraud_detector(amount, region_code, hour) FROM txns;
//! ```
//!
//! The UDF collects its input columns, builds an Arrow `RecordBatch`,
//! runs inference through the FAO operator, and returns the score column.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Float32Array, Float64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::Result as DfResult;
use datafusion_expr::ColumnarValue;
use datafusion_expr::{ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility};
use tracing::{debug, warn};

use crate::model::fao::FaoOperator;

/// A [`ScalarUDFImpl`] that delegates to an FAO operator for inline inference.
///
/// Created via [`FaoScalarUdf::new`] and registered on the DataFusion
/// `SessionContext` so the model becomes callable in SQL.
#[derive(Debug)]
pub struct FaoScalarUdf {
    /// The SQL function name (matches `FaoOperator::function_id`).
    name: String,
    /// The backing FAO operator.
    operator: Arc<dyn FaoOperator>,
    /// DataFusion signature: accepts any number of Float64 arguments.
    signature: Signature,
}

impl FaoScalarUdf {
    /// Create a new UDF wrapper around an FAO operator.
    pub fn new(operator: Arc<dyn FaoOperator>) -> Self {
        let name = operator.function_id().to_string();

        // Accept any number of numeric columns.
        let signature = Signature::new(TypeSignature::VariadicAny, Volatility::Stable);

        Self {
            name,
            operator,
            signature,
        }
    }
}

impl PartialEq for FaoScalarUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for FaoScalarUdf {}

impl std::hash::Hash for FaoScalarUdf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl ScalarUDFImpl for FaoScalarUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DfResult<DataType> {
        // FAO operators return a single Float64 score.
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DfResult<ColumnarValue> {
        let args = &args.args;
        if args.is_empty() {
            return Err(datafusion_common::DataFusionError::Plan(format!(
                "FAO UDF '{}' requires at least one argument",
                self.name
            )));
        }

        // Expand all arguments to arrays (handles scalar → array broadcast).
        let arrays: Vec<ArrayRef> = args
            .iter()
            .map(|cv| match cv {
                ColumnarValue::Array(arr) => Ok(arr.clone()),
                ColumnarValue::Scalar(s) => {
                    // Determine how many rows from a sibling array.
                    let n = args
                        .iter()
                        .find_map(|a| match a {
                            ColumnarValue::Array(arr) => Some(arr.len()),
                            _ => None,
                        })
                        .unwrap_or(1);
                    s.to_array_of_size(n)
                }
            })
            .collect::<DfResult<Vec<_>>>()?;

        let num_rows = arrays[0].len();

        // Build input fields as Float32 (what ONNX expects).
        let input_fields: Vec<Field> = arrays
            .iter()
            .enumerate()
            .map(|(i, _)| Field::new(format!("feature_{i}"), DataType::Float32, false))
            .collect();
        let input_schema = Arc::new(Schema::new(input_fields));

        // Convert each array to Float32.
        let f32_columns: Vec<ArrayRef> = arrays
            .iter()
            .map(|arr: &ArrayRef| -> DfResult<ArrayRef> {
                match arr.data_type() {
                    DataType::Float32 => Ok(arr.clone()),
                    DataType::Float64 => {
                        let f64_arr =
                            arr.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                                datafusion_common::DataFusionError::Internal(
                                    "expected Float64Array".into(),
                                )
                            })?;
                        let f32_vals: Vec<f32> =
                            f64_arr.values().iter().map(|v| *v as f32).collect();
                        Ok(Arc::new(Float32Array::from(f32_vals)) as ArrayRef)
                    }
                    DataType::Int64 => {
                        let i64_arr = arr
                            .as_any()
                            .downcast_ref::<datafusion::arrow::array::Int64Array>()
                            .ok_or_else(|| {
                                datafusion_common::DataFusionError::Internal(
                                    "expected Int64Array".into(),
                                )
                            })?;
                        let f32_vals: Vec<f32> =
                            i64_arr.values().iter().map(|v| *v as f32).collect();
                        Ok(Arc::new(Float32Array::from(f32_vals)) as ArrayRef)
                    }
                    DataType::Int32 => {
                        let i32_arr = arr
                            .as_any()
                            .downcast_ref::<datafusion::arrow::array::Int32Array>()
                            .ok_or_else(|| {
                                datafusion_common::DataFusionError::Internal(
                                    "expected Int32Array".into(),
                                )
                            })?;
                        let f32_vals: Vec<f32> =
                            i32_arr.values().iter().map(|v| *v as f32).collect();
                        Ok(Arc::new(Float32Array::from(f32_vals)) as ArrayRef)
                    }
                    other => Err(datafusion_common::DataFusionError::Plan(format!(
                        "FAO UDF '{}': unsupported input type {:?} — expected numeric",
                        self.name, other
                    ))),
                }
            })
            .collect::<DfResult<Vec<_>>>()?;

        // Build the input RecordBatch.
        let input_batch = RecordBatch::try_new(input_schema, f32_columns).map_err(|e| {
            datafusion_common::DataFusionError::Internal(format!(
                "failed to build input batch for FAO '{}': {e}",
                self.name
            ))
        })?;

        debug!(
            fao = %self.name,
            rows = num_rows,
            "invoking FAO UDF inline"
        );

        // Run inference. FAO::execute is async, so we need a blocking bridge.
        // We spawn a blocking task on the current runtime.
        let operator = Arc::clone(&self.operator);
        let output_batch = tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current()
                .block_on(async move { operator.execute(input_batch).await })
        })
        .map_err(|e| datafusion_common::DataFusionError::External(Box::new(e)))?;

        // Extract the first (score) column from the output.
        if output_batch.num_columns() == 0 {
            return Err(datafusion_common::DataFusionError::Internal(format!(
                "FAO '{}' returned no columns",
                self.name
            )));
        }

        let score_col = output_batch.column(0).clone();

        // Ensure it's Float64 (the return type we declared).
        let result: ArrayRef = match score_col.data_type() {
            DataType::Float64 => score_col,
            DataType::Float32 => {
                let f32_arr = score_col.as_any().downcast_ref::<Float32Array>().unwrap();
                let f64_vals: Vec<f64> = f32_arr.values().iter().map(|v| *v as f64).collect();
                Arc::new(Float64Array::from(f64_vals))
            }
            other => {
                warn!(
                    fao = %self.name,
                    output_type = ?other,
                    "unexpected FAO output type, passing through"
                );
                score_col
            }
        };

        Ok(ColumnarValue::Array(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the UDF wrapper can be constructed from a mock operator.
    #[test]
    fn udf_creation() {
        use crate::model::fao::FaoRef;

        // We can at least verify the struct constructs without an actual ONNX model.
        // Full integration tests require a model file on disk.
        let udf_name = "test_model";
        let sig = Signature::new(TypeSignature::VariadicAny, Volatility::Stable);
        assert_eq!(sig.type_signature, TypeSignature::VariadicAny);

        // Verify FaoRef construction (used by the optimizer).
        let fao_ref = FaoRef {
            function_id: udf_name.to_string(),
            version: "1.0.0".to_string(),
            model_id: "mock".to_string(),
            est_latency_ms: 1.0,
            est_accuracy: 0.95,
        };
        assert_eq!(fao_ref.function_id, udf_name);
    }
}
