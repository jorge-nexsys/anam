//! `PredictExec` — a DataFusion `ExecutionPlan` for PREDICT queries.
//!
//! This node lives in the physical plan tree and handles `PREDICT CLASS OF`
//! and `PREDICT VALUE OF` queries by running the backing FAO operator and
//! appending the prediction column to the child's output schema.
//!
//! Unlike `NeuralScanExec` (which replaces the child's schema entirely),
//! `PredictExec` **projects** — it concatenates the child's columns with the
//! model's output, preserving the original data alongside predictions.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Float32Array, Float64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::SendableRecordBatchStream;
use datafusion_common::Result as DfResult;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::StreamExt;
use tracing::debug;

use crate::model::fao::FaoOperator;

/// Prediction type: classification (thresholded) or regression (raw score).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictType {
    /// `PREDICT CLASS OF` — the output is thresholded at 0.5 for binary classification.
    Class,
    /// `PREDICT VALUE OF` — the output is a raw regression score.
    Value,
}

/// A physical operator that runs an FAO model and appends prediction columns
/// to the child plan's output.
#[derive(Debug)]
pub struct PredictExec {
    /// The child plan producing input batches.
    child: Arc<dyn ExecutionPlan>,
    /// The FAO operator to apply.
    operator: Arc<dyn FaoOperator>,
    /// The prediction column name.
    prediction_column: String,
    /// Prediction type.
    predict_type: PredictType,
    /// Output schema (child schema + prediction column).
    schema: SchemaRef,
    /// Plan properties (inherited from child).
    properties: PlanProperties,
}

impl PredictExec {
    /// Create a new predict execution node.
    ///
    /// The output schema will be the child's schema plus a `prediction` column
    /// of type Float64 (for Value) or Int64 (for Class).
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        operator: Arc<dyn FaoOperator>,
        prediction_column: impl Into<String>,
        predict_type: PredictType,
    ) -> Self {
        let prediction_column = prediction_column.into();
        let child_schema = child.schema();

        // Build output schema: child fields + prediction field.
        let mut fields: Vec<Arc<Field>> = child_schema.fields().to_vec();
        let pred_field = match predict_type {
            PredictType::Class => Field::new(&prediction_column, DataType::Int64, false),
            PredictType::Value => Field::new(&prediction_column, DataType::Float64, false),
        };
        fields.push(Arc::new(pred_field));
        let schema = Arc::new(Schema::new(fields));

        let properties = child.properties().clone();

        Self {
            child,
            operator,
            prediction_column,
            predict_type,
            schema,
            properties,
        }
    }
}

impl DisplayAs for PredictExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PredictExec: fao={}@{}, model={}, type={:?}, col={}",
            self.operator.function_id(),
            self.operator.version(),
            self.operator.model_id(),
            self.predict_type,
            self.prediction_column,
        )
    }
}

impl ExecutionPlan for PredictExec {
    fn name(&self) -> &str {
        "PredictExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.child]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(PredictExec::new(
            children[0].clone(),
            Arc::clone(&self.operator),
            self.prediction_column.clone(),
            self.predict_type,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let child_stream = self.child.execute(partition, context)?;
        let operator = Arc::clone(&self.operator);
        let schema = self.schema.clone();
        let predict_type = self.predict_type;

        let output_stream = child_stream.then(move |batch_result| {
            let op = Arc::clone(&operator);
            let out_schema = schema.clone();
            async move {
                match batch_result {
                    Ok(batch) => {
                        debug!(
                            rows = batch.num_rows(),
                            fao = op.function_id(),
                            "PredictExec: processing batch"
                        );

                        // Run FAO inference on the input batch.
                        let inference_result = op.execute(batch.clone()).await.map_err(|e| {
                            datafusion_common::DataFusionError::External(Box::new(e))
                        })?;

                        // Extract the score column from the inference result.
                        if inference_result.num_columns() == 0 {
                            return Err(datafusion_common::DataFusionError::Internal(
                                "FAO operator returned no columns".into(),
                            ));
                        }

                        let score_col = inference_result.column(0);

                        // Convert score to the appropriate prediction type.
                        let pred_col: ArrayRef = match predict_type {
                            PredictType::Value => {
                                // Ensure Float64.
                                match score_col.data_type() {
                                    DataType::Float64 => score_col.clone(),
                                    DataType::Float32 => {
                                        let f32_arr =
                                            score_col.as_any().downcast_ref::<Float32Array>().unwrap();
                                        let f64_vals: Vec<f64> =
                                            f32_arr.values().iter().map(|v| *v as f64).collect();
                                        Arc::new(Float64Array::from(f64_vals))
                                    }
                                    _ => score_col.clone(),
                                }
                            }
                            PredictType::Class => {
                                // Threshold at 0.5 for binary classification.
                                let scores: Vec<f64> = match score_col.data_type() {
                                    DataType::Float64 => score_col
                                        .as_any()
                                        .downcast_ref::<Float64Array>()
                                        .unwrap()
                                        .values()
                                        .iter()
                                        .copied()
                                        .collect(),
                                    DataType::Float32 => score_col
                                        .as_any()
                                        .downcast_ref::<Float32Array>()
                                        .unwrap()
                                        .values()
                                        .iter()
                                        .map(|v| *v as f64)
                                        .collect(),
                                    _ => vec![0.0; batch.num_rows()],
                                };
                                let classes: Vec<i64> =
                                    scores.iter().map(|s| if *s >= 0.5 { 1 } else { 0 }).collect();
                                Arc::new(datafusion::arrow::array::Int64Array::from(classes))
                            }
                        };

                        // Concatenate child columns + prediction column.
                        let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
                            .map(|i| batch.column(i).clone())
                            .collect();
                        columns.push(pred_col);

                        RecordBatch::try_new(out_schema, columns).map_err(|e| {
                            datafusion_common::DataFusionError::ArrowError(Box::new(e), None)
                        })
                    }
                    Err(e) => Err(e),
                }
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            output_stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_type_display() {
        assert_eq!(format!("{:?}", PredictType::Class), "Class");
        assert_eq!(format!("{:?}", PredictType::Value), "Value");
    }
}
