//! `ProvenanceExec` — attaches or merges provenance columns on record batches.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, BinaryArray, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::SendableRecordBatchStream;
use datafusion_common::Result as DfResult;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::StreamExt;
use tracing::debug;

use crate::core::provenance::{PolynomialSemiring, ProvenanceMode, ProvenanceToken, Semiring};

/// A physical operator that attaches provenance information to output batches.
#[derive(Debug)]
pub struct ProvenanceExec {
    /// Child plan.
    child: Arc<dyn ExecutionPlan>,
    /// The provenance mode to apply.
    provenance_mode: ProvenanceMode,
    /// Model/function metadata for provenance tokens.
    model_ver_id: String,
    func_id: String,
    /// Output schema (child schema + provenance column).
    schema: SchemaRef,
    /// Plan properties.
    properties: PlanProperties,
}

impl ProvenanceExec {
    /// Create a new provenance attachment node.
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        provenance_mode: ProvenanceMode,
        model_ver_id: impl Into<String>,
        func_id: impl Into<String>,
    ) -> Self {
        let child_schema = child.schema();

        // Add provenance column if not already present.
        let schema = if child_schema.column_with_name("provenance").is_some() {
            child_schema
        } else {
            let mut fields: Vec<Arc<Field>> = child_schema.fields().to_vec();
            fields.push(Arc::new(Field::new("provenance", DataType::Binary, true)));
            Arc::new(Schema::new(fields))
        };

        let properties = child.properties().clone();

        Self {
            child,
            provenance_mode,
            model_ver_id: model_ver_id.into(),
            func_id: func_id.into(),
            schema,
            properties,
        }
    }
}

impl DisplayAs for ProvenanceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProvenanceExec: mode={:?}, model={}, func={}",
            self.provenance_mode, self.model_ver_id, self.func_id
        )
    }
}

impl ExecutionPlan for ProvenanceExec {
    fn name(&self) -> &str {
        "ProvenanceExec"
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
        Ok(Arc::new(ProvenanceExec::new(
            children[0].clone(),
            self.provenance_mode,
            self.model_ver_id.clone(),
            self.func_id.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let child_stream = self.child.execute(partition, context)?;
        let schema = self.schema.clone();
        let mode = self.provenance_mode;
        let model_ver = self.model_ver_id.clone();
        let func = self.func_id.clone();

        let output_stream = child_stream.map(move |batch_result| {
            match batch_result {
                Ok(batch) => {
                    let num_rows = batch.num_rows();
                    debug!(rows = num_rows, "ProvenanceExec: attaching provenance");

                    // Generate provenance bytes for each row.
                    let prov_bytes: Vec<Vec<u8>> = (0..num_rows)
                        .map(|row_idx| {
                            match mode {
                                ProvenanceMode::Boolean => {
                                    vec![1u8] // EXISTS = true
                                }
                                ProvenanceMode::Probability => 1.0_f64.to_le_bytes().to_vec(),
                                ProvenanceMode::Polynomial => {
                                    let token = ProvenanceToken {
                                        model_ver_id: model_ver.clone(),
                                        func_id: func.clone(),
                                        source_record_ids: vec![format!("row_{row_idx}")],
                                    };
                                    let poly = PolynomialSemiring::singleton(token);
                                    poly.to_bytes().unwrap_or_default()
                                }
                            }
                        })
                        .collect();

                    let prov_refs: Vec<&[u8]> = prov_bytes.iter().map(|b| b.as_slice()).collect();
                    let prov_array: ArrayRef = Arc::new(BinaryArray::from(prov_refs));

                    // Append provenance column to existing batch.
                    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
                        .map(|i| batch.column(i).clone())
                        .collect();

                    // If the schema already has a provenance column, replace it.
                    if let Some((idx, _)) = batch.schema().column_with_name("provenance") {
                        columns[idx] = prov_array;
                    } else {
                        columns.push(prov_array);
                    }

                    RecordBatch::try_new(schema.clone(), columns).map_err(|e| {
                        datafusion_common::DataFusionError::ArrowError(Box::new(e), None)
                    })
                }
                Err(e) => Err(e),
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            output_stream,
        )))
    }
}
