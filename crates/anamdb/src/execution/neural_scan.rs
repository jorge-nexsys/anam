//! `NeuralScanExec` — a DataFusion `ExecutionPlan` that wraps an FAO operator.
//!
//! This node lives in the physical plan tree. When DataFusion pulls batches
//! from it, it runs the backing neural model via the FAO interface.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::execution::SendableRecordBatchStream;
use datafusion_common::Result as DfResult;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use futures::StreamExt;
use tracing::debug;

use crate::model::fao::FaoOperator;

/// A physical operator that applies an FAO (neural inference) to its child's
/// output batches.
#[derive(Debug)]
pub struct NeuralScanExec {
    /// The child plan producing input batches.
    child: Arc<dyn ExecutionPlan>,
    /// The FAO operator to apply.
    operator: Arc<dyn FaoOperator>,
    /// Output schema (from the FAO).
    schema: SchemaRef,
    /// Plan properties (inherited from child, with output schema overridden).
    properties: PlanProperties,
}

impl NeuralScanExec {
    /// Create a new neural scan node.
    pub fn new(child: Arc<dyn ExecutionPlan>, operator: Arc<dyn FaoOperator>) -> Self {
        let schema = operator.output_schema().clone();
        let properties = child.properties().clone();

        Self {
            child,
            operator,
            schema,
            properties,
        }
    }
}

impl DisplayAs for NeuralScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NeuralScanExec: fao={}@{}, model={}",
            self.operator.function_id(),
            self.operator.version(),
            self.operator.model_id()
        )
    }
}

impl ExecutionPlan for NeuralScanExec {
    fn name(&self) -> &str {
        "NeuralScanExec"
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
        Ok(Arc::new(NeuralScanExec::new(
            children[0].clone(),
            Arc::clone(&self.operator),
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

        let output_stream = child_stream.then(move |batch_result| {
            let op = Arc::clone(&operator);
            async move {
                match batch_result {
                    Ok(batch) => {
                        debug!(
                            rows = batch.num_rows(),
                            fao = op.function_id(),
                            "NeuralScanExec: processing batch"
                        );
                        op.execute(batch)
                            .await
                            .map_err(|e| datafusion_common::DataFusionError::External(Box::new(e)))
                    }
                    Err(e) => Err(e),
                }
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            output_stream,
        )))
    }
}
