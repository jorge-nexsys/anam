//! `LogicFilterExec` — a DataFusion `ExecutionPlan` that applies Datalog rules
//! as a filter/join node in the physical plan.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::execution::SendableRecordBatchStream;
use datafusion_common::Result as DfResult;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::StreamExt;
use parking_lot::RwLock;
use tracing::debug;

use crate::logic::engine::LogicEngine;

/// A physical operator that evaluates Datalog rules against input batches.
#[derive(Debug)]
pub struct LogicFilterExec {
    /// Child plan providing input data.
    child: Arc<dyn ExecutionPlan>,
    /// The name of the rule to evaluate.
    rule_name: String,
    /// Shared logic engine reference.
    logic_engine: Arc<RwLock<LogicEngine>>,
    /// Output schema (same as child for filter operations).
    schema: SchemaRef,
    /// Plan properties.
    properties: PlanProperties,
}

impl LogicFilterExec {
    /// Create a new logic filter node.
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        rule_name: impl Into<String>,
        logic_engine: Arc<RwLock<LogicEngine>>,
    ) -> Self {
        let schema = child.schema();
        let properties = child.properties().clone();

        Self {
            child,
            rule_name: rule_name.into(),
            logic_engine,
            schema,
            properties,
        }
    }
}

impl DisplayAs for LogicFilterExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LogicFilterExec: rule={}", self.rule_name)
    }
}

impl ExecutionPlan for LogicFilterExec {
    fn name(&self) -> &str {
        "LogicFilterExec"
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
        Ok(Arc::new(LogicFilterExec::new(
            children[0].clone(),
            self.rule_name.clone(),
            Arc::clone(&self.logic_engine),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let child_stream = self.child.execute(partition, context)?;
        let logic_engine = Arc::clone(&self.logic_engine);
        let rule_name = self.rule_name.clone();
        let schema = self.schema.clone();

        let output_stream = child_stream.filter_map(move |batch_result| {
            let engine = Arc::clone(&logic_engine);
            let rule = rule_name.clone();
            async move {
                match batch_result {
                    Ok(batch) => {
                        debug!(
                            rows = batch.num_rows(),
                            rule = %rule,
                            "LogicFilterExec: evaluating batch"
                        );
                        let engine_read = engine.read();
                        match engine_read.evaluate(&rule) {
                            Ok(results) => results.into_iter().next().map(Ok),
                            Err(e) => Some(Err(datafusion_common::DataFusionError::External(
                                Box::new(e),
                            ))),
                        }
                    }
                    Err(e) => Some(Err(e)),
                }
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            output_stream,
        )))
    }
}
