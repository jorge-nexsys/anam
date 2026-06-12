//! Execution engine: custom DataFusion operators, Pareto optimizer,
//! heterogeneous hardware dispatcher, distributed task routing,
//! adaptive concurrency control, and 5-stage symbolic integration pipeline.

pub mod concurrency;
pub mod dispatcher;
pub mod distributed_optimizer;
pub mod fao_udf;
pub mod logic_filter;
pub mod logic_optimizer_rule;
pub mod neural_scan;
pub mod optimizer;
pub mod pipeline;
pub mod predict_exec;
pub mod provenance_exec;
pub mod task_router;
