//! The `Session` — AnamDB's primary public API surface.
//!
//! A session owns the DataFusion context, the logic engine, the model manager,
//! the HITL monitor, and the heterogeneous dispatcher. All neurosymbolic queries
//! flow through here.

use std::sync::Arc;

use datafusion::arrow::array::{Array, BinaryArray, RecordBatch};
use datafusion::prelude::*;
use parking_lot::RwLock;
use tracing::{info, instrument};

use crate::core::error::{AnamError, Result};
use crate::core::provenance::{PolynomialSemiring, ProvenanceMode, Semiring};
use crate::execution::dispatcher::DevicePool;
use crate::execution::optimizer::ParetoOptimizer;
use crate::hitl::monitor::SemanticMonitor;
use crate::hitl::triage::Anomaly;
use crate::logic::engine::LogicEngine;
use crate::logic::nl_compiler::NlCompiler;
use crate::model::registry::ModelRegistry;
use crate::storage::lance_provider::LanceTableManager;

/// Configuration for a new [`Session`].
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Which semiring to use for provenance tracking.
    pub provenance_mode: ProvenanceMode,
    /// Whether to enable NPU / GPU acceleration.
    pub enable_hardware_accel: bool,
    /// LLM API key for NL-to-Datalog compilation.
    pub llm_api_key: Option<String>,
    /// LLM endpoint URL (defaults to OpenAI-compatible).
    pub llm_endpoint: Option<String>,
    /// LLM model name.
    pub llm_model: Option<String>,
    /// Anomaly confidence threshold for the semantic monitor.
    pub anomaly_threshold: f64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            provenance_mode: ProvenanceMode::Polynomial,
            enable_hardware_accel: false,
            llm_api_key: None,
            llm_endpoint: None,
            llm_model: None,
            anomaly_threshold: 0.5,
        }
    }
}

/// The result of a neurosymbolic query.
#[derive(Debug)]
pub struct QueryResult {
    /// The computed record batches.
    pub batches: Vec<RecordBatch>,
    /// If the semantic monitor detected anomalies, they are collected here.
    pub anomalies: Vec<Anomaly>,
    /// Serialized reasoning tree (provenance trace).
    pub reasoning_tree: Option<String>,
}

impl QueryResult {
    /// Returns `true` if the semantic monitor flagged anomalies requiring human input.
    pub fn requires_clarification(&self) -> bool {
        !self.anomalies.is_empty()
    }

    /// Pretty-print the reasoning / proof trace to stdout.
    pub async fn explain_reasoning(&self) -> Result<()> {
        match &self.reasoning_tree {
            Some(tree) => {
                println!("═══ Reasoning Tree ═══\n{tree}");
                Ok(())
            }
            None => {
                println!("(no reasoning tree attached — provenance mode may be Boolean)");
                Ok(())
            }
        }
    }
}

/// Primary entry-point for all AnamDB operations.
pub struct Session {
    /// Underlying DataFusion session (extended with neuro-operators).
    pub(crate) df_ctx: SessionContext,
    /// The logic engine (Scallop-backed Datalog).
    pub(crate) logic_engine: Arc<RwLock<LogicEngine>>,
    /// AI-Tables model registry.
    pub(crate) model_registry: Arc<ModelRegistry>,
    /// NL-to-Datalog compiler.
    #[allow(dead_code)]
    pub(crate) nl_compiler: Arc<NlCompiler>,
    /// Pareto multi-objective optimizer.
    pub(crate) optimizer: Arc<ParetoOptimizer>,
    /// Heterogeneous hardware dispatcher.
    pub(crate) device_pool: Arc<DevicePool>,
    /// Semantic anomaly monitor.
    pub(crate) monitor: Arc<SemanticMonitor>,
    /// Lance table manager.
    pub(crate) lance_mgr: Arc<LanceTableManager>,
    /// Session-level config.
    pub config: SessionConfig,
}

impl Session {
    /// Create a new session with default settings (CPU-only).
    #[instrument(name = "Session::new")]
    pub async fn new() -> Result<Self> {
        Self::with_config(SessionConfig::default()).await
    }

    /// Create a new session with NPU / GPU acceleration enabled.
    #[instrument(name = "Session::new_with_npu")]
    pub async fn new_with_npu() -> Result<Self> {
        let config = SessionConfig {
            enable_hardware_accel: true,
            ..Default::default()
        };
        Self::with_config(config).await
    }

    /// Create a session from an explicit [`SessionConfig`].
    pub async fn with_config(config: SessionConfig) -> Result<Self> {
        info!(
            provenance = ?config.provenance_mode,
            hw_accel = config.enable_hardware_accel,
            "initializing AnamDB session"
        );

        let df_ctx = SessionContext::new();

        let logic_engine = Arc::new(RwLock::new(LogicEngine::new(config.provenance_mode)?));
        let model_registry = Arc::new(ModelRegistry::new());

        let nl_compiler = Arc::new(NlCompiler::new(
            config.llm_api_key.clone(),
            config.llm_endpoint.clone(),
            config.llm_model.clone(),
        ));

        let device_pool = Arc::new(if config.enable_hardware_accel {
            DevicePool::detect_hardware().await?
        } else {
            DevicePool::cpu_only()
        });

        let optimizer = Arc::new(ParetoOptimizer::new(
            Arc::clone(&model_registry),
            Arc::clone(&device_pool),
        ));

        let monitor = Arc::new(SemanticMonitor::new(config.anomaly_threshold));
        let lance_mgr = Arc::new(LanceTableManager::new());

        Ok(Self {
            df_ctx,
            logic_engine,
            model_registry,
            nl_compiler,
            optimizer,
            device_pool,
            monitor,
            lance_mgr,
            config,
        })
    }

    // ── Table operations ───────────────────────────────────────────────

    /// Register an existing Lance dataset as a queryable table.
    #[instrument(skip(self))]
    pub async fn register_table(&self, name: &str, path: &str) -> Result<()> {
        info!(table = name, path, "registering Lance table");
        let provider = self.lance_mgr.open_table(path).await?;
        self.df_ctx
            .register_table(name, provider)
            .map_err(AnamError::DataFusion)?;
        Ok(())
    }

    // ── Logic operations ───────────────────────────────────────────────

    /// Define a logical constraint from natural language.
    #[instrument(skip(self))]
    pub async fn register_logic_from_nl(
        &self,
        name: &str,
        table: &str,
        nl_description: &str,
    ) -> Result<()> {
        info!(rule = name, table, "compiling NL → Datalog");
        let datalog_source = self.nl_compiler.compile(nl_description, table).await?;
        info!(datalog = %datalog_source, "generated Datalog");
        self.logic_engine
            .write()
            .register_rule(name, &datalog_source)?;
        Ok(())
    }

    /// Register a raw Datalog rule directly.
    pub fn register_logic(&self, name: &str, datalog: &str) -> Result<()> {
        self.logic_engine.write().register_rule(name, datalog)
    }

    // ── Query execution ────────────────────────────────────────────────

    /// Execute a neurosymbolic SQL query.
    #[instrument(skip(self))]
    pub async fn sql(&self, query: &str) -> Result<QueryResult> {
        info!(query, "executing neurosymbolic query");

        let (clean_sql, constraints) = self.optimizer.parse_constraints(query)?;

        let df = self
            .df_ctx
            .sql(&clean_sql)
            .await
            .map_err(AnamError::DataFusion)?;

        let batches = if let Some(c) = constraints {
            self.optimizer.execute_with_constraints(df, c).await?
        } else {
            df.collect().await.map_err(AnamError::DataFusion)?
        };

        let anomalies = self.monitor.inspect_batches(&batches)?;
        let reasoning_tree = self.build_reasoning_tree(&batches)?;

        Ok(QueryResult {
            batches,
            anomalies,
            reasoning_tree,
        })
    }

    /// Refine a query after human feedback on an anomaly.
    #[instrument(skip(self))]
    pub async fn refine_query(&self, correction: &str) -> Result<QueryResult> {
        info!(correction, "refining query with human feedback");

        let patch = self
            .nl_compiler
            .compile(correction, "__refinement__")
            .await?;

        self.logic_engine
            .write()
            .register_rule("__refinement_patch__", &patch)?;

        let batches = self.logic_engine.read().evaluate_all()?;
        let anomalies = self.monitor.inspect_batches(&batches)?;
        let reasoning_tree = self.build_reasoning_tree(&batches)?;

        Ok(QueryResult {
            batches,
            anomalies,
            reasoning_tree,
        })
    }

    // ── Model management ───────────────────────────────────────────────

    /// Access the model registry for AI-Table operations.
    pub fn models(&self) -> &ModelRegistry {
        &self.model_registry
    }

    // ── Internal helpers ───────────────────────────────────────────────

    fn build_reasoning_tree(
        &self,
        batches: &[RecordBatch],
    ) -> Result<Option<String>> {
        if self.config.provenance_mode == ProvenanceMode::Boolean {
            return Ok(None);
        }

        let mut tree = String::new();
        for (i, batch) in batches.iter().enumerate() {
            if let Some(col_idx) = batch.schema().column_with_name("provenance") {
                tree.push_str(&format!("── Batch {i} ({} rows) ──\n", batch.num_rows()));
                let col = batch.column(col_idx.0);
                if let Some(binary_arr) = col.as_any().downcast_ref::<BinaryArray>() {
                    for row in 0..binary_arr.len() {
                        let nulls = binary_arr.nulls();
                        let valid = nulls.map_or(true, |n| n.is_valid(row));
                        if valid {
                            let bytes = binary_arr.value(row);
                            match PolynomialSemiring::from_bytes(bytes) {
                                Ok(poly) => {
                                    tree.push_str(&format!(
                                        "  row {row}: {}\n",
                                        poly.explain()
                                    ));
                                }
                                Err(_) => {
                                    tree.push_str(&format!(
                                        "  row {row}: <raw {} bytes>\n",
                                        bytes.len()
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        if tree.is_empty() {
            Ok(None)
        } else {
            Ok(Some(tree))
        }
    }
}
