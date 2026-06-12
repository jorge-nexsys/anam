//! The `Session` — AnamDB's primary public API surface.
//!
//! A session owns the DataFusion context, the logic engine, the model manager,
//! the HITL monitor, and the heterogeneous dispatcher. All neurosymbolic queries
//! flow through here.

use std::sync::Arc;

use datafusion::arrow::array::{Array, BinaryArray, RecordBatch};
use datafusion::logical_expr::ScalarUDF;
use datafusion::prelude::*;
use parking_lot::RwLock;
use tracing::{info, instrument};

use crate::execution::fao_udf::FaoScalarUdf;

use crate::core::error::{AnamError, Result};
use crate::core::provenance::{
    AdaptiveProvenanceSelector, PolynomialSemiring, ProvenanceMode, ProvenanceToken, Semiring,
};
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

impl SessionConfig {
    /// Attempt to load configuration from a TOML file.
    pub fn load_from_toml(path: &str) -> std::result::Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let table: toml::Table = toml::from_str(&content).map_err(|e| e.to_string())?;

        let mut config = SessionConfig::default();
        if let Some(engine) = table.get("engine").and_then(|v| v.as_table()) {
            if let Some(prov) = engine.get("provenance_mode").and_then(|v| v.as_str()) {
                config.provenance_mode = match prov.to_lowercase().as_str() {
                    "boolean" | "bool" => ProvenanceMode::Boolean,
                    "probability" | "prob" => ProvenanceMode::Probability,
                    _ => ProvenanceMode::Polynomial,
                };
            }
            if let Some(gpu) = engine.get("gpu").and_then(|v| v.as_bool()) {
                config.enable_hardware_accel = gpu;
            }
            if let Some(threshold) = engine.get("anomaly_threshold").and_then(|v| v.as_float()) {
                config.anomaly_threshold = threshold;
            }
        }
        Ok(config)
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
    /// Lance table manager (legacy — kept for backward compatibility).
    #[allow(dead_code)]
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

        // Register the Datalog filter pushdown optimizer rule.
        let logic_rule = Arc::new(
            crate::execution::logic_optimizer_rule::LogicOptimizerRule::new(
                Arc::clone(&logic_engine),
            ),
        );
        df_ctx.add_optimizer_rule(logic_rule);

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
    ///
    /// Uses the streaming Lance provider for push-down projection and
    /// filter support (no eager memory loading).
    #[instrument(skip(self))]
    pub async fn register_table(&self, name: &str, path: &str) -> Result<()> {
        info!(table = name, path, "registering Lance table (streaming)");
        let provider =
            crate::storage::streaming_provider::LanceStreamingProvider::open(path).await?;
        self.df_ctx
            .register_table(name, Arc::new(provider))
            .map_err(AnamError::DataFusion)?;
        Ok(())
    }

    // ── Write path ──────────────────────────────────────────────────────

    /// Insert rows into a registered Lance table.
    ///
    /// Appends the given batches to the underlying Lance dataset.
    #[instrument(skip(self, batches))]
    pub async fn insert(
        &self,
        _table_name: &str,
        lance_path: &str,
        batches: Vec<RecordBatch>,
        schema: Arc<datafusion::arrow::datatypes::Schema>,
    ) -> Result<crate::storage::write_path::WriteResult> {
        info!(table = _table_name, "INSERT into table");
        crate::storage::write_path::insert_rows(lance_path, batches, schema).await
    }

    /// Delete rows from a registered Lance table matching a predicate.
    #[instrument(skip(self))]
    pub async fn delete(
        &self,
        _table_name: &str,
        lance_path: &str,
        predicate: &str,
    ) -> Result<crate::storage::write_path::WriteResult> {
        info!(table = _table_name, predicate, "DELETE from table");
        crate::storage::write_path::delete_rows(lance_path, predicate).await
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
    ///
    /// Supports standard SQL, constraint-annotated SQL (`WITH (max_latency_ms = ...)`),
    /// and the extended `PREDICT CLASS OF` / `PREDICT VALUE OF` syntax.
    #[instrument(skip(self))]
    pub async fn sql(&self, query: &str) -> Result<QueryResult> {
        info!(query, "executing neurosymbolic query");

        // ── PREDICT SQL interception ──────────────────────────────────
        let upper = query.trim().to_uppercase();
        if upper.starts_with("PREDICT ") {
            return self.execute_predict(query).await;
        }

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

        // Apply registered Datalog rules as post-filters.
        // Rules whose columns don't match the result schema are skipped.
        let batches = self.logic_engine.read().filter_batches(&batches)?;

        // Attach provenance column to every output batch.
        let batches = self.attach_provenance(&batches)?;

        let anomalies = self.monitor.inspect_batches(&batches)?;
        let reasoning_tree = self.build_reasoning_tree(&batches)?;

        Ok(QueryResult {
            batches,
            anomalies,
            reasoning_tree,
        })
    }

    /// Execute a PREDICT SQL query.
    ///
    /// Intercepts `PREDICT CLASS OF <col> FROM <table> WITH (model = '...')`
    /// and routes through the FAO operator pipeline.
    async fn execute_predict(&self, query: &str) -> Result<QueryResult> {
        use crate::execution::predict_exec::{PredictExec, PredictType};

        let (target_col, source_table, model_name, pred_type) =
            ParetoOptimizer::parse_predict_query(query).ok_or_else(|| {
                AnamError::QueryParse(format!("could not parse PREDICT query: {query}"))
            })?;

        // Map the parser's prediction type to PredictExec's enum.
        let predict_type = match pred_type {
            crate::execution::optimizer::PredictionType::Classification => PredictType::Class,
            crate::execution::optimizer::PredictionType::Regression => PredictType::Value,
        };

        info!(
            target = %target_col,
            table = %source_table,
            model = %model_name,
            pred_type = ?predict_type,
            "executing PREDICT query via PredictExec"
        );

        // Find the FAO operator for the requested model.
        let operator = self
            .model_registry
            .get_latest_operator(&model_name)
            .map_err(|_| {
                AnamError::ModelNotFound(format!(
                    "model '{}' not found for PREDICT query",
                    model_name
                ))
            })?;

        // Build a DataFusion physical plan for the source table.
        let select_sql = format!("SELECT * FROM {source_table}");
        let df = self
            .df_ctx
            .sql(&select_sql)
            .await
            .map_err(AnamError::DataFusion)?;

        // Get the physical plan from the DataFrame.
        let logical_plan = df.logical_plan().clone();
        let state = self.df_ctx.state();
        let physical_plan = state
            .create_physical_plan(&logical_plan)
            .await
            .map_err(AnamError::DataFusion)?;

        // Wrap with PredictExec — this becomes a native DataFusion node.
        let predict_node = Arc::new(PredictExec::new(
            physical_plan,
            operator,
            format!("prediction_{}", target_col),
            predict_type,
        ));

        // Execute through DataFusion's pipeline.
        use datafusion_physical_plan::ExecutionPlan;
        let task_ctx = state.task_ctx();
        let stream = predict_node
            .execute(0, task_ctx)
            .map_err(AnamError::DataFusion)?;

        use futures::TryStreamExt;
        let result_batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(AnamError::DataFusion)?;

        let result_batches = self.attach_provenance(&result_batches)?;
        let anomalies = self.monitor.inspect_batches(&result_batches)?;
        let reasoning_tree = self.build_reasoning_tree(&result_batches)?;

        Ok(QueryResult {
            batches: result_batches,
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

    /// Access the logic engine.
    pub fn logic_engine(&self) -> &Arc<RwLock<LogicEngine>> {
        &self.logic_engine
    }

    /// Access the heterogeneous device pool.
    pub fn device_pool(&self) -> &DevicePool {
        &self.device_pool
    }

    /// Load an ONNX model from disk and register it as both an AI-Table entry
    /// and an FAO operator.
    #[instrument(skip(self))]
    pub fn load_onnx_model(
        &self,
        name: &str,
        model_path: &str,
        function_id: &str,
        num_input_features: usize,
    ) -> Result<String> {
        use crate::model::ai_tables::{AiModelEntry, DeviceAffinity, ModelFormat};
        use crate::model::onnx_adapter::OnnxFaoOperator;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};

        info!(name, model_path, function_id, "loading ONNX model");

        // Build input/output schemas for the operator.
        let input_fields: Vec<Field> = (0..num_input_features)
            .map(|i| Field::new(format!("feature_{i}"), DataType::Float32, false))
            .collect();
        let input_schema = Arc::new(Schema::new(input_fields));
        let output_schema = Arc::new(Schema::new(vec![Field::new(
            "score",
            DataType::Float64,
            false,
        )]));

        // Load the ONNX model.
        let file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

        let entry = AiModelEntry::builder(name, "1.0.0")
            .format(ModelFormat::Onnx)
            .artifact_path(model_path)
            .avg_latency_ms(1.0)
            .accuracy(0.95)
            .size_bytes(file_size)
            .device_affinity(DeviceAffinity::Any)
            .build();

        let model_id = entry.model_id.clone();

        // Register in AI-Tables catalog.
        self.model_registry.register_model(entry)?;

        // Create and register the FAO operator with device pool.
        let operator = OnnxFaoOperator::load(
            model_path,
            function_id,
            "1.0.0",
            &model_id,
            input_schema,
            output_schema,
            1.0,
            0.95,
        )?
        .with_device_pool(Arc::clone(&self.device_pool));
        let operator: Arc<dyn crate::model::fao::FaoOperator> = Arc::new(operator);
        self.model_registry
            .register_operator(Arc::clone(&operator))?;
        self.register_fao_udf(Arc::clone(&operator));

        info!(model_id = %model_id, "ONNX model registered");
        Ok(model_id)
    }

    /// Load an ONNX model with custom performance metrics.
    ///
    /// Use this to register multiple model variants with different
    /// latency/accuracy trade-offs for Pareto optimization.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip(self))]
    pub fn load_onnx_model_with_metrics(
        &self,
        name: &str,
        version: &str,
        model_path: &str,
        function_id: &str,
        num_input_features: usize,
        avg_latency_ms: f64,
        accuracy: f64,
    ) -> Result<String> {
        use crate::model::ai_tables::{AiModelEntry, DeviceAffinity, ModelFormat};
        use crate::model::onnx_adapter::OnnxFaoOperator;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};

        info!(
            name,
            version,
            model_path,
            function_id,
            avg_latency_ms,
            accuracy,
            "loading ONNX model variant"
        );

        let input_fields: Vec<Field> = (0..num_input_features)
            .map(|i| Field::new(format!("feature_{i}"), DataType::Float32, false))
            .collect();
        let input_schema = Arc::new(Schema::new(input_fields));
        let output_schema = Arc::new(Schema::new(vec![Field::new(
            "score",
            DataType::Float64,
            false,
        )]));

        let file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

        let entry = AiModelEntry::builder(name, version)
            .format(ModelFormat::Onnx)
            .artifact_path(model_path)
            .avg_latency_ms(avg_latency_ms)
            .accuracy(accuracy)
            .size_bytes(file_size)
            .device_affinity(DeviceAffinity::Any)
            .build();

        let model_id = entry.model_id.clone();
        self.model_registry.register_model(entry)?;

        let operator = OnnxFaoOperator::load(
            model_path,
            function_id,
            version,
            &model_id,
            input_schema,
            output_schema,
            avg_latency_ms,
            accuracy,
        )?
        .with_device_pool(Arc::clone(&self.device_pool));
        let operator: Arc<dyn crate::model::fao::FaoOperator> = Arc::new(operator);
        self.model_registry
            .register_operator(Arc::clone(&operator))?;
        self.register_fao_udf(Arc::clone(&operator));

        info!(model_id = %model_id, "ONNX model variant registered");
        Ok(model_id)
    }

    // ── Beta API: Logic Pack SDK ─────────────────────────────────────────

    /// Load a Logic Pack into this session.
    ///
    /// Registers all bundled rules and models from the pack.
    #[instrument(skip(self, pack), fields(pack_name = %pack.name, pack_version = %pack.version))]
    pub fn load_logic_pack(&self, pack: &crate::sdk::LogicPack) -> Result<String> {
        info!(
            name = %pack.name,
            version = %pack.version,
            rules = pack.rules.len(),
            models = pack.models.len(),
            "loading Logic Pack"
        );

        // Register all rules.
        let mut engine = self.logic_engine.write();
        for rule in &pack.rules {
            engine.register_rule(&rule.name, &rule.datalog)?;
        }
        drop(engine);

        // Register all model references.
        for model in &pack.models {
            self.load_onnx_model_with_metrics(
                &model.name,
                &pack.version,
                &model.artifact_path,
                &model.name,
                model.num_features,
                model.avg_latency_ms,
                model.accuracy,
            )?;
        }

        let summary = pack.summary();
        info!("Logic Pack loaded successfully");
        Ok(summary)
    }

    // ── Beta API: Query Explainer ────────────────────────────────────────

    /// Generate an explanation of the last query results.
    ///
    /// - `Coarse`: High-level summary (rules, models, stats, hardware).
    /// - `Fine`: Per-row provenance trace with source record lineage.
    pub fn explain_query(
        &self,
        batches: &[RecordBatch],
        level: crate::hitl::explainer::ExplainLevel,
    ) -> Result<crate::hitl::explainer::QueryExplanation> {
        let engine = self.logic_engine.read();
        let rules: Vec<(String, String)> = engine
            .list_rules()
            .into_iter()
            .map(|name| {
                let body = engine
                    .get_rule_body(&name)
                    .unwrap_or_else(|| "<unknown>".to_string());
                (name, body)
            })
            .collect();

        let models: Vec<(String, String)> = self
            .model_registry
            .list_models()
            .into_iter()
            .map(|e| (e.name.clone(), e.version.clone()))
            .collect();

        let context = crate::hitl::explainer::ExplainContext {
            rules,
            models,
            provenance_mode: format!("{:?}", self.config.provenance_mode),
            device_summary: self.device_pool.summary(),
        };

        crate::hitl::explainer::Explainer::explain(level, batches, &context)
    }

    // ── Beta API: Self-Repair ────────────────────────────────────────────

    /// Trigger the syntactic self-repair agent for a failed operation.
    ///
    /// When the session has an `llm_api_key` configured, the agent uses
    /// LLM-powered diagnosis and repair. Otherwise, it falls back to
    /// pattern-matching heuristics.
    pub async fn self_repair(
        &self,
        error_msg: &str,
        operator_name: &str,
        context: &str,
    ) -> Result<crate::hitl::self_repair::RepairReport> {
        let mut agent = if let Some(ref api_key) = self.config.llm_api_key {
            crate::hitl::self_repair::SelfRepairAgent::with_llm(
                api_key.clone(),
                self.config.llm_endpoint.clone(),
                self.config.llm_model.clone(),
            )
        } else {
            crate::hitl::self_repair::SelfRepairAgent::new()
        };

        // Provide available model names for swap recommendations.
        let model_names: Vec<String> = self
            .model_registry
            .list_models()
            .into_iter()
            .map(|e| e.name.clone())
            .collect();
        agent.register_available_models(model_names);

        agent.diagnose_and_repair(error_msg, operator_name, context).await
    }

    // ── Internal helpers ───────────────────────────────────────────────

    fn register_fao_udf(&self, operator: Arc<dyn crate::model::fao::FaoOperator>) {
        let udf_impl = FaoScalarUdf::new(operator);
        let udf = ScalarUDF::from(udf_impl);
        self.df_ctx.register_udf(udf);
        info!("registered FAO as DataFusion scalar UDF");
    }

    /// Attach provenance column to every output batch.
    ///
    /// Uses adaptive provenance selection when configured: automatically
    /// downgrades from Polynomial → Probability → Boolean based on batch size.
    fn attach_provenance(&self, batches: &[RecordBatch]) -> Result<Vec<RecordBatch>> {
        use datafusion::arrow::array::{ArrayRef, BinaryArray};
        use datafusion::arrow::datatypes::{DataType, Field, Schema};

        if batches.is_empty() {
            return Ok(batches.to_vec());
        }

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        // Adaptive mode: select the appropriate provenance mode based on batch size.
        let selector = AdaptiveProvenanceSelector::default();
        let mode = selector.select(self.config.provenance_mode, total_rows);

        let mut result = Vec::with_capacity(batches.len());

        for batch in batches {
            // Skip if provenance column already exists.
            if batch.schema().column_with_name("provenance").is_some() {
                result.push(batch.clone());
                continue;
            }

            let num_rows = batch.num_rows();

            // Generate provenance bytes for each row.
            let prov_bytes: Vec<Vec<u8>> = (0..num_rows)
                .map(|row_idx| match mode {
                    ProvenanceMode::Boolean | ProvenanceMode::Adaptive => vec![1u8],
                    ProvenanceMode::Probability => 1.0_f64.to_le_bytes().to_vec(),
                    ProvenanceMode::Polynomial => {
                        let token = ProvenanceToken {
                            model_ver_id: "query_pipeline".to_string(),
                            func_id: "sql".to_string(),
                            source_record_ids: vec![format!("row_{row_idx}")],
                        };
                        let poly = PolynomialSemiring::singleton(token);
                        poly.to_bytes().unwrap_or_default()
                    }
                })
                .collect();

            let prov_refs: Vec<&[u8]> = prov_bytes.iter().map(|b| b.as_slice()).collect();
            let prov_array: ArrayRef = Arc::new(BinaryArray::from(prov_refs));

            // Build new schema with provenance column.
            let mut fields: Vec<Arc<Field>> = batch.schema().fields().to_vec();
            fields.push(Arc::new(Field::new("provenance", DataType::Binary, true)));
            let new_schema = Arc::new(Schema::new(fields));

            // Build new columns.
            let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
                .map(|i| batch.column(i).clone())
                .collect();
            columns.push(prov_array);

            let new_batch = RecordBatch::try_new(new_schema, columns).map_err(AnamError::Arrow)?;
            result.push(new_batch);
        }

        Ok(result)
    }

    fn build_reasoning_tree(&self, batches: &[RecordBatch]) -> Result<Option<String>> {
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
                        let valid = nulls.is_none_or(|n| n.is_valid(row));
                        if valid {
                            let bytes = binary_arr.value(row);
                            match PolynomialSemiring::from_bytes(bytes) {
                                Ok(poly) => {
                                    tree.push_str(&format!("  row {row}: {}\n", poly.explain()));
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
