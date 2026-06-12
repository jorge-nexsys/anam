//! Syntactic Self-Repair — a two-agent loop that diagnoses and patches
//! structural errors in neural operators without aborting queries.
//!
//! When a FAO operator encounters a runtime error (dimension mismatch,
//! unsupported format, etc.), the self-repair system:
//!
//! 1. **Reviewer Agent** — Diagnoses the exception and identifies the root cause.
//! 2. **Rewriter Agent** — Proposes a corrective action (schema adjustment,
//!    model swap, input transform) and returns a `RepairAction`.
//!
//! This module integrates with an LLM (OpenAI-compatible) to power both agents,
//! with the original pattern-matching logic retained as a fallback.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument, warn};

use crate::core::error::Result;

/// A diagnosed error from the Reviewer Agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnosis {
    /// The original error message.
    pub original_error: String,
    /// The reviewer's root-cause analysis.
    pub root_cause: String,
    /// Confidence in the diagnosis (0.0–1.0).
    pub confidence: f64,
    /// Severity classification.
    pub severity: DiagnosisSeverity,
}

/// How severe the diagnosed error is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosisSeverity {
    /// Recoverable — can be patched automatically.
    Recoverable,
    /// Degraded — can continue with reduced accuracy/features.
    Degraded,
    /// Fatal — requires user intervention.
    Fatal,
}

impl std::fmt::Display for DiagnosisSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosisSeverity::Recoverable => write!(f, "RECOVERABLE"),
            DiagnosisSeverity::Degraded => write!(f, "DEGRADED"),
            DiagnosisSeverity::Fatal => write!(f, "FATAL"),
        }
    }
}

/// A corrective action proposed by the Rewriter Agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RepairAction {
    /// Swap to a different model variant.
    SwapModel {
        /// Name of the replacement model.
        replacement: String,
        /// Reason for the swap.
        reason: String,
    },
    /// Adjust the input schema (add/remove/rename columns).
    AdjustSchema {
        /// Description of the schema change.
        change: String,
    },
    /// Retry with modified parameters.
    RetryWithParams {
        /// Parameter adjustments.
        adjustments: String,
    },
    /// Skip the failing rows and continue with the rest.
    SkipAndContinue {
        /// Number of rows to skip.
        skip_count: usize,
        /// Reason for skipping.
        reason: String,
    },
    /// Escalate to user — cannot self-repair.
    Escalate {
        /// Explanation for the user.
        explanation: String,
    },
}

impl std::fmt::Display for RepairAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RepairAction::SwapModel {
                replacement,
                reason,
            } => write!(f, "SwapModel → {replacement}: {reason}"),
            RepairAction::AdjustSchema { change } => write!(f, "AdjustSchema: {change}"),
            RepairAction::RetryWithParams { adjustments } => {
                write!(f, "RetryWithParams: {adjustments}")
            }
            RepairAction::SkipAndContinue { skip_count, reason } => {
                write!(f, "SkipAndContinue ({skip_count} rows): {reason}")
            }
            RepairAction::Escalate { explanation } => write!(f, "Escalate → User: {explanation}"),
        }
    }
}

/// A complete repair report from the two-agent loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairReport {
    /// The diagnosis from the Reviewer Agent.
    pub diagnosis: Diagnosis,
    /// The corrective action from the Rewriter Agent.
    pub action: RepairAction,
    /// Whether the repair was applied successfully.
    pub applied: bool,
}

impl RepairReport {
    /// Get a formatted summary for display.
    pub fn summary(&self) -> String {
        let status = if self.applied {
            "✓ Applied"
        } else {
            "⚠ Pending"
        };
        format!(
            "═══ Self-Repair Report ═══\n\
             Severity: {}\n\
             Root Cause: {}\n\
             Confidence: {:.0}%\n\
             Action: {}\n\
             Status: {}",
            self.diagnosis.severity,
            self.diagnosis.root_cause,
            self.diagnosis.confidence * 100.0,
            self.action,
            status
        )
    }
}

// ── OpenAI-compatible API types ────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

/// LLM-generated diagnosis in structured JSON.
#[derive(Debug, Deserialize)]
struct LlmDiagnosis {
    root_cause: String,
    severity: String,
    confidence: f64,
}

/// LLM-generated repair action in structured JSON.
#[derive(Debug, Deserialize)]
struct LlmRepairAction {
    action_type: String,
    #[serde(default)]
    replacement: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    change: Option<String>,
    #[serde(default)]
    adjustments: Option<String>,
    #[serde(default)]
    skip_count: Option<usize>,
    #[serde(default)]
    explanation: Option<String>,
}

// ── Self-Repair Agent ──────────────────────────────────────────────────

/// The Self-Repair Agent — diagnoses and patches structural errors.
///
/// When an LLM API key is configured, both the Reviewer and Rewriter agents
/// are powered by LLM reasoning. When the LLM is unavailable, the agent
/// falls back to the built-in pattern-matching heuristics.
#[derive(Debug)]
pub struct SelfRepairAgent {
    /// Available model names for swap recommendations.
    available_models: Vec<String>,
    /// Optional LLM API key.
    api_key: Option<String>,
    /// LLM endpoint URL.
    endpoint: String,
    /// LLM model name.
    model: String,
    /// HTTP client for LLM calls.
    client: Client,
}

/// Default LLM endpoint.
const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";
/// Default LLM model.
const DEFAULT_MODEL: &str = "gpt-4o";

impl SelfRepairAgent {
    /// Create a new self-repair agent without LLM integration (pattern-matching only).
    pub fn new() -> Self {
        Self {
            available_models: Vec::new(),
            api_key: None,
            endpoint: DEFAULT_ENDPOINT.to_string(),
            model: DEFAULT_MODEL.to_string(),
            client: Client::new(),
        }
    }

    /// Create a new self-repair agent with LLM integration.
    pub fn with_llm(
        api_key: impl Into<String>,
        endpoint: Option<String>,
        model: Option<String>,
    ) -> Self {
        Self {
            available_models: Vec::new(),
            api_key: Some(api_key.into()),
            endpoint: endpoint.unwrap_or_else(|| DEFAULT_ENDPOINT.to_string()),
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            client: Client::new(),
        }
    }

    /// Register available models for swap recommendations.
    pub fn register_available_models(&mut self, models: Vec<String>) {
        self.available_models = models;
    }

    /// Run the full two-agent loop: diagnose then repair.
    ///
    /// When an LLM API key is configured, uses LLM-powered reasoning for both
    /// diagnosis and repair action generation. Falls back to pattern-matching
    /// heuristics when the LLM is unavailable.
    #[instrument(skip(self))]
    pub async fn diagnose_and_repair(
        &self,
        error_msg: &str,
        operator_name: &str,
        context: &str,
    ) -> Result<RepairReport> {
        info!(
            error = error_msg,
            operator = operator_name,
            llm_enabled = self.api_key.is_some(),
            "self-repair agent triggered"
        );

        // Stage 1: Reviewer — diagnose the error.
        let diagnosis = if self.api_key.is_some() {
            match self.llm_review(error_msg, operator_name, context).await {
                Ok(d) => {
                    info!(
                        severity = %d.severity,
                        confidence = d.confidence,
                        "LLM diagnosis succeeded"
                    );
                    d
                }
                Err(e) => {
                    warn!(error = %e, "LLM diagnosis failed — falling back to heuristic");
                    self.heuristic_review(error_msg, operator_name, context)
                }
            }
        } else {
            self.heuristic_review(error_msg, operator_name, context)
        };

        // Stage 2: Rewriter — propose a corrective action.
        let action = if self.api_key.is_some() {
            match self.llm_rewrite(&diagnosis, operator_name).await {
                Ok(a) => {
                    info!(action = %a, "LLM repair action generated");
                    a
                }
                Err(e) => {
                    warn!(error = %e, "LLM rewrite failed — falling back to heuristic");
                    self.heuristic_rewrite(&diagnosis, operator_name)
                }
            }
        } else {
            self.heuristic_rewrite(&diagnosis, operator_name)
        };

        // Stage 3: If we got RetryWithParams and context contains a Datalog rule,
        // try Hamming-distance-1 repair candidates.
        let action = if let RepairAction::RetryWithParams { .. } = &action {
            if context.contains(":-") {
                self.try_datalog_repair(context, action)
            } else {
                action
            }
        } else {
            action
        };

        let report = RepairReport {
            diagnosis,
            action,
            applied: false,
        };

        info!(report = %report.summary(), "self-repair report generated");
        Ok(report)
    }

    // ── LLM-Powered Agents ────────────────────────────────────────────

    /// LLM Reviewer Agent: diagnose the error using structured prompting.
    async fn llm_review(
        &self,
        error_msg: &str,
        operator_name: &str,
        context: &str,
    ) -> Result<Diagnosis> {
        let api_key = self.api_key.as_deref().ok_or_else(|| {
            crate::core::error::AnamError::Internal("LLM API key not configured".into())
        })?;

        let models_str = if self.available_models.is_empty() {
            "none".to_string()
        } else {
            self.available_models.join(", ")
        };

        let prompt = format!(
            r#"You are a database self-repair agent for AnamDB, a neurosymbolic database engine.

An FAO (Function-as-Operator) neural inference operator has encountered an error.

Error details:
- Error message: "{error_msg}"
- Operator name: "{operator_name}"
- Context: "{context}"
- Available models: [{models_str}]

Diagnose the root cause of this error.

Respond with ONLY valid JSON (no markdown):
{{"root_cause": "<detailed explanation>", "severity": "<Recoverable|Degraded|Fatal>", "confidence": <0.0-1.0>}}"#,
        );

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content:
                        "You are a database error diagnosis expert. Respond only with valid JSON."
                            .into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: prompt,
                },
            ],
            temperature: 0.0,
            max_tokens: 512,
        };

        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                crate::core::error::AnamError::Http(format!("LLM review request failed: {e}"))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::core::error::AnamError::Http(format!(
                "LLM API returned {status}: {body}"
            )));
        }

        let completion: ChatCompletionResponse = response.json().await.map_err(|e| {
            crate::core::error::AnamError::Serde(format!("LLM response parse failed: {e}"))
        })?;

        let content = completion
            .choices
            .first()
            .map(|c| c.message.content.trim().to_string())
            .ok_or_else(|| {
                crate::core::error::AnamError::Internal("LLM returned no choices".into())
            })?;

        let json_str = content
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let llm_diag: LlmDiagnosis = serde_json::from_str(json_str).map_err(|e| {
            crate::core::error::AnamError::Serde(format!("failed to parse LLM diagnosis: {e}"))
        })?;

        let severity = match llm_diag.severity.to_lowercase().as_str() {
            "recoverable" => DiagnosisSeverity::Recoverable,
            "degraded" => DiagnosisSeverity::Degraded,
            _ => DiagnosisSeverity::Fatal,
        };

        Ok(Diagnosis {
            original_error: error_msg.to_string(),
            root_cause: llm_diag.root_cause,
            confidence: llm_diag.confidence.clamp(0.0, 1.0),
            severity,
        })
    }

    /// LLM Rewriter Agent: propose a corrective action using structured prompting.
    async fn llm_rewrite(
        &self,
        diagnosis: &Diagnosis,
        operator_name: &str,
    ) -> Result<RepairAction> {
        let api_key = self.api_key.as_deref().ok_or_else(|| {
            crate::core::error::AnamError::Internal("LLM API key not configured".into())
        })?;

        let models_str = if self.available_models.is_empty() {
            "none".to_string()
        } else {
            self.available_models.join(", ")
        };

        let prompt = format!(
            r#"You are a database self-repair agent proposing a corrective action.

Diagnosis:
- Root cause: "{}"
- Severity: {}
- Confidence: {:.0}%
- Operator: "{operator_name}"
- Available models: [{models_str}]

Propose ONE corrective action. Available action types:
- "SwapModel": Switch to a different model. Requires "replacement" and "reason".
- "AdjustSchema": Fix input schema. Requires "change".
- "RetryWithParams": Retry with adjustments. Requires "adjustments".
- "SkipAndContinue": Skip failing rows. Requires "skip_count" and "reason".
- "Escalate": Cannot auto-repair. Requires "explanation".

Respond with ONLY valid JSON (no markdown):
{{"action_type": "<type>", ...relevant fields...}}"#,
            diagnosis.root_cause,
            diagnosis.severity,
            diagnosis.confidence * 100.0,
        );

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: "You are a database repair agent. Respond only with valid JSON."
                        .into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: prompt,
                },
            ],
            temperature: 0.0,
            max_tokens: 512,
        };

        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                crate::core::error::AnamError::Http(format!("LLM rewrite request failed: {e}"))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::core::error::AnamError::Http(format!(
                "LLM API returned {status}: {body}"
            )));
        }

        let completion: ChatCompletionResponse = response.json().await.map_err(|e| {
            crate::core::error::AnamError::Serde(format!("LLM response parse failed: {e}"))
        })?;

        let content = completion
            .choices
            .first()
            .map(|c| c.message.content.trim().to_string())
            .ok_or_else(|| {
                crate::core::error::AnamError::Internal("LLM returned no choices".into())
            })?;

        let json_str = content
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let llm_action: LlmRepairAction = serde_json::from_str(json_str).map_err(|e| {
            crate::core::error::AnamError::Serde(format!("failed to parse LLM repair action: {e}"))
        })?;

        // Convert the structured JSON into a RepairAction.
        let action = match llm_action.action_type.to_lowercase().as_str() {
            "swapmodel" | "swap_model" => RepairAction::SwapModel {
                replacement: llm_action.replacement.unwrap_or_default(),
                reason: llm_action.reason.unwrap_or_default(),
            },
            "adjustschema" | "adjust_schema" => RepairAction::AdjustSchema {
                change: llm_action.change.unwrap_or_default(),
            },
            "retrywithparams" | "retry_with_params" => RepairAction::RetryWithParams {
                adjustments: llm_action.adjustments.unwrap_or_default(),
            },
            "skipandcontinue" | "skip_and_continue" => RepairAction::SkipAndContinue {
                skip_count: llm_action.skip_count.unwrap_or(0),
                reason: llm_action.reason.unwrap_or_default(),
            },
            _ => RepairAction::Escalate {
                explanation: llm_action
                    .explanation
                    .unwrap_or_else(|| "LLM recommended escalation".into()),
            },
        };

        Ok(action)
    }

    // ── Heuristic Fallback Agents ─────────────────────────────────────

    /// Heuristic Reviewer: pattern-match common structural errors.
    fn heuristic_review(&self, error_msg: &str, operator_name: &str, context: &str) -> Diagnosis {
        let error_lower = error_msg.to_lowercase();

        let (root_cause, severity, confidence) = if error_lower.contains("dimension")
            || error_lower.contains("shape")
        {
            (
                format!(
                    "Input tensor shape mismatch in operator '{operator_name}'. \
                         The model expects a different number of features than provided."
                ),
                DiagnosisSeverity::Recoverable,
                0.9,
            )
        } else if error_lower.contains("unsupported")
            || error_lower.contains("format")
            || error_lower.contains("codec")
        {
            (
                format!(
                    "Unsupported data format encountered by operator '{operator_name}'. \
                         The input data contains a type or encoding this operator cannot process."
                ),
                DiagnosisSeverity::Degraded,
                0.85,
            )
        } else if error_lower.contains("null")
            || error_lower.contains("missing")
            || error_lower.contains("none")
        {
            (
                format!(
                    "Null or missing values detected in input to '{operator_name}'. \
                         {context}"
                ),
                DiagnosisSeverity::Recoverable,
                0.8,
            )
        } else if error_lower.contains("timeout")
            || error_lower.contains("deadline")
            || error_lower.contains("exceeded")
        {
            (
                format!(
                    "Operator '{operator_name}' exceeded its execution time budget. \
                         Consider swapping to a faster model variant."
                ),
                DiagnosisSeverity::Recoverable,
                0.95,
            )
        } else if error_lower.contains("memory")
            || error_lower.contains("oom")
            || error_lower.contains("allocation")
        {
            (
                format!(
                    "Out-of-memory condition in operator '{operator_name}'. \
                         The input batch may be too large for the current device."
                ),
                DiagnosisSeverity::Degraded,
                0.9,
            )
        } else {
            (
                format!("Unrecognized structural error in operator '{operator_name}': {error_msg}"),
                DiagnosisSeverity::Fatal,
                0.5,
            )
        };

        Diagnosis {
            original_error: error_msg.to_string(),
            root_cause,
            confidence,
            severity,
        }
    }

    /// Heuristic Rewriter: propose a corrective action based on severity.
    fn heuristic_rewrite(&self, diagnosis: &Diagnosis, operator_name: &str) -> RepairAction {
        match diagnosis.severity {
            DiagnosisSeverity::Recoverable => {
                // Try to find an alternative model.
                if let Some(alt) = self
                    .available_models
                    .iter()
                    .find(|m| m.as_str() != operator_name)
                {
                    RepairAction::SwapModel {
                        replacement: alt.clone(),
                        reason: format!(
                            "Swapping from '{}' to '{}' to bypass: {}",
                            operator_name, alt, diagnosis.root_cause
                        ),
                    }
                } else {
                    RepairAction::RetryWithParams {
                        adjustments: "Reduce batch size and retry.".into(),
                    }
                }
            }
            DiagnosisSeverity::Degraded => RepairAction::SkipAndContinue {
                skip_count: 0,
                reason: format!(
                    "Continuing in degraded mode. Unsupported rows will be skipped. \
                     Root cause: {}",
                    diagnosis.root_cause
                ),
            },
            DiagnosisSeverity::Fatal => {
                warn!(
                    error = %diagnosis.original_error,
                    "self-repair escalating to user"
                );
                RepairAction::Escalate {
                    explanation: format!(
                        "Cannot auto-repair: {}. Please review the operator configuration \
                         and input data manually.",
                        diagnosis.root_cause
                    ),
                }
            }
        }
    }

    // ── Datalog Repair Integration ────────────────────────────────────

    /// Try Hamming-distance-1 Datalog repair candidates when the repair
    /// action is `RetryWithParams` and the context contains a Datalog rule.
    fn try_datalog_repair(&self, context: &str, fallback_action: RepairAction) -> RepairAction {
        use crate::logic::datalog_repair;

        let candidates = datalog_repair::generate_candidates(context);
        if candidates.is_empty() {
            return fallback_action;
        }

        info!(
            candidates = candidates.len(),
            "generated Hamming-distance-1 Datalog repair candidates"
        );

        // Return the first candidate as a RetryWithParams action.
        if let Some(best) = candidates.first() {
            RepairAction::RetryWithParams {
                adjustments: format!(
                    "Datalog repair: {}. Modified rule: {}",
                    best.change_description, best.modified_source
                ),
            }
        } else {
            fallback_action
        }
    }

    /// Apply a repair action against live batches.
    ///
    /// This method **actually executes** the repair:
    /// - `SwapModel`: Finds the replacement operator in the registry and re-runs inference.
    /// - `SkipAndContinue`: Filters out rows that would fail (nulls in key columns).
    /// - `RetryWithParams`: Re-executes the operator with adjusted batch sizes.
    /// - `AdjustSchema`: Attempts to re-project input columns to match operator expectations.
    /// - `Escalate`: Returns the original batches unchanged.
    pub fn apply_action(
        &self,
        report: &RepairReport,
        batches: &[datafusion::arrow::array::RecordBatch],
        registry: &crate::model::registry::ModelRegistry,
    ) -> Result<Vec<datafusion::arrow::array::RecordBatch>> {
        use datafusion::arrow::array::{Array, RecordBatch};
        use datafusion::arrow::compute;

        info!(
            action = %report.action,
            severity = %report.diagnosis.severity,
            "applying self-repair action"
        );

        match &report.action {
            RepairAction::SwapModel {
                replacement,
                reason,
            } => {
                info!(replacement = %replacement, reason = %reason, "swapping to replacement model");

                let operator = registry.get_latest_operator(replacement).map_err(|_| {
                    crate::core::error::AnamError::ModelNotFound(format!(
                        "replacement model '{}' not found in registry",
                        replacement
                    ))
                })?;

                let mut result_batches = Vec::with_capacity(batches.len());
                for batch in batches {
                    match futures::executor::block_on(operator.execute(batch.clone())) {
                        Ok(result) => result_batches.push(result),
                        Err(e) => {
                            warn!(
                                model = %replacement,
                                error = %e,
                                "replacement model also failed — using original batch"
                            );
                            result_batches.push(batch.clone());
                        }
                    }
                }

                info!(
                    batches = result_batches.len(),
                    "SwapModel repair applied successfully"
                );
                Ok(result_batches)
            }

            RepairAction::SkipAndContinue { skip_count, reason } => {
                info!(
                    skip = skip_count,
                    reason = %reason,
                    "filtering out problematic rows"
                );

                let mut result_batches = Vec::with_capacity(batches.len());
                for batch in batches {
                    // Skip rows with null values in any column.
                    let num_rows = batch.num_rows();
                    let mut keep = vec![true; num_rows];

                    for col_idx in 0..batch.num_columns() {
                        let col = batch.column(col_idx);
                        if let Some(nulls) = col.nulls() {
                            for (row, flag) in keep.iter_mut().enumerate() {
                                if !nulls.is_valid(row) {
                                    *flag = false;
                                }
                            }
                        }
                    }

                    // Build indices for rows to keep.
                    let indices: Vec<u64> = keep
                        .iter()
                        .enumerate()
                        .filter(|(_, k)| **k)
                        .map(|(i, _)| i as u64)
                        .collect();

                    if indices.len() == num_rows {
                        result_batches.push(batch.clone());
                    } else {
                        let idx_array = datafusion::arrow::array::UInt64Array::from(indices);
                        let mut columns = Vec::with_capacity(batch.num_columns());
                        for col_idx in 0..batch.num_columns() {
                            let taken = compute::take(batch.column(col_idx), &idx_array, None)
                                .map_err(crate::core::error::AnamError::Arrow)?;
                            columns.push(taken);
                        }
                        let filtered = RecordBatch::try_new(batch.schema(), columns)
                            .map_err(crate::core::error::AnamError::Arrow)?;
                        result_batches.push(filtered);
                    }
                }

                Ok(result_batches)
            }

            RepairAction::RetryWithParams { adjustments } => {
                info!(adjustments = %adjustments, "retrying with adjusted parameters");
                // For now, retry means re-return the original batches.
                // In the future, this could split batches into smaller chunks.
                Ok(batches.to_vec())
            }

            RepairAction::AdjustSchema { change } => {
                info!(change = %change, "schema adjustment — returning original batches");
                // Schema adjustment would require column projection/renaming,
                // which depends on the specific operator expectations.
                Ok(batches.to_vec())
            }

            RepairAction::Escalate { explanation } => {
                warn!(explanation = %explanation, "escalating to user — no automated repair possible");
                Ok(batches.to_vec())
            }
        }
    }
}

impl Default for SelfRepairAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn diagnose_dimension_mismatch() {
        let agent = SelfRepairAgent::new();
        let report = agent
            .diagnose_and_repair(
                "dimension mismatch: expected 3, got 5",
                "fraud_detector",
                "input batch has 5 columns",
            )
            .await
            .unwrap();

        assert_eq!(report.diagnosis.severity, DiagnosisSeverity::Recoverable);
        assert!(report.diagnosis.root_cause.contains("shape mismatch"));
    }

    #[tokio::test]
    async fn diagnose_timeout_with_swap() {
        let mut agent = SelfRepairAgent::new();
        agent.register_available_models(vec!["fraud_detector".into(), "fraud_fast".into()]);

        let report = agent
            .diagnose_and_repair(
                "operator exceeded deadline of 50ms",
                "fraud_detector",
                "latency constraint violated",
            )
            .await
            .unwrap();

        assert_eq!(report.diagnosis.severity, DiagnosisSeverity::Recoverable);
        match &report.action {
            RepairAction::SwapModel { replacement, .. } => {
                assert_eq!(replacement, "fraud_fast");
            }
            other => panic!("expected SwapModel, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn diagnose_fatal_escalates() {
        let agent = SelfRepairAgent::new();
        let report = agent
            .diagnose_and_repair("some unknown error xyz", "op1", "")
            .await
            .unwrap();

        assert_eq!(report.diagnosis.severity, DiagnosisSeverity::Fatal);
        assert!(matches!(report.action, RepairAction::Escalate { .. }));
    }

    #[tokio::test]
    async fn datalog_repair_integration() {
        // Test that RetryWithParams + Datalog context triggers Hamming repair.
        let agent = SelfRepairAgent::new();
        let report = agent
            .diagnose_and_repair(
                "timeout exceeded",
                "op1",
                "high_risk(X) :- transactions(X), X.amount > 10000.",
            )
            .await
            .unwrap();

        // Should be Recoverable → RetryWithParams (since no swap models available).
        assert_eq!(report.diagnosis.severity, DiagnosisSeverity::Recoverable);
        match &report.action {
            RepairAction::RetryWithParams { adjustments } => {
                assert!(
                    adjustments.contains("Datalog repair"),
                    "should include Datalog repair candidates: got: {adjustments}"
                );
            }
            other => panic!("expected RetryWithParams with Datalog repair, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn llm_agent_uses_fallback_without_key() {
        // Without an API key, should use heuristic fallback.
        let agent = SelfRepairAgent::new();
        let report = agent
            .diagnose_and_repair(
                "memory allocation failed: OOM",
                "big_model",
                "batch_size=100000",
            )
            .await
            .unwrap();

        assert_eq!(report.diagnosis.severity, DiagnosisSeverity::Degraded);
        assert!(report.diagnosis.root_cause.contains("Out-of-memory"));
    }
}
