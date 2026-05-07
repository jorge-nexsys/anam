//! NL-to-Datalog compiler.
//!
//! Translates natural-language constraint descriptions into valid Datalog
//! rules by calling an LLM API (OpenAI-compatible).

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use crate::core::error::{AnamError, Result};

/// Default LLM endpoint (OpenAI-compatible).
const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";
/// Default model for NL-to-Datalog compilation.
const DEFAULT_MODEL: &str = "gpt-4o";

/// Compiles natural-language descriptions into Datalog rules via an LLM.
#[derive(Debug)]
pub struct NlCompiler {
    api_key: Option<String>,
    endpoint: String,
    model: String,
    client: Client,
}

impl NlCompiler {
    /// Create a new compiler.
    ///
    /// If no API key is provided, `compile()` will return an error explaining
    /// that the NL compiler is unconfigured.
    pub fn new(
        api_key: Option<String>,
        endpoint: Option<String>,
        model: Option<String>,
    ) -> Self {
        Self {
            api_key,
            endpoint: endpoint.unwrap_or_else(|| DEFAULT_ENDPOINT.to_string()),
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            client: Client::new(),
        }
    }

    /// Compile a natural-language description into a Datalog rule.
    ///
    /// # Arguments
    /// * `nl` — The natural-language constraint or rule description.
    /// * `table` — The primary table this rule operates over.
    #[instrument(skip(self))]
    pub async fn compile(&self, nl: &str, table: &str) -> Result<String> {
        let api_key = self.api_key.as_deref().ok_or_else(|| {
            AnamError::NlCompilation(
                "NL compiler is not configured — set `llm_api_key` in SessionConfig".into(),
            )
        })?;

        info!(nl, table, model = %self.model, "compiling NL → Datalog");

        let system_prompt = format!(
            r#"You are an expert Datalog compiler for AnamDB, a neurosymbolic database engine.

Your job is to translate natural-language constraints into valid Datalog rules that
can be evaluated by the Scallop runtime.

Rules:
1. Output ONLY the Datalog rule — no explanations, no markdown.
2. Use Scallop syntax: `head(vars) :- body_atom_1, body_atom_2, condition.`
3. The primary input relation is `{table}`.
4. Column references use the pattern `VAR.column_name`.
5. Supported comparison operators: >, <, >=, <=, =, !=.
6. String literals are single-quoted: 'value'.
7. If multiple conditions are specified, they must ALL be true (conjunction).
8. The output relation name should be descriptive of the constraint.

Examples:
  NL: "Flag a transaction as high risk if fraud probability > 90% and amount > $10k"
  Table: transactions
  Output: high_risk(X) :- transactions(X), X.fraud_prob > 0.90, X.amount > 10000.

  NL: "Identify EU customers with more than 5 orders"
  Table: customers
  Output: eu_high_volume(X) :- customers(X), X.region = 'EU', X.order_count > 5.
"#
        );

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user".into(),
                    content: format!("NL: \"{nl}\"\nTable: {table}"),
                },
            ],
            temperature: 0.0,
            max_tokens: 256,
        };

        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AnamError::NlCompilation(format!(
                "LLM API returned {status}: {body}"
            )));
        }

        let completion: ChatCompletionResponse = response.json().await?;

        let datalog = completion
            .choices
            .first()
            .map(|c| c.message.content.trim().to_string())
            .ok_or_else(|| AnamError::NlCompilation("LLM returned no choices".into()))?;

        debug!(datalog = %datalog, "LLM generated Datalog");

        // Validate the generated Datalog before returning.
        self.validate_output(&datalog)?;

        Ok(datalog)
    }

    /// Basic validation of LLM-generated Datalog.
    fn validate_output(&self, datalog: &str) -> Result<()> {
        let trimmed = datalog.trim();

        if trimmed.is_empty() {
            return Err(AnamError::NlCompilation("LLM returned empty output".into()));
        }

        // Must contain either a rule structure or comparison operators.
        if !trimmed.contains(":-")
            && !trimmed.contains('>')
            && !trimmed.contains('<')
            && !trimmed.contains('=')
        {
            return Err(AnamError::NlCompilation(format!(
                "LLM output does not look like valid Datalog: '{trimmed}'"
            )));
        }

        // Reject markdown fences that the LLM might sneak in.
        if trimmed.contains("```") {
            return Err(AnamError::NlCompilation(
                "LLM output contains markdown fences — stripping".into(),
            ));
        }

        Ok(())
    }
}

// ── OpenAI-compatible API types ────────────────────────────────────────────

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
