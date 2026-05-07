//! `anam` — interactive CLI / REPL for AnamDB.


use anyhow::Result;
use datafusion::arrow::util::pretty::pretty_format_batches;
use clap::Parser;
use comfy_table::{Cell, Table};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use tracing_subscriber::EnvFilter;

use anamdb::core::provenance::ProvenanceMode;
use anamdb::core::session::{QueryResult, Session, SessionConfig};

/// AnamDB interactive CLI.
#[derive(Parser, Debug)]
#[command(name = "anam", version, about = "AnamDB — the AI-native logic kernel")]
struct Cli {
    /// Enable GPU / NPU hardware acceleration.
    #[arg(long, default_value_t = false)]
    gpu: bool,

    /// Provenance mode: boolean, probability, polynomial.
    #[arg(long, default_value = "polynomial")]
    provenance: String,

    /// LLM API key for NL-to-Datalog compilation.
    #[arg(long, env = "ANAM_LLM_API_KEY")]
    llm_api_key: Option<String>,

    /// LLM endpoint URL.
    #[arg(long, env = "ANAM_LLM_ENDPOINT")]
    llm_endpoint: Option<String>,

    /// LLM model name.
    #[arg(long, env = "ANAM_LLM_MODEL", default_value = "gpt-4o")]
    llm_model: String,

    /// Execute a single SQL query and exit.
    #[arg(short = 'e', long)]
    execute: Option<String>,

    /// Log level (trace, debug, info, warn, error).
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file (ignore if missing).
    dotenvy::dotenv().ok();

    // Map OPENAI_API_KEY → ANAM_LLM_API_KEY if the latter isn't already set.
    // SAFETY: called before any threads are spawned.
    if std::env::var("ANAM_LLM_API_KEY").is_err() {
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            unsafe { std::env::set_var("ANAM_LLM_API_KEY", &key); }
        }
    }

    let cli = Cli::parse();

    // Initialize tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&cli.log_level)),
        )
        .with_target(false)
        .init();

    // Parse provenance mode.
    let provenance_mode = match cli.provenance.to_lowercase().as_str() {
        "boolean" | "bool" => ProvenanceMode::Boolean,
        "probability" | "prob" => ProvenanceMode::Probability,
        "polynomial" | "poly" => ProvenanceMode::Polynomial,
        other => {
            eprintln!("Unknown provenance mode: {other}. Using polynomial.");
            ProvenanceMode::Polynomial
        }
    };

    // Build session config.
    let config = SessionConfig {
        provenance_mode,
        enable_hardware_accel: cli.gpu,
        llm_api_key: cli.llm_api_key,
        llm_endpoint: cli.llm_endpoint,
        llm_model: Some(cli.llm_model),
        anomaly_threshold: 0.5,
    };

    let session = Session::with_config(config).await?;

    // Print banner.
    print_banner(&session);

    // Single-query mode.
    if let Some(query) = &cli.execute {
        let result = session.sql(query).await?;
        print_result(&result);
        return Ok(());
    }

    // Interactive REPL.
    repl(session).await
}

async fn repl(session: Session) -> Result<()> {
    let mut rl = DefaultEditor::new()?;
    let history_path = dirs_next::data_dir()
        .map(|d| d.join("anamdb").join("history.txt"))
        .unwrap_or_else(|| ".anam_history".into());

    if let Some(parent) = history_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let _ = rl.load_history(&history_path);

    println!("\nType SQL queries, or use dot-commands:");
    println!("  .load <path>    — Register a Lance table (streaming)");
    println!("  .ingest <csv>   — Ingest CSV → Lance dataset");
    println!("  .logic <n> <d>  — Register a Datalog rule");
    println!("  .models         — List registered AI models");
    println!("  .rules          — List Datalog rules");
    println!("  .devices        — List available compute devices");
    println!("  .explain        — Explain the last query's reasoning");
    println!("  .help           — Show all commands");
    println!("  .quit           — Exit\n");

    let mut last_result: Option<QueryResult> = None;

    loop {
        let prompt = "anam> ";
        match rl.readline(prompt) {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(line);

                if line.starts_with('.') {
                    // Dot-command.
                    match handle_dot_command(line, &session, &last_result).await {
                        Ok(true) => break, // .quit
                        Ok(false) => continue,
                        Err(e) => {
                            eprintln!("Error: {e}");
                            continue;
                        }
                    }
                }

                // SQL query.
                match session.sql(line).await {
                    Ok(result) => {
                        print_result(&result);

                        // HITL: if anomalies were detected, show them.
                        if result.requires_clarification() {
                            println!("\n⚠️  Semantic anomalies detected:");
                            for anomaly in &result.anomalies {
                                println!("  {anomaly}");
                            }
                            println!(
                                "\nUse `.refine <correction>` to provide feedback, or \
                                 `.accept` to proceed."
                            );
                        }

                        last_result = Some(result);
                    }
                    Err(e) => {
                        eprintln!("Query error: {e}");
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("Goodbye.");
                break;
            }
            Err(e) => {
                eprintln!("Readline error: {e}");
                break;
            }
        }
    }

    let _ = rl.save_history(&history_path);
    Ok(())
}

async fn handle_dot_command(
    cmd: &str,
    session: &Session,
    last_result: &Option<QueryResult>,
) -> Result<bool> {
    let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
    let command = parts[0].to_lowercase();
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match command.as_str() {
        ".quit" | ".exit" | ".q" => {
            println!("Goodbye.");
            return Ok(true);
        }

        ".load" => {
            if arg.is_empty() {
                println!("Usage: .load <path_to_lance_dataset> [table_name]");
            } else {
                let parts: Vec<&str> = arg.splitn(2, ' ').collect();
                let path = parts[0];
                let name = parts.get(1).copied().unwrap_or_else(|| {
                    std::path::Path::new(path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("table")
                });
                session.register_table(name, path).await?;
                println!("Registered table '{name}' from {path}");
            }
        }

        ".models" => {
            let models = session.models().list_models();
            if models.is_empty() {
                println!("No models registered.");
            } else {
                let mut table = Table::new();
                table.set_header(vec!["ID", "Name", "Version", "Format", "Latency (ms)", "Accuracy"]);
                for m in &models {
                    table.add_row(vec![
                        Cell::new(&m.model_id[..8]),
                        Cell::new(&m.name),
                        Cell::new(&m.version),
                        Cell::new(m.format.to_string()),
                        Cell::new(format!("{:.1}", m.avg_latency_ms)),
                        Cell::new(format!("{:.2}", m.accuracy)),
                    ]);
                }
                println!("{table}");
            }
        }

        ".operators" => {
            let ops = session.models().list_operators();
            if ops.is_empty() {
                println!("No FAO operators registered.");
            } else {
                let mut table = Table::new();
                table.set_header(vec!["Function", "Version", "Model", "Latency", "Accuracy"]);
                for op in &ops {
                    table.add_row(vec![
                        Cell::new(&op.function_id),
                        Cell::new(&op.version),
                        Cell::new(&op.model_id[..8.min(op.model_id.len())]),
                        Cell::new(format!("{:.1}ms", op.est_latency_ms)),
                        Cell::new(format!("{:.2}", op.est_accuracy)),
                    ]);
                }
                println!("{table}");
            }
        }

        ".rules" => {
            let rules = session.logic_engine().read().list_rules().iter().map(|r| {
                (r.name.clone(), r.datalog_source.clone())
            }).collect::<Vec<_>>();
            if rules.is_empty() {
                println!("No Datalog rules registered.");
            } else {
                let mut table = Table::new();
                table.set_header(vec!["Name", "Datalog Source"]);
                for (name, source) in &rules {
                    table.add_row(vec![
                        Cell::new(name),
                        Cell::new(source),
                    ]);
                }
                println!("{table}");
            }
        }

        ".devices" => {
            println!("═══ Device Pool ═══");
            println!("{}", session.device_pool().summary());
        }

        ".explain" => {
            println!("═══════════════════════════════════════════════════════════");
            println!("  AnamDB Reasoning Trace");
            println!("═══════════════════════════════════════════════════════════");
            println!();

            // 1. Provenance mode
            println!("─── Provenance ─────────────────────────────────────────");
            println!("  Mode: {:?}", session.config.provenance_mode);
            if let Some(result) = last_result {
                let total_rows: usize = result.batches.iter().map(|b| b.num_rows()).sum();
                println!("  Last query: {} batch(es), {} rows", result.batches.len(), total_rows);
                if let Some(tree) = &result.reasoning_tree {
                    println!();
                    println!("{tree}");
                }
            } else {
                println!("  (no query executed yet)");
            }

            // 2. Registered Datalog rules
            println!();
            println!("─── Datalog Rules ──────────────────────────────────────");
            let engine = session.logic_engine().read();
            let rules = engine.list_rules();
            if rules.is_empty() {
                println!("  (none)");
            } else {
                for rule in &rules {
                    println!("  • {} ← {}", rule.name, rule.datalog_source);
                }
            }
            drop(engine);

            // 3. Model catalog + Pareto frontier
            println!();
            println!("─── AI-Tables Catalog ──────────────────────────────────");
            let models = session.models().list_models();
            let operators = session.models().list_operators();
            if models.is_empty() {
                println!("  (no models registered)");
            } else {
                for m in &models {
                    println!(
                        "  • {} v{} [{}] — latency: {:.1}ms, accuracy: {:.2}, {}",
                        m.name, m.version, m.format,
                        m.avg_latency_ms, m.accuracy,
                        m.artifact_path
                    );
                }
            }

            if operators.len() >= 2 {
                println!();
                println!("─── Pareto Frontier ────────────────────────────────────");
                use anamdb::execution::optimizer::{ParetoOptimizer, CandidatePlan};
                let pool = session.device_pool();
                let device_mult = pool.speed_multiplier();
                let candidates: Vec<CandidatePlan> = operators.iter().map(|fao| {
                    CandidatePlan {
                        fao_ref: fao.clone(),
                        est_latency_ms: fao.est_latency_ms / device_mult,
                        est_accuracy: fao.est_accuracy,
                        est_cost: fao.est_latency_ms * 0.001 / device_mult,
                    }
                }).collect();

                // Check which dominate
                for (i, c) in candidates.iter().enumerate() {
                    let dominated = candidates.iter().enumerate().any(|(j, other)| {
                        i != j && other.dominates(c)
                    });
                    let label = if dominated { "  ✗ dominated" } else { "  ★ frontier " };
                    println!(
                        "{} {} v{}: latency={:.3}ms, accuracy={:.2}, cost={:.4}",
                        label, c.fao_ref.function_id, c.fao_ref.version,
                        c.est_latency_ms, c.est_accuracy, c.est_cost
                    );
                }
            }

            // 4. Device pool
            println!();
            println!("─── Device Pool ────────────────────────────────────────");
            println!("{}", session.device_pool().summary());

            // 5. Anomaly status
            if let Some(result) = last_result {
                if !result.anomalies.is_empty() {
                    println!();
                    println!("─── Anomalies ──────────────────────────────────────────");
                    for a in &result.anomalies {
                        println!("  ⚠ [{}] {}", a.severity, a.description);
                        println!("    → {}", a.suggested_action);
                    }
                }
            }

            println!();
            println!("═══════════════════════════════════════════════════════════");
        }

        ".help" | ".h" => {
            println!("Available commands:");
            println!("  .load <path> [name]          Register a Lance table");
            println!("  .ingest <csv> [lance]         Ingest CSV → Lance dataset");
            println!("  .model load <path> [name]     Load an ONNX model");
            println!("  .logic <name> <datalog>       Register a Datalog rule");
            println!("  .nl <name> <table> <desc>     NL → Datalog (via LLM)");
            println!("  .models                       List AI-Tables models");
            println!("  .operators                    List FAO operators");
            println!("  .rules                        List Datalog rules");
            println!("  .devices                      List compute devices");
            println!("  .explain                      Show reasoning tree");
            println!("  .quit                         Exit");
        }

        ".ingest" => {
            if arg.is_empty() {
                println!("Usage: .ingest <csv_path> [lance_output_path]");
            } else {
                let parts: Vec<&str> = arg.splitn(2, ' ').collect();
                let csv_path = parts[0];
                let lance_path = parts.get(1).copied().unwrap_or_else(|| {
                    csv_path
                });
                let lance_output = if lance_path.ends_with(".lance") {
                    lance_path.to_string()
                } else {
                    let stem = std::path::Path::new(lance_path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("data");
                    format!("{stem}.lance")
                };
                match anamdb::storage::ingestion::ingest_csv(csv_path, &lance_output).await {
                    Ok(()) => println!("✓ Ingested '{csv_path}' → '{lance_output}'"),
                    Err(e) => eprintln!("Ingestion error: {e}"),
                }
            }
        }

        ".logic" => {
            if arg.is_empty() {
                println!("Usage: .logic <rule_name> <datalog_source>");
            } else {
                let parts: Vec<&str> = arg.splitn(2, ' ').collect();
                if parts.len() < 2 {
                    println!("Usage: .logic <rule_name> <datalog_source>");
                } else {
                    let name = parts[0];
                    let source = parts[1].trim_matches('"');
                    match session.register_logic(name, source) {
                        Ok(()) => println!("✓ Registered rule '{name}'"),
                        Err(e) => eprintln!("Logic error: {e}"),
                    }
                }
            }
        }

        ".model" => {
            if arg.is_empty() {
                println!("Usage: .model load <onnx_path> [name] [features] [latency_ms] [accuracy]");
            } else {
                let parts: Vec<&str> = arg.splitn(6, ' ').collect();
                match parts[0] {
                    "load" => {
                        if parts.len() < 2 {
                            println!("Usage: .model load <onnx_path> [name] [features] [latency_ms] [accuracy]");
                        } else {
                            let model_path = parts[1];
                            let func_name = parts.get(2).copied().unwrap_or_else(|| {
                                std::path::Path::new(model_path)
                                    .file_stem()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("model")
                            });
                            let num_features: usize = parts.get(3)
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(3);
                            let latency: f64 = parts.get(4)
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(1.0);
                            let accuracy: f64 = parts.get(5)
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(0.95);

                            match session.load_onnx_model_with_metrics(
                                func_name, "1.0.0", model_path, func_name,
                                num_features, latency, accuracy,
                            ) {
                                Ok(id) => println!(
                                    "✓ Loaded ONNX model '{func_name}' (id: {}, latency: {latency}ms, accuracy: {accuracy})",
                                    &id[..8]
                                ),
                                Err(e) => eprintln!("Model error: {e}"),
                            }
                        }
                    }
                    _ => println!("Unknown model command. Use: .model load <path>"),
                }
            }
        }

        ".nl" => {
            if arg.is_empty() {
                println!("Usage: .nl <rule_name> <table> <natural language description>");
                println!("Example: .nl high_risk transactions Flag transactions with fraud probability above 90%");
            } else {
                let parts: Vec<&str> = arg.splitn(3, ' ').collect();
                if parts.len() < 3 {
                    println!("Usage: .nl <rule_name> <table> <description>");
                } else {
                    let name = parts[0];
                    let table = parts[1];
                    let description = parts[2];
                    println!("Compiling NL → Datalog via LLM...");
                    match session.register_logic_from_nl(name, table, description).await {
                        Ok(()) => {
                            // Show the generated rule.
                            let engine = session.logic_engine().read();
                            let rules = engine.list_rules();
                            if let Some(rule) = rules.iter().find(|r| r.name == name) {
                                println!("✓ Generated and registered rule '{name}':");
                                println!("  Datalog: {}", rule.datalog_source);
                            } else {
                                println!("✓ Registered rule '{name}'");
                            }
                        }
                        Err(e) => eprintln!("NL compilation error: {e}"),
                    }
                }
            }
        }

        _ => {
            println!("Unknown command: {command}. Type .help for available commands.");
        }
    }

    Ok(false)
}

fn print_result(result: &QueryResult) {
    if result.batches.is_empty() {
        println!("(no results)");
        return;
    }

    match pretty_format_batches(&result.batches) {
        Ok(formatted) => println!("{formatted}"),
        Err(e) => eprintln!("Failed to format results: {e}"),
    }

    let total_rows: usize = result.batches.iter().map(|b| b.num_rows()).sum();
    println!("({total_rows} rows)");
}

fn print_banner(session: &Session) {
    println!(r#"
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     █████╗ ███╗   ██╗ █████╗ ███╗   ███╗██████╗ ██████╗  ║
    ║    ██╔══██╗████╗  ██║██╔══██╗████╗ ████║██╔══██╗██╔══██╗ ║
    ║    ███████║██╔██╗ ██║███████║██╔████╔██║██║  ██║██████╔╝ ║
    ║    ██╔══██║██║╚██╗██║██╔══██║██║╚██╔╝██║██║  ██║██╔══██╗ ║
    ║    ██║  ██║██║ ╚████║██║  ██║██║ ╚═╝ ██║██████╔╝██████╔╝ ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚═════╝  ║
    ║                                                          ║
    ║    The AI-Native, Differentiable Logic Kernel            ║
    ║    v0.1.0-alpha                                          ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
"#);

    println!("  Provenance: {:?}", session.config.provenance_mode);
    println!("  Devices:    {} slots", session.device_pool().list_slots().len());
    println!("  Models:     {} registered", session.models().list_models().len());
    let llm_status = if session.config.llm_api_key.is_some() {
        format!("✓ connected ({})", session.config.llm_model.as_deref().unwrap_or("gpt-4o"))
    } else {
        "✗ not configured".to_string()
    };
    println!("  LLM:        {llm_status}");
}
