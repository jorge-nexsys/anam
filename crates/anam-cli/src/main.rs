//! `anam` — interactive CLI / REPL for AnamDB.

use std::sync::Arc;

use anyhow::Result;
use datafusion::arrow::util::pretty::pretty_format_batches;
use clap::Parser;
use comfy_table::{Cell, Color, Table};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use tracing::info;
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
    println!("  .load <path>    — Register a Lance table");
    println!("  .models         — List registered AI models");
    println!("  .operators      — List registered FAO operators");
    println!("  .rules          — List Datalog rules");
    println!("  .devices        — List available compute devices");
    println!("  .explain        — Explain the last query's reasoning");
    println!("  .help           — Show this help");
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
            let rules = session.logic_engine.read().list_rules().iter().map(|r| {
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
            println!("{}", session.device_pool.summary());
        }

        ".explain" => {
            if let Some(result) = last_result {
                result.explain_reasoning().await?;
            } else {
                println!("No query results to explain. Run a query first.");
            }
        }

        ".help" | ".h" => {
            println!("Available commands:");
            println!("  .load <path> [name]  Register a Lance table");
            println!("  .models              List AI-Tables models");
            println!("  .operators           List FAO operators");
            println!("  .rules               List Datalog rules");
            println!("  .devices             List compute devices");
            println!("  .explain             Show reasoning tree");
            println!("  .quit                Exit");
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
    println!("  Devices:    {} slots", session.device_pool.list_slots().len());
    println!("  Models:     {} registered", session.models().list_models().len());
}
