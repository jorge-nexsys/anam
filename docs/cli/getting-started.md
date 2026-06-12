# Getting Started with Anam CLI

The `anam` binary (compiled from the `anam-cli` crate) is the main developer tool for starting the AnamDB server, initializing new workspace projects, checking health, and running the interactive REPL.

---

## Installation

### From Cargo (Rust Package Manager)
If you have Rust installed, you can build and install the binary from source:

```bash
cargo install --git https://github.com/AnamDB/anam-db anam-cli
```

### Via Docker
If you prefer containerized execution, you can run the server directly:

```bash
docker run -p 8080:8080 ghcr.io/anamdb/anam-db
```

---

## Workspace Initialization

To structure a local project, create a directory and initialize it with:

```bash
mkdir my-anam-project && cd my-anam-project
anam init
```

This scaffolds the following layout in your workspace:

```
├── anamdb.toml      # Project configuration
├── catalog.json     # Registers loaded tables, models, and rules
├── .env.example     # Environment template for keys
├── queries/
│   └── example.sql  # Boilerplate query structure
├── tables/          # Local Lance files
├── models/          # Local ONNX model files
└── packs/           # Logic pack configurations
```

---

## Environment Configuration

Copy `.env.example` to `.env` to configure optional integrations like OpenAI for natural language translation:

```bash
# LLM API key for NL-to-Datalog compilation (optional)
ANAM_LLM_API_KEY=sk-your-openai-api-key
ANAM_LLM_ENDPOINT=https://api.openai.com/v1
ANAM_LLM_MODEL=gpt-4o
```

---

## Workspace Workflow

Once your project directory is set up, you can:
1. Start the server using the workspace config:
   ```bash
   anam start
   ```
2. Connect or run the interactive shell:
   ```bash
   anam
   ```
3. Check the status of a running node:
   ```bash
   anam status
   ```
