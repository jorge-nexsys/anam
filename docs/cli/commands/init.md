# CLI Command: init

The `init` subcommand initializes a new project directory structure for an AnamDB workspace.

## Usage

```bash
anam init [PATH]
```

* **`[PATH]`**: (Optional) The directory to scaffold. Defaults to the current directory (`.`).

---

## Scaffolding Output

Running this command creates the following structure:

1. **`anamdb.toml`**: The main configuration settings for the database engine.
2. **`catalog.json`**: An empty registration file database catalog.
3. **`queries/example.sql`**: A template demonstrating neurosymbolic querying.
4. **Folders**:
   * `tables/`: Location for Arrow-backed Lance datasets.
   * `models/`: Location for local ONNX model graphs.
   * `queries/`: For persistent query scripts.
   * `packs/`: For modular logic packs.

---

## Example

```bash
$ anam init my-project
📁 Initializing AnamDB project: my-project
  ✓ Created my-project/anamdb.toml
  ✓ Created my-project/.env.example
  ✓ Created my-project/queries/example.sql
  ✓ Created my-project/README.md
  ✓ Created my-project/catalog.json
  ✓ Created my-project/tables/
  ✓ Created my-project/models/
  ✓ Created my-project/queries/
  ✓ Created my-project/packs/

  Done! Next steps:
    cd my-project
    cp .env.example .env
    anam start
```
