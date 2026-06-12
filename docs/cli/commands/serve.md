# CLI Command: serve

The `serve` subcommand boots a stateless AnamDB query engine server directly, bypassing workspace configurations.

## Usage

```bash
anam serve [OPTIONS]
```

---

## Options

* **`--port <IP:PORT>`**: Address to bind the server to (defaults to `0.0.0.0:8080`).
* **`--provenance <MODE>`**: Sets the default provenance calculation model. Choices: `boolean`, `probability`, `polynomial` (default: `polynomial`).
* **`--gpu`**: Force enable GPU hardware acceleration (Metal / CUDA / NPU) for neural execution.
* **`--log-level <LEVEL>`**: Logging verbosity. Options: `trace`, `debug`, `info`, `warn`, `error` (default: `info`).

---

## Example

```bash
$ anam serve --port 127.0.0.1:9000 --gpu
🚀 Starting AnamDB server on 127.0.0.1:9000...
```
