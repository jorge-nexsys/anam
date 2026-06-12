# CLI Command: start

The `start` subcommand runs the AnamDB server using configuration files in your current workspace directory.

## Usage

```bash
anam start [OPTIONS]
```

---

## Options

* **`--port <PORT>`**: Overrides the host and port address defined in `anamdb.toml`.
* **`--config <PATH>`**: Specifies the path to the configuration file. Defaults to `anamdb.toml`.
* **`--gpu`**: Force enable GPU hardware acceleration (Metal / CUDA / NPU) for neural execution.

---

## Config Resolution

`anam start` looks for `anamdb.toml` and parses configuration blocks:

```toml
[server]
bind = "0.0.0.0:8080"
log_level = "info"

[engine]
provenance_mode = "polynomial"
gpu = false
anomaly_threshold = 0.5
```

If the file does not exist, it falls back to defaults and alerts you to run `anam init` first.

---

## Example

```bash
$ anam start
🚀 Starting AnamDB server on 0.0.0.0:8080...
```
