# CLI Command: status

The `status` subcommand checks the health of a running AnamDB server and fetches its schema catalog properties.

## Usage

```bash
anam status [OPTIONS]
```

---

## Options

* **`--addr <IP:PORT>`**: Server address to query status from (defaults to `127.0.0.1:8080`).

---

## Example

```bash
$ anam status
Connecting to AnamDB at 127.0.0.1:8080...
✓ AnamDB is healthy
  Version:  1.0.0
  Tables:   4
  Models:   2
  Rules:    12
```
