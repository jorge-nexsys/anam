# Getting Started

To get started with AnamDB, you can either install the CLI locally or run the production-ready server.

## Installation

```bash
cargo install --git https://github.com/AnamDB/anam-db anam-cli
```

## Running the Server

AnamDB comes with a highly concurrent, asynchronous gRPC server.

```bash
anam serve --port 8080
```

## Community Hub

AnamDB includes a package manager to share AI models and Datalog logic packs.

```bash
anam hub search fraud
anam hub install anamdb/financial-compliance@1.0.0
```
