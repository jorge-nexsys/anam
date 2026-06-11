# Changelog

All notable changes to AnamDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- License changed from Business Source License 1.1 (BSL) to Apache License 2.0.
- Added crates.io publish metadata (repository, homepage, keywords, categories).

### Added
- Python SDK (`sdks/python/`) — async client for the AnamDB wire protocol.
- Docker Compose and GHCR publish workflow.
- CLI `start` and `status` subcommands for streamlined workflows.

## [0.1.0-alpha] - 2026-05-07

### Added
- Core Datalog Engine (Scallop-backed) embedded via `logic_engine`.
- AI-Tables implementation with Function-as-Operator (FAO) registry.
- Zero-copy Lance 2.2 storage integration (`LanceTableManager`).
- Polynomial Semiring provenance tracking framework.
- Multi-objective Pareto optimizer for dynamic heterogeneous execution.
- Hardware dispatcher for multi-device (CPU/GPU/NPU) orchestration.
- Human-in-the-Loop (HITL) interactive reasoning and anomaly triage.
- Interactive Web Playground and CLI tool (`anam`).
- Natural Language to Datalog compilation via LLM `nl_compiler`.
