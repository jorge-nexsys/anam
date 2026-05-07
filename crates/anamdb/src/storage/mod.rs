//! Storage engine: Lance-backed table provider with snapshot versioning,
//! streaming scans, write path, and persistent catalog.

pub mod catalog;
pub mod ingestion;
pub mod lance_provider;
pub mod streaming_provider;
pub mod write_path;
