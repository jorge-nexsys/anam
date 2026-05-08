//! Storage engine: Lance-backed table provider with snapshot versioning,
//! streaming scans, write path, persistent catalog, and spatial/audio types.

pub mod catalog;
pub mod ingestion;
pub mod lance_provider;
pub mod spatial_audio;
pub mod streaming_provider;
pub mod write_path;
