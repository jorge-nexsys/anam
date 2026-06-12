//! Datalog logic engine, NL-to-Datalog compiler, schema validation,
//! canonical normalization, and Hamming-distance-1 repair.

pub mod canonical;
pub mod datalog_checker;
pub mod datalog_repair;
pub mod engine;
pub mod nl_compiler;
