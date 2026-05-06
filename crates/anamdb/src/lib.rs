//! # AnamDB
//!
//! **The AI-Native, Differentiable Logic Kernel for Autonomous Agents.**
//!
//! AnamDB is a neurosymbolic database engine that natively integrates probabilistic
//! neural inference with deterministic symbolic reasoning. Models are first-class
//! citizens, logic is verifiable, and every query returns a provenance-backed
//! reasoning trace.
//!
//! ## Modules
//!
//! - [`core`] — Session API, Arrow schemas, and semiring provenance.
//! - [`model`] — AI-Tables, Function-as-Operator (FAO) registry, and inference adapters.
//! - [`logic`] — Differentiable Datalog engine (Scallop) and NL-to-Logic compiler.
//! - [`execution`] — Extended DataFusion operators, Pareto optimizer, and heterogeneous dispatch.
//! - [`storage`] — Lance-backed table provider with snapshot versioning.
//! - [`hitl`] — Human-in-the-Loop semantic monitoring and interactive triage.

#![deny(clippy::all)]
#![warn(missing_docs)]

pub mod core;
pub mod execution;
pub mod hitl;
pub mod logic;
pub mod model;
pub mod storage;

// Re-export the primary public API surface.
pub use crate::core::session::Session;
pub use crate::core::error::{AnamError, Result};
