//! Logic Pack SDK — modular, domain-specific neurosymbolic rulesets.
//!
//! A Logic Pack bundles Datalog rules, model references, and metadata into
//! a reusable, distributable unit. Packs can be loaded into a [`Session`](crate::core::session::Session)
//! to instantly configure domain-specific reasoning.

pub mod logic_pack;

pub use logic_pack::{LogicPack, LogicPackBuilder};
