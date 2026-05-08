//! Logic Pack SDK — modular, domain-specific neurosymbolic rulesets,
//! and the AI-Tables Community Hub package manager.

pub mod hub;
pub mod logic_pack;
pub mod python;

pub use logic_pack::{LogicPack, LogicPackBuilder};
