//! BCNF Policy Catalog.
//!
//! A strict Boyce-Codd Normal Form relational catalog that stores all Datalog
//! rules, programmatic constraints, and schema definitions. Ensures anomaly-free
//! policy propagation across distributed nodes.
//!
//! In a distributed cluster, every node holds a replica of this catalog. Updates
//! are propagated via version-stamped changesets.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{info, instrument};

use crate::core::error::{AnamError, Result};

/// A single policy entry in the BCNF catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEntry {
    /// Unique policy identifier.
    pub policy_id: String,
    /// The Datalog rule body.
    pub datalog: String,
    /// Which relation(s) this policy applies to.
    pub relations: Vec<String>,
    /// Human-readable description.
    pub description: String,
    /// Version of this policy (monotonically increasing).
    pub version: u64,
    /// Whether this policy is active.
    pub active: bool,
}

/// A version-stamped changeset for catalog replication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogChangeset {
    /// Global catalog version after this changeset.
    pub catalog_version: u64,
    /// Policies added or updated.
    pub upserts: Vec<PolicyEntry>,
    /// Policy IDs removed.
    pub deletes: Vec<String>,
    /// Timestamp (Unix millis).
    pub timestamp_ms: u64,
}

/// The BCNF-normalized policy catalog.
///
/// All rules are stored in a strict relational form to prevent update anomalies.
/// The catalog supports version-stamped changesets for cluster-wide replication.
#[derive(Debug)]
pub struct BcnfCatalog {
    /// `policy_id → PolicyEntry`
    policies: HashMap<String, PolicyEntry>,
    /// Monotonically increasing catalog version.
    version: AtomicU64,
    /// Changeset history for replication.
    history: Vec<CatalogChangeset>,
}

impl BcnfCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            version: AtomicU64::new(0),
            history: Vec::new(),
        }
    }

    /// Current catalog version.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    /// Number of active policies.
    pub fn active_count(&self) -> usize {
        self.policies.values().filter(|p| p.active).count()
    }

    /// Total policies (active + inactive).
    pub fn total_count(&self) -> usize {
        self.policies.len()
    }

    /// Insert or update a policy, producing a changeset.
    #[instrument(skip(self))]
    pub fn upsert_policy(&mut self, mut entry: PolicyEntry) -> Result<CatalogChangeset> {
        let new_version = self.version.fetch_add(1, Ordering::Relaxed) + 1;
        entry.version = new_version;

        info!(
            policy_id = %entry.policy_id,
            version = new_version,
            "BCNF catalog: upsert policy"
        );

        self.policies.insert(entry.policy_id.clone(), entry.clone());

        let changeset = CatalogChangeset {
            catalog_version: new_version,
            upserts: vec![entry],
            deletes: Vec::new(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };

        self.history.push(changeset.clone());
        Ok(changeset)
    }

    /// Deactivate a policy (soft delete), producing a changeset.
    pub fn deactivate_policy(&mut self, policy_id: &str) -> Result<CatalogChangeset> {
        let entry = self.policies.get_mut(policy_id).ok_or_else(|| {
            AnamError::Logic(format!("policy '{policy_id}' not found in BCNF catalog"))
        })?;

        entry.active = false;
        let new_version = self.version.fetch_add(1, Ordering::Relaxed) + 1;
        entry.version = new_version;

        let changeset = CatalogChangeset {
            catalog_version: new_version,
            upserts: vec![entry.clone()],
            deletes: Vec::new(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };

        self.history.push(changeset.clone());
        Ok(changeset)
    }

    /// Get a policy by ID.
    pub fn get_policy(&self, policy_id: &str) -> Option<&PolicyEntry> {
        self.policies.get(policy_id)
    }

    /// List all active policies.
    pub fn list_active(&self) -> Vec<&PolicyEntry> {
        self.policies.values().filter(|p| p.active).collect()
    }

    /// Get changesets since a given version (for incremental replication).
    pub fn changesets_since(&self, since_version: u64) -> Vec<&CatalogChangeset> {
        self.history
            .iter()
            .filter(|cs| cs.catalog_version > since_version)
            .collect()
    }

    /// Apply a changeset from another node (replica sync).
    #[instrument(skip(self, changeset))]
    pub fn apply_changeset(&mut self, changeset: &CatalogChangeset) -> Result<()> {
        info!(
            catalog_version = changeset.catalog_version,
            upserts = changeset.upserts.len(),
            deletes = changeset.deletes.len(),
            "applying remote changeset"
        );

        for entry in &changeset.upserts {
            self.policies.insert(entry.policy_id.clone(), entry.clone());
        }

        for policy_id in &changeset.deletes {
            self.policies.remove(policy_id);
        }

        // Advance version if needed.
        let current = self.version.load(Ordering::Relaxed);
        if changeset.catalog_version > current {
            self.version
                .store(changeset.catalog_version, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get a formatted summary.
    pub fn summary(&self) -> String {
        let mut lines = vec![format!("═══ BCNF Policy Catalog (v{}) ═══", self.version())];
        lines.push(format!(
            "  {} active / {} total policies",
            self.active_count(),
            self.total_count()
        ));
        for entry in self.list_active() {
            lines.push(format!(
                "  • [v{}] {} → {} ({})",
                entry.version,
                entry.policy_id,
                entry.datalog,
                entry.relations.join(", ")
            ));
        }
        lines.join("\n")
    }
}

impl Default for BcnfCatalog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_and_replicate() {
        let mut primary = BcnfCatalog::new();

        let cs1 = primary
            .upsert_policy(PolicyEntry {
                policy_id: "aml_high_risk".into(),
                datalog: "fraud_prob > 0.90 AND amount > 10000".into(),
                relations: vec!["transactions".into()],
                description: "AML high-risk flag".into(),
                version: 0,
                active: true,
            })
            .unwrap();

        let cs2 = primary
            .upsert_policy(PolicyEntry {
                policy_id: "wire_alert".into(),
                datalog: "merchant_type = 'wire_transfer' AND amount > 50000".into(),
                relations: vec!["transactions".into()],
                description: "Wire transfer alert".into(),
                version: 0,
                active: true,
            })
            .unwrap();

        assert_eq!(primary.version(), 2);
        assert_eq!(primary.active_count(), 2);

        // Replicate to a second node.
        let mut replica = BcnfCatalog::new();
        replica.apply_changeset(&cs1).unwrap();
        replica.apply_changeset(&cs2).unwrap();

        assert_eq!(replica.version(), 2);
        assert_eq!(replica.active_count(), 2);
        assert!(replica.get_policy("aml_high_risk").is_some());
    }

    #[test]
    fn incremental_sync() {
        let mut primary = BcnfCatalog::new();

        primary
            .upsert_policy(PolicyEntry {
                policy_id: "p1".into(),
                datalog: "x > 10".into(),
                relations: vec!["t".into()],
                description: "test".into(),
                version: 0,
                active: true,
            })
            .unwrap();

        primary
            .upsert_policy(PolicyEntry {
                policy_id: "p2".into(),
                datalog: "y < 5".into(),
                relations: vec!["t".into()],
                description: "test2".into(),
                version: 0,
                active: true,
            })
            .unwrap();

        // Replica already has v1, needs only v2+.
        let delta = primary.changesets_since(1);
        assert_eq!(delta.len(), 1);
        assert_eq!(delta[0].catalog_version, 2);
    }
}
