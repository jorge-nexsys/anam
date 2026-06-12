//! AI-Tables Community Hub — package manager for FAO models and Datalog packs.
//!
//! Provides a centralized registry where developers can publish, discover,
//! and install pre-trained FAO (Function-as-Operator) model packages and
//! Datalog logic constraint packs.
//!
//! ## Design
//! - **Local index**: a JSON manifest stored in `~/.anam/hub/index.json`
//! - **Remote index**: fetched from a registry URL (defaults to the official hub)
//! - **Pack format**: a `.anampack` file (ZIP) containing:
//!     - `manifest.json` — metadata (name, version, description, author)
//!     - `models/` — ONNX model files
//!     - `rules/` — Datalog rule files (`.dl`)
//!     - `README.md` — usage instructions
//!
//! ## CLI Usage
//! ```bash
//! anam hub search fraud
//! anam hub install anamdb/financial-compliance-pack@1.0.0
//! anam hub publish ./my-pack/
//! anam hub list
//! ```
//!
//! ## SQL Usage
//! ```sql
//! SELECT hub_install('anamdb/financial-compliance-pack') AS result;
//! SELECT hub_search('fraud') AS packs;
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::core::error::{AnamError, Result};

// ── Pack Manifest ─────────────────────────────────────────────────────

/// Semantic version string (major.minor.patch).
pub type Version = String;

/// Metadata for a published AI-Tables pack.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PackManifest {
    /// Unique package name (e.g., "anamdb/financial-compliance").
    pub name: String,
    /// Semantic version.
    pub version: Version,
    /// Short human-readable description.
    pub description: String,
    /// Author / organization.
    pub author: String,
    /// License identifier (e.g., "Apache-2.0", "MIT").
    pub license: String,
    /// Tags for discovery (e.g., ["fraud", "finance", "classification"]).
    pub tags: Vec<String>,
    /// ONNX model files included in this pack.
    pub models: Vec<PackModel>,
    /// Datalog rule files included in this pack.
    pub rules: Vec<PackRule>,
    /// Compatible AnamDB version range.
    pub anamdb_version: String,
    /// SHA-256 checksum of the pack file.
    pub checksum: Option<String>,
}

/// A model entry within a pack manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PackModel {
    /// Model name (becomes the SQL function name on install).
    pub name: String,
    /// Relative path to ONNX file within the pack.
    pub path: String,
    /// Number of input features.
    pub num_features: usize,
    /// Average inference latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Benchmark accuracy.
    pub accuracy: f64,
}

/// A Datalog rule file within a pack manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PackRule {
    /// Rule set name.
    pub name: String,
    /// Relative path to `.dl` file within the pack.
    pub path: String,
    /// Short description of what the rules enforce.
    pub description: String,
}

// ── Hub Index ─────────────────────────────────────────────────────────

/// The local hub index — a map of `name@version` → manifest.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HubIndex {
    /// Installed packs, keyed by `name@version`.
    pub installed: HashMap<String, PackManifest>,
    /// Available remote packs (fetched from registry URL).
    pub available: HashMap<String, PackManifest>,
    /// Registry URL.
    pub registry_url: String,
}

impl HubIndex {
    /// Create a new empty index with the given registry URL.
    pub fn new(registry_url: &str) -> Self {
        Self {
            registry_url: registry_url.to_string(),
            ..Default::default()
        }
    }

    fn pack_key(name: &str, version: &str) -> String {
        format!("{name}@{version}")
    }
}

// ── Hub Client ────────────────────────────────────────────────────────

/// The AI-Tables Hub client — manages discovery, installation, and publishing.
pub struct HubClient {
    /// Path to the local hub directory (default: `~/.anam/hub/`).
    pub hub_dir: PathBuf,
    /// Loaded index.
    pub index: HubIndex,
    /// Registry base URL.
    pub registry_url: String,
}

impl HubClient {
    /// The default official registry URL.
    pub const DEFAULT_REGISTRY: &'static str = "https://anamdb.github.io/anam-db/registry";

    /// Create a new hub client rooted at `hub_dir`.
    pub fn new(hub_dir: impl AsRef<Path>) -> Result<Self> {
        let hub_dir = hub_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&hub_dir).map_err(AnamError::Io)?;

        let index_path = hub_dir.join("index.json");
        let index = if index_path.exists() {
            let raw = std::fs::read_to_string(&index_path).map_err(AnamError::Io)?;
            serde_json::from_str(&raw).map_err(|e| AnamError::Serde(e.to_string()))?
        } else {
            HubIndex::new(Self::DEFAULT_REGISTRY)
        };

        Ok(Self {
            registry_url: index.registry_url.clone(),
            hub_dir,
            index,
        })
    }

    /// Persist the current index to disk.
    pub fn save_index(&self) -> Result<()> {
        let index_path = self.hub_dir.join("index.json");
        let raw = serde_json::to_string_pretty(&self.index)
            .map_err(|e| AnamError::Serde(e.to_string()))?;
        std::fs::write(&index_path, raw).map_err(AnamError::Io)?;
        Ok(())
    }

    // ── Search ────────────────────────────────────────────────────────

    /// Search available packs by keyword (matches name, description, or tags).
    pub fn search(&self, keyword: &str) -> Vec<&PackManifest> {
        let kw = keyword.to_lowercase();
        let mut results: Vec<&PackManifest> = self
            .index
            .available
            .values()
            .filter(|m| {
                m.name.to_lowercase().contains(&kw)
                    || m.description.to_lowercase().contains(&kw)
                    || m.tags.iter().any(|t| t.to_lowercase().contains(&kw))
            })
            .collect();
        results.sort_by(|a, b| a.name.cmp(&b.name));
        results
    }

    // ── Install ───────────────────────────────────────────────────────

    /// Install a pack by `name@version` string.
    ///
    /// In production this fetches the `.anampack` from the registry,
    /// verifies the checksum, extracts models/rules, and registers them.
    /// Here we record the installation in the local index.
    pub fn install(&mut self, pack_ref: &str) -> Result<InstallResult> {
        info!(pack = %pack_ref, "installing pack from hub");

        // Parse "name@version" or just "name" (resolves to latest).
        let (name, version) = parse_pack_ref(pack_ref)?;

        // Check if already installed.
        let key = HubIndex::pack_key(&name, &version);
        if self.index.installed.contains_key(&key) {
            return Ok(InstallResult {
                pack_ref: pack_ref.to_string(),
                installed: false,
                message: format!("'{pack_ref}' is already installed"),
            });
        }

        // Find in available index (or create a placeholder for demo).
        let manifest = self
            .index
            .available
            .get(&key)
            .cloned()
            .unwrap_or_else(|| PackManifest {
                name: name.clone(),
                version: version.clone(),
                description: format!("Community pack: {name}"),
                author: "community".into(),
                license: "Apache-2.0".into(),
                tags: vec![name.split('/').next_back().unwrap_or("").to_string()],
                models: vec![],
                rules: vec![],
                anamdb_version: ">=0.1.0".into(),
                checksum: None,
            });

        // Record as installed.
        self.index.installed.insert(key.clone(), manifest);
        self.save_index()?;

        info!(pack = %pack_ref, "pack installed successfully");

        Ok(InstallResult {
            pack_ref: pack_ref.to_string(),
            installed: true,
            message: format!("'{pack_ref}' installed successfully"),
        })
    }

    // ── Publish ───────────────────────────────────────────────────────

    /// Publish a local pack to the hub.
    ///
    /// Reads the manifest from `pack_dir/manifest.json`, validates it,
    /// and adds it to the local available index (simulating a registry push).
    pub fn publish(&mut self, pack_dir: impl AsRef<Path>) -> Result<PublishResult> {
        let pack_dir = pack_dir.as_ref();
        let manifest_path = pack_dir.join("manifest.json");

        if !manifest_path.exists() {
            return Err(AnamError::Logic(format!(
                "manifest.json not found in {}",
                pack_dir.display()
            )));
        }

        let raw = std::fs::read_to_string(&manifest_path).map_err(AnamError::Io)?;
        let manifest: PackManifest =
            serde_json::from_str(&raw).map_err(|e| AnamError::Serde(e.to_string()))?;

        let key = HubIndex::pack_key(&manifest.name, &manifest.version);
        info!(pack = %key, "publishing pack to hub");

        self.index.available.insert(key.clone(), manifest.clone());
        self.save_index()?;

        Ok(PublishResult {
            name: manifest.name,
            version: manifest.version,
            key,
            message: "Pack published successfully".into(),
        })
    }

    // ── List ──────────────────────────────────────────────────────────

    /// List all installed packs.
    pub fn list_installed(&self) -> Vec<&PackManifest> {
        let mut packs: Vec<&PackManifest> = self.index.installed.values().collect();
        packs.sort_by(|a, b| a.name.cmp(&b.name));
        packs
    }

    /// List all available packs.
    pub fn list_available(&self) -> Vec<&PackManifest> {
        let mut packs: Vec<&PackManifest> = self.index.available.values().collect();
        packs.sort_by(|a, b| a.name.cmp(&b.name));
        packs
    }

    /// Seed the available index with well-known community packs.
    pub fn seed_community_packs(&mut self) -> Result<()> {
        let packs = vec![
            PackManifest {
                name: "anamdb/financial-compliance".into(),
                version: "1.0.0".into(),
                description: "AML/KYC compliance rules + fraud detection models for financial data".into(),
                author: "AnamDB Core Team".into(),
                license: "Apache-2.0".into(),
                tags: vec!["fraud".into(), "finance".into(), "aml".into(), "kyc".into()],
                models: vec![PackModel {
                    name: "fraud_detector".into(),
                    path: "models/fraud_detector.onnx".into(),
                    num_features: 4,
                    avg_latency_ms: 2.1,
                    accuracy: 0.95,
                }],
                rules: vec![PackRule {
                    name: "high_risk_transactions".into(),
                    path: "rules/high_risk.dl".into(),
                    description: "Flags transactions > $10k with high fraud scores".into(),
                }],
                anamdb_version: ">=0.1.0".into(),
                checksum: None,
            },
            PackManifest {
                name: "anamdb/medical-imaging".into(),
                version: "0.9.0".into(),
                description: "DICOM segmentation models + clinical Datalog constraints".into(),
                author: "community".into(),
                license: "Apache-2.0".into(),
                tags: vec!["medical".into(), "imaging".into(), "dicom".into(), "segmentation".into()],
                models: vec![PackModel {
                    name: "lesion_detector".into(),
                    path: "models/lesion_detector.onnx".into(),
                    num_features: 64,
                    avg_latency_ms: 28.5,
                    accuracy: 0.91,
                }],
                rules: vec![PackRule {
                    name: "critical_findings".into(),
                    path: "rules/critical.dl".into(),
                    description: "Escalates high-confidence lesion detections for radiologist review".into(),
                }],
                anamdb_version: ">=0.1.0".into(),
                checksum: None,
            },
            PackManifest {
                name: "anamdb/autonomous-driving".into(),
                version: "0.2.0".into(),
                description: "3D object detection + spatial safety constraint pack for AV pipelines".into(),
                author: "community".into(),
                license: "MIT".into(),
                tags: vec!["autonomous".into(), "lidar".into(), "3d".into(), "spatial".into()],
                models: vec![PackModel {
                    name: "bbox3d_detector".into(),
                    path: "models/bbox3d.onnx".into(),
                    num_features: 128,
                    avg_latency_ms: 12.0,
                    accuracy: 0.88,
                }],
                rules: vec![PackRule {
                    name: "collision_avoidance".into(),
                    path: "rules/collision.dl".into(),
                    description: "Raises alert when predicted trajectory intersects another agent's bounding box".into(),
                }],
                anamdb_version: ">=0.1.0".into(),
                checksum: None,
            },
        ];

        for pack in packs {
            let key = HubIndex::pack_key(&pack.name, &pack.version);
            self.index.available.insert(key, pack);
        }
        self.save_index()?;
        Ok(())
    }
}

// ── Result Types ──────────────────────────────────────────────────────

/// Result of a `hub install` operation.
#[derive(Debug, Clone)]
pub struct InstallResult {
    /// The pack reference string that was installed.
    pub pack_ref: String,
    /// Whether the pack was newly installed (`false` if already present).
    pub installed: bool,
    /// Human-readable status message.
    pub message: String,
}

/// Result of a `hub publish` operation.
#[derive(Debug, Clone)]
pub struct PublishResult {
    /// Package name.
    pub name: String,
    /// Package version.
    pub version: String,
    /// Registry key (`name@version`).
    pub key: String,
    /// Human-readable status message.
    pub message: String,
}

// ── Helpers ───────────────────────────────────────────────────────────

/// Parse a pack reference string into (name, version).
/// Accepts "name@version" or just "name" (defaults to "latest").
fn parse_pack_ref(pack_ref: &str) -> Result<(String, String)> {
    if let Some((name, version)) = pack_ref.split_once('@') {
        Ok((name.to_string(), version.to_string()))
    } else {
        Ok((pack_ref.to_string(), "latest".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_hub() -> (HubClient, TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let mut hub = HubClient::new(dir.path()).unwrap();
        hub.seed_community_packs().unwrap();
        (hub, dir) // keep TempDir alive so the directory isn't deleted
    }

    #[test]
    fn hub_search_finds_packs() {
        let (hub, _dir) = temp_hub();

        let results = hub.search("fraud");
        assert!(!results.is_empty(), "should find fraud pack");
        assert!(results.iter().any(|m| m.name.contains("financial")));

        let results3d = hub.search("3d");
        assert!(results3d.iter().any(|m| m.name.contains("autonomous")));

        println!("\n═══ Hub Search Test ═══");
        println!("  'fraud' → {} result(s)", hub.search("fraud").len());
        println!("  '3d'    → {} result(s)", hub.search("3d").len());
        println!("  'dicom' → {} result(s)", hub.search("dicom").len());
        println!("  ✓ Community hub search works");
    }

    #[test]
    fn hub_install_and_list() {
        let (mut hub, _dir) = temp_hub();

        assert_eq!(hub.list_installed().len(), 0, "nothing installed yet");

        let result = hub.install("anamdb/financial-compliance@1.0.0").unwrap();
        assert!(result.installed);

        assert_eq!(hub.list_installed().len(), 1);
        assert_eq!(hub.list_installed()[0].name, "anamdb/financial-compliance");

        // Installing again should be a no-op.
        let r2 = hub.install("anamdb/financial-compliance@1.0.0").unwrap();
        assert!(!r2.installed, "double-install should be a no-op");
        assert_eq!(
            hub.list_installed().len(),
            1,
            "still 1 pack after duplicate install"
        );

        println!("\n═══ Hub Install Test ═══");
        println!("  ✓ Install: {}", result.message);
        println!("  ✓ List: {} installed pack(s)", hub.list_installed().len());
        println!("  ✓ Duplicate install no-op: {}", r2.message);
    }

    #[test]
    fn hub_publish_and_search() {
        let (mut hub, _dir) = temp_hub();

        let pack_dir = tempfile::tempdir().unwrap();
        let manifest = PackManifest {
            name: "testauthor/my-custom-pack".into(),
            version: "0.1.0".into(),
            description: "Test pack for custom domain".into(),
            author: "testauthor".into(),
            license: "MIT".into(),
            tags: vec!["custom".into(), "test".into()],
            models: vec![],
            rules: vec![],
            anamdb_version: ">=0.1.0".into(),
            checksum: None,
        };
        let manifest_path = pack_dir.path().join("manifest.json");
        std::fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();

        let result = hub.publish(pack_dir.path()).unwrap();
        assert_eq!(result.name, "testauthor/my-custom-pack");

        let search = hub.search("custom");
        assert!(search.iter().any(|m| m.name == "testauthor/my-custom-pack"));

        println!("\n═══ Hub Publish Test ═══");
        println!("  ✓ Published: {}", result.key);
        println!("  ✓ Searchable after publish: {} result(s)", search.len());
    }

    #[test]
    fn hub_index_persists() {
        let dir = tempfile::tempdir().unwrap();

        {
            let mut hub = HubClient::new(dir.path()).unwrap();
            hub.seed_community_packs().unwrap();
            hub.install("anamdb/medical-imaging@0.9.0").unwrap();
        }

        // Re-open the hub — should load from disk.
        let hub2 = HubClient::new(dir.path()).unwrap();
        assert_eq!(hub2.list_installed().len(), 1);
        assert_eq!(hub2.list_installed()[0].name, "anamdb/medical-imaging");

        println!("\n═══ Hub Persistence Test ═══");
        println!("  ✓ Hub index persisted and reloaded from disk");
        println!("  ✓ Installed packs survive restart");
    }
}
