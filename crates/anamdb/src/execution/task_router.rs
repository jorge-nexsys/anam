//! Multi-Agent Task Router.
//!
//! Routes FAO operator invocations across a cluster of agent nodes. Edge nodes
//! handle lightweight perception tasks (OCR, image classification) while core
//! nodes handle heavy symbolic joins and reasoning.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, instrument};

use crate::core::error::{AnamError, Result};

/// The role of a node in the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeRole {
    /// High-memory, high-compute core node — handles symbolic joins.
    Core,
    /// Lightweight edge node — handles perception (NPU/GPU inference).
    Edge,
    /// Hybrid node — can handle both, prefers perception tasks.
    Hybrid,
}

impl std::fmt::Display for NodeRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeRole::Core => write!(f, "Core"),
            NodeRole::Edge => write!(f, "Edge"),
            NodeRole::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// An agent node in the distributed cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentNode {
    /// Unique node identifier.
    pub node_id: String,
    /// Node role.
    pub role: NodeRole,
    /// Available memory in MB.
    pub memory_mb: u64,
    /// Number of hardware accelerators (GPUs/NPUs).
    pub accelerators: u32,
    /// Current load (0.0 = idle, 1.0 = fully saturated).
    pub load: f64,
    /// Network latency to this node from the coordinator (ms).
    pub latency_ms: f64,
}

/// A routing decision for a FAO operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaoRoute {
    /// The FAO function ID being routed.
    pub function_id: String,
    /// The target node.
    pub target_node: String,
    /// Reason for this routing decision.
    pub reason: String,
    /// Estimated network overhead in ms.
    pub network_overhead_ms: f64,
}

/// The category of a FAO operation (for routing decisions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaoCategory {
    /// Neural perception: image/audio/video classification, OCR, etc.
    Perception,
    /// Symbolic reasoning: Datalog joins, rule evaluation, etc.
    SymbolicJoin,
    /// Mixed: requires both neural inference and symbolic reasoning.
    Mixed,
}

impl std::fmt::Display for FaoCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FaoCategory::Perception => write!(f, "Perception"),
            FaoCategory::SymbolicJoin => write!(f, "SymbolicJoin"),
            FaoCategory::Mixed => write!(f, "Mixed"),
        }
    }
}

/// The Task Router — routes FAO invocations across the cluster.
#[derive(Debug)]
pub struct TaskRouter {
    /// Known agent nodes.
    nodes: HashMap<String, AgentNode>,
}

impl TaskRouter {
    /// Create a new router.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Register a node in the cluster.
    pub fn register_node(&mut self, node: AgentNode) {
        info!(
            node_id = %node.node_id,
            role = %node.role,
            memory_mb = node.memory_mb,
            accelerators = node.accelerators,
            "registered agent node"
        );
        self.nodes.insert(node.node_id.clone(), node);
    }

    /// Remove a node from the cluster.
    pub fn deregister_node(&mut self, node_id: &str) {
        self.nodes.remove(node_id);
    }

    /// Get the number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Route a FAO operation to the best available node.
    #[instrument(skip(self))]
    pub fn route(
        &self,
        function_id: &str,
        category: FaoCategory,
        min_memory_mb: u64,
    ) -> Result<FaoRoute> {
        if self.nodes.is_empty() {
            return Err(AnamError::Logic(
                "no agent nodes registered in the cluster".into(),
            ));
        }

        let (target, reason) = match category {
            FaoCategory::Perception => {
                // Prefer edge/hybrid nodes with accelerators and low load.
                let best = self
                    .nodes
                    .values()
                    .filter(|n| n.role != NodeRole::Core && n.accelerators > 0)
                    .min_by(|a, b| {
                        a.load
                            .partial_cmp(&b.load)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                if let Some(node) = best {
                    (
                        node.clone(),
                        format!(
                            "Edge perception: {} has {} accelerators, load={:.0}%",
                            node.node_id,
                            node.accelerators,
                            node.load * 100.0
                        ),
                    )
                } else {
                    // Fall back to any node with accelerators.
                    let fallback = self
                        .nodes
                        .values()
                        .filter(|n| n.accelerators > 0)
                        .min_by(|a, b| {
                            a.load
                                .partial_cmp(&b.load)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .ok_or_else(|| {
                            AnamError::Logic(
                                "no nodes with accelerators available for perception".into(),
                            )
                        })?;
                    (
                        fallback.clone(),
                        format!("Fallback to {} (core with accelerators)", fallback.node_id),
                    )
                }
            }
            FaoCategory::SymbolicJoin => {
                // Prefer core nodes with high memory.
                let best = self
                    .nodes
                    .values()
                    .filter(|n| n.role != NodeRole::Edge && n.memory_mb >= min_memory_mb)
                    .min_by(|a, b| {
                        a.load
                            .partial_cmp(&b.load)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .ok_or_else(|| {
                        AnamError::Logic(format!(
                            "no core nodes with >= {min_memory_mb}MB available for symbolic join"
                        ))
                    })?;
                (
                    best.clone(),
                    format!(
                        "Core symbolic join: {} has {}MB, load={:.0}%",
                        best.node_id,
                        best.memory_mb,
                        best.load * 100.0
                    ),
                )
            }
            FaoCategory::Mixed => {
                // Prefer hybrid nodes, then core with accelerators.
                let best = self
                    .nodes
                    .values()
                    .filter(|n| {
                        n.role == NodeRole::Hybrid
                            || (n.role == NodeRole::Core && n.accelerators > 0)
                    })
                    .filter(|n| n.memory_mb >= min_memory_mb)
                    .min_by(|a, b| {
                        a.load
                            .partial_cmp(&b.load)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .ok_or_else(|| {
                        AnamError::Logic("no hybrid/core nodes available for mixed workload".into())
                    })?;
                (
                    best.clone(),
                    format!(
                        "Mixed workload: {} ({}, {}MB, {} accelerators)",
                        best.node_id, best.role, best.memory_mb, best.accelerators
                    ),
                )
            }
        };

        let route = FaoRoute {
            function_id: function_id.to_string(),
            target_node: target.node_id.clone(),
            reason,
            network_overhead_ms: target.latency_ms,
        };

        debug!(
            function = function_id,
            target = %route.target_node,
            overhead_ms = route.network_overhead_ms,
            "routed FAO"
        );

        Ok(route)
    }

    /// Get a formatted summary of the cluster.
    pub fn summary(&self) -> String {
        let mut lines = vec![format!(
            "═══ Agent Cluster ({} nodes) ═══",
            self.nodes.len()
        )];
        for node in self.nodes.values() {
            lines.push(format!(
                "  [{}] {} — {}MB, {} accel, {:.0}% load, {:.1}ms latency",
                node.role,
                node.node_id,
                node.memory_mb,
                node.accelerators,
                node.load * 100.0,
                node.latency_ms
            ));
        }
        lines.join("\n")
    }
}

impl Default for TaskRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cluster() -> TaskRouter {
        let mut router = TaskRouter::new();
        router.register_node(AgentNode {
            node_id: "core-0".into(),
            role: NodeRole::Core,
            memory_mb: 65536,
            accelerators: 0,
            load: 0.2,
            latency_ms: 1.0,
        });
        router.register_node(AgentNode {
            node_id: "edge-0".into(),
            role: NodeRole::Edge,
            memory_mb: 4096,
            accelerators: 2,
            load: 0.1,
            latency_ms: 5.0,
        });
        router.register_node(AgentNode {
            node_id: "hybrid-0".into(),
            role: NodeRole::Hybrid,
            memory_mb: 32768,
            accelerators: 4,
            load: 0.3,
            latency_ms: 2.0,
        });
        router
    }

    #[test]
    fn route_perception_to_edge() {
        let router = test_cluster();
        let route = router
            .route("ocr_model", FaoCategory::Perception, 0)
            .unwrap();
        assert_eq!(route.target_node, "edge-0");
    }

    #[test]
    fn route_symbolic_to_core() {
        let router = test_cluster();
        let route = router
            .route("datalog_join", FaoCategory::SymbolicJoin, 8192)
            .unwrap();
        assert_eq!(route.target_node, "core-0");
    }

    #[test]
    fn route_mixed_to_hybrid() {
        let router = test_cluster();
        let route = router.route("nlp_classify", FaoCategory::Mixed, 0).unwrap();
        assert_eq!(route.target_node, "hybrid-0");
    }
}
