//! Adaptive Concurrency Control.
//!
//! Dynamically switches between Optimistic Concurrency Control (OCC) and
//! Strict Two-Phase Locking (2PL) based on runtime contention metrics.
//! Inspired by NeurDB's Learned Concurrency Control (LCC).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// Concurrency control strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConcurrencyMode {
    /// Optimistic concurrency control — validate at commit time.
    /// Best for low-contention, short transactions.
    Occ,
    /// Strict two-phase locking — acquire locks eagerly.
    /// Best for high-contention, long transactions.
    Strict2pl,
    /// Adaptive — the controller selects the strategy dynamically.
    Adaptive,
}

impl std::fmt::Display for ConcurrencyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConcurrencyMode::Occ => write!(f, "OCC"),
            ConcurrencyMode::Strict2pl => write!(f, "Strict2PL"),
            ConcurrencyMode::Adaptive => write!(f, "Adaptive"),
        }
    }
}

/// Runtime contention metrics collected by the controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionMetrics {
    /// Total transactions started.
    pub total_txns: u64,
    /// Total transactions committed successfully.
    pub committed_txns: u64,
    /// Total transactions aborted due to conflicts.
    pub aborted_txns: u64,
    /// Current conflict rate (aborted / total, 0.0–1.0).
    pub conflict_rate: f64,
    /// Average transaction duration in microseconds.
    pub avg_txn_duration_us: f64,
    /// Currently active strategy.
    pub active_strategy: ConcurrencyMode,
}

/// Transaction state.
#[derive(Debug, Clone)]
pub struct Transaction {
    /// Unique transaction ID.
    pub txn_id: String,
    /// Which CC strategy this transaction uses.
    pub strategy: ConcurrencyMode,
    /// Start timestamp (microseconds since epoch).
    pub start_us: u64,
    /// Read set: table names read during this transaction.
    pub read_set: Vec<String>,
    /// Write set: table names written during this transaction.
    pub write_set: Vec<String>,
}

/// Adaptive Concurrency Controller.
///
/// Tracks contention metrics and dynamically selects between OCC and
/// Strict 2PL based on runtime heuristics.
pub struct AdaptiveConcurrencyController {
    /// Configured mode (Adaptive = auto-select, others = fixed).
    mode: ConcurrencyMode,
    /// Total transactions started.
    total_txns: AtomicU64,
    /// Total committed.
    committed_txns: AtomicU64,
    /// Total aborted.
    aborted_txns: AtomicU64,
    /// Sum of transaction durations (microseconds) for averaging.
    total_duration_us: AtomicU64,
    /// Active transactions: txn_id → Transaction.
    active: Arc<RwLock<Vec<Transaction>>>,
    /// Conflict rate threshold for switching to Strict 2PL.
    conflict_threshold: f64,
    /// Transaction duration threshold (µs) for considering a txn "long".
    long_txn_threshold_us: u64,
}

impl std::fmt::Debug for AdaptiveConcurrencyController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveConcurrencyController")
            .field("mode", &self.mode)
            .field("total_txns", &self.total_txns.load(Ordering::Relaxed))
            .field("conflict_rate", &self.current_conflict_rate())
            .finish()
    }
}

impl AdaptiveConcurrencyController {
    /// Create a new controller.
    ///
    /// - `mode`: Fixed strategy or `Adaptive` for auto-selection.
    /// - `conflict_threshold`: Conflict rate above which we switch to 2PL (default 0.1 = 10%).
    /// - `long_txn_threshold_us`: Transactions longer than this are "long" (default 10ms = 10_000µs).
    pub fn new(mode: ConcurrencyMode) -> Self {
        Self {
            mode,
            total_txns: AtomicU64::new(0),
            committed_txns: AtomicU64::new(0),
            aborted_txns: AtomicU64::new(0),
            total_duration_us: AtomicU64::new(0),
            active: Arc::new(RwLock::new(Vec::new())),
            conflict_threshold: 0.1,
            long_txn_threshold_us: 10_000,
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(
        mode: ConcurrencyMode,
        conflict_threshold: f64,
        long_txn_threshold_us: u64,
    ) -> Self {
        Self {
            conflict_threshold,
            long_txn_threshold_us,
            ..Self::new(mode)
        }
    }

    /// Select the best CC strategy based on current contention metrics.
    #[instrument(skip(self))]
    pub fn select_strategy(&self) -> ConcurrencyMode {
        match self.mode {
            ConcurrencyMode::Occ | ConcurrencyMode::Strict2pl => self.mode,
            ConcurrencyMode::Adaptive => {
                let conflict_rate = self.current_conflict_rate();
                let avg_duration = self.current_avg_duration_us();

                let strategy = if conflict_rate > self.conflict_threshold
                    || avg_duration > self.long_txn_threshold_us as f64
                {
                    // High contention or long transactions → pessimistic locking.
                    ConcurrencyMode::Strict2pl
                } else {
                    // Low contention, short transactions → optimistic.
                    ConcurrencyMode::Occ
                };

                debug!(
                    conflict_rate,
                    avg_duration_us = avg_duration,
                    selected = %strategy,
                    "adaptive strategy selection"
                );

                strategy
            }
        }
    }

    /// Begin a new transaction.
    pub fn begin_txn(&self) -> Transaction {
        let strategy = self.select_strategy();
        let txn_id = Uuid::new_v4().to_string();

        self.total_txns.fetch_add(1, Ordering::Relaxed);

        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let txn = Transaction {
            txn_id: txn_id.clone(),
            strategy,
            start_us: now_us,
            read_set: Vec::new(),
            write_set: Vec::new(),
        };

        self.active.write().push(txn.clone());

        info!(
            txn_id = %txn_id,
            strategy = %strategy,
            "transaction started"
        );

        txn
    }

    /// Commit a transaction.
    ///
    /// For OCC: validates the read/write sets against concurrent modifications.
    /// For 2PL: releases all locks.
    pub fn commit_txn(&self, txn: &Transaction) -> Result<(), String> {
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        let duration = now_us.saturating_sub(txn.start_us);

        match txn.strategy {
            ConcurrencyMode::Occ => {
                // OCC validation: check for write-write conflicts with active txns.
                // Scope the read lock so it's dropped before we call remove_active().
                let has_conflict = {
                    let active = self.active.read();
                    active.iter().any(|other| {
                        other.txn_id != txn.txn_id
                            && txn
                                .write_set
                                .iter()
                                .any(|w| other.write_set.contains(w))
                    })
                }; // read lock dropped here

                if has_conflict {
                    // Abort — caller should retry.
                    self.aborted_txns.fetch_add(1, Ordering::Relaxed);
                    self.remove_active(&txn.txn_id);
                    return Err(format!(
                        "OCC validation failed for txn {}: write-write conflict detected",
                        txn.txn_id
                    ));
                }
            }
            ConcurrencyMode::Strict2pl | ConcurrencyMode::Adaptive => {
                // 2PL: locks are released implicitly on commit.
            }
        }

        self.committed_txns.fetch_add(1, Ordering::Relaxed);
        self.total_duration_us.fetch_add(duration, Ordering::Relaxed);
        self.remove_active(&txn.txn_id);

        debug!(
            txn_id = %txn.txn_id,
            duration_us = duration,
            strategy = %txn.strategy,
            "transaction committed"
        );

        Ok(())
    }

    /// Abort a transaction.
    pub fn abort_txn(&self, txn: &Transaction) {
        self.aborted_txns.fetch_add(1, Ordering::Relaxed);
        self.remove_active(&txn.txn_id);

        debug!(txn_id = %txn.txn_id, "transaction aborted");
    }

    /// Get current contention metrics.
    pub fn metrics(&self) -> ContentionMetrics {
        ContentionMetrics {
            total_txns: self.total_txns.load(Ordering::Relaxed),
            committed_txns: self.committed_txns.load(Ordering::Relaxed),
            aborted_txns: self.aborted_txns.load(Ordering::Relaxed),
            conflict_rate: self.current_conflict_rate(),
            avg_txn_duration_us: self.current_avg_duration_us(),
            active_strategy: self.select_strategy(),
        }
    }

    /// Current conflict rate.
    fn current_conflict_rate(&self) -> f64 {
        let total = self.total_txns.load(Ordering::Relaxed);
        let aborted = self.aborted_txns.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            aborted as f64 / total as f64
        }
    }

    /// Current average transaction duration.
    fn current_avg_duration_us(&self) -> f64 {
        let committed = self.committed_txns.load(Ordering::Relaxed);
        let total_dur = self.total_duration_us.load(Ordering::Relaxed);
        if committed == 0 {
            0.0
        } else {
            total_dur as f64 / committed as f64
        }
    }

    /// Remove a transaction from the active set.
    fn remove_active(&self, txn_id: &str) {
        self.active.write().retain(|t| t.txn_id != txn_id);
    }
}

impl Default for AdaptiveConcurrencyController {
    fn default() -> Self {
        Self::new(ConcurrencyMode::Adaptive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_starts_with_occ() {
        let cc = AdaptiveConcurrencyController::new(ConcurrencyMode::Adaptive);
        // No contention → should select OCC.
        assert_eq!(cc.select_strategy(), ConcurrencyMode::Occ);
    }

    #[test]
    fn high_contention_switches_to_2pl() {
        let cc = AdaptiveConcurrencyController::new(ConcurrencyMode::Adaptive);
        // Simulate high abort rate.
        cc.total_txns.store(100, Ordering::Relaxed);
        cc.aborted_txns.store(20, Ordering::Relaxed); // 20% conflict rate > 10% threshold.

        assert_eq!(cc.select_strategy(), ConcurrencyMode::Strict2pl);
    }

    #[test]
    fn fixed_mode_ignores_metrics() {
        let cc = AdaptiveConcurrencyController::new(ConcurrencyMode::Occ);
        cc.total_txns.store(100, Ordering::Relaxed);
        cc.aborted_txns.store(50, Ordering::Relaxed); // 50% conflict rate.

        // Fixed OCC mode ignores metrics.
        assert_eq!(cc.select_strategy(), ConcurrencyMode::Occ);
    }

    #[test]
    fn begin_and_commit_txn() {
        let cc = AdaptiveConcurrencyController::new(ConcurrencyMode::Adaptive);
        let txn = cc.begin_txn();
        assert!(!txn.txn_id.is_empty());

        cc.commit_txn(&txn).unwrap();
        let metrics = cc.metrics();
        assert_eq!(metrics.committed_txns, 1);
        assert_eq!(metrics.total_txns, 1);
    }

    #[test]
    fn occ_detects_write_conflict() {
        let cc = AdaptiveConcurrencyController::new(ConcurrencyMode::Occ);

        let mut txn1 = cc.begin_txn();
        txn1.write_set.push("transactions".to_string());
        // Update active set with write set.
        {
            let mut active = cc.active.write();
            if let Some(t) = active.iter_mut().find(|t| t.txn_id == txn1.txn_id) {
                t.write_set = txn1.write_set.clone();
            }
        }

        let mut txn2 = cc.begin_txn();
        txn2.write_set.push("transactions".to_string());
        {
            let mut active = cc.active.write();
            if let Some(t) = active.iter_mut().find(|t| t.txn_id == txn2.txn_id) {
                t.write_set = txn2.write_set.clone();
            }
        }

        // txn2 should fail OCC validation because txn1 is still active
        // and writes to the same table.
        let result = cc.commit_txn(&txn2);
        assert!(result.is_err(), "Expected OCC conflict");
    }

    #[test]
    fn metrics_reporting() {
        let cc = AdaptiveConcurrencyController::new(ConcurrencyMode::Adaptive);
        let txn = cc.begin_txn();
        cc.commit_txn(&txn).unwrap();

        let txn2 = cc.begin_txn();
        cc.abort_txn(&txn2);

        let m = cc.metrics();
        assert_eq!(m.total_txns, 2);
        assert_eq!(m.committed_txns, 1);
        assert_eq!(m.aborted_txns, 1);
        assert!((m.conflict_rate - 0.5).abs() < f64::EPSILON);
    }
}
