//! Usage metering for AnamDB.
//!
//! Tracks queries, storage, and GPU usage per tenant. In production the
//! accumulated counters are periodically flushed to a billing backend
//! (e.g. Stripe Metering).

use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::info;

/// In-memory usage counters for a single tenant.
#[derive(Debug, Default)]
pub struct TenantMeter {
    /// Total queries executed.
    pub query_count: AtomicUsize,
    /// Total rows scanned across all queries.
    pub rows_scanned: AtomicUsize,
    /// Cumulative GPU milliseconds consumed.
    pub gpu_ms: AtomicUsize,
}

impl TenantMeter {
    /// Create a zeroed meter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single query execution.
    pub fn record_query(&self, rows: usize, gpu_time_ms: usize) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.rows_scanned.fetch_add(rows, Ordering::Relaxed);
        self.gpu_ms.fetch_add(gpu_time_ms, Ordering::Relaxed);
    }

    /// Atomically read and reset all counters, returning `(queries, rows, gpu_ms)`.
    pub fn flush(&self) -> (usize, usize, usize) {
        let q = self.query_count.swap(0, Ordering::Relaxed);
        let r = self.rows_scanned.swap(0, Ordering::Relaxed);
        let g = self.gpu_ms.swap(0, Ordering::Relaxed);
        (q, r, g)
    }
}

/// Global metering coordinator.
///
/// In a real deployment this would hold a `DashMap<String, TenantMeter>`
/// and run a background flush loop pushing data to Stripe.
pub struct MeteringSystem {
    // Placeholder — expand when billing integration lands.
}

impl MeteringSystem {
    /// Create an idle metering system.
    pub fn new() -> Self {
        Self {}
    }

    /// Start the background flush loop (currently a no-op stub).
    pub async fn start_flush_loop(&self) {
        info!("metering flush loop started (stub — no billing backend configured)");
    }
}

impl Default for MeteringSystem {
    fn default() -> Self {
        Self::new()
    }
}
