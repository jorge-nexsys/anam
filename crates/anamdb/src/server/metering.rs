//! Usage metering for AnamDB.
//!
//! Tracks queries, storage, and GPU usage per tenant, flushing
//! to a billing backend (like Stripe Metering) periodically.

use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::info;

/// A simple in-memory meter for a tenant.
#[derive(Debug, Default)]
pub struct TenantMeter {
    pub query_count: AtomicUsize,
    pub rows_scanned: AtomicUsize,
    pub gpu_ms: AtomicUsize,
}

impl TenantMeter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_query(&self, rows: usize, gpu_time_ms: usize) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.rows_scanned.fetch_add(rows, Ordering::Relaxed);
        self.gpu_ms.fetch_add(gpu_time_ms, Ordering::Relaxed);
    }

    /// Flush and reset the metrics.
    pub fn flush(&self) -> (usize, usize, usize) {
        let q = self.query_count.swap(0, Ordering::Relaxed);
        let r = self.rows_scanned.swap(0, Ordering::Relaxed);
        let g = self.gpu_ms.swap(0, Ordering::Relaxed);
        (q, r, g)
    }
}

/// The global metering system.
pub struct MeteringSystem {
    // In a real system, this would be a DashMap<String, TenantMeter>
    // mapping tenant_id to their current usage.
}

impl MeteringSystem {
    pub fn new() -> Self {
        Self {}
    }

    /// Background task to periodically flush metrics to Stripe.
    pub async fn start_flush_loop(&self) {
        info!("Starting metering flush loop (stub)");
        // loop {
        //     tokio::time::sleep(Duration::from_secs(60)).await;
        //     // flush all tenants...
        // }
    }
}
