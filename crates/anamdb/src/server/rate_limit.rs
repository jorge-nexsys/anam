//! Rate limiting for AnamDB server requests.
//!
//! Implements a per-tenant token-bucket algorithm to prevent single
//! tenants from degrading the shared cluster.

use std::collections::HashMap;
use std::time::Instant;
use tokio::sync::Mutex;

use crate::core::error::{AnamError, Result};
use crate::server::auth::SubscriptionTier;

/// Per-tenant token-bucket rate limiter.
pub struct RateLimiter {
    /// Maps `tenant_id` → `(tokens_remaining, last_refill_time)`.
    buckets: Mutex<HashMap<String, (f64, Instant)>>,
}

impl RateLimiter {
    /// Create a new, empty rate limiter.
    pub fn new() -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Check whether a request is allowed for the given tenant and tier.
    ///
    /// Returns `Ok(())` if the request is allowed, or an error if the
    /// tenant has exhausted their token budget.
    pub async fn check_limit(&self, tenant_id: &str, tier: &SubscriptionTier) -> Result<()> {
        let (capacity, refill_rate) = match tier {
            SubscriptionTier::Community => (10.0, 1.0),
            SubscriptionTier::Pro => (100.0, 10.0),
            SubscriptionTier::Team => (1000.0, 100.0),
            SubscriptionTier::Enterprise => (10000.0, 1000.0),
        };

        let mut buckets = self.buckets.lock().await;
        let now = Instant::now();

        let bucket = buckets
            .entry(tenant_id.to_string())
            .or_insert((capacity, now));

        let elapsed = now.duration_since(bucket.1).as_secs_f64();
        bucket.0 = (bucket.0 + elapsed * refill_rate).min(capacity);
        bucket.1 = now;

        if bucket.0 >= 1.0 {
            bucket.0 -= 1.0;
            Ok(())
        } else {
            Err(AnamError::Internal("Rate limit exceeded (HTTP 429)".into()))
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}
