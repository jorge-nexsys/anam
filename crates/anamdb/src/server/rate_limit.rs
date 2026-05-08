//! Rate limiting for AnamDB server requests.
//! 
//! Prevents single tenants from degrading the entire cluster.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::{Instant, Duration};

use crate::core::error::{AnamError, Result};
use crate::server::auth::SubscriptionTier;

/// Basic Token Bucket rate limiter per tenant.
pub struct RateLimiter {
    // Maps tenant_id to (tokens_remaining, last_refill_time)
    buckets: Mutex<HashMap<String, (f64, Instant)>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Check if a request is allowed for the given tenant and tier.
    pub async fn check_limit(&self, tenant_id: &str, tier: &SubscriptionTier) -> Result<()> {
        let (capacity, refill_rate) = match tier {
            SubscriptionTier::Community => (10.0, 1.0),   // 10 burst, 1 req/sec
            SubscriptionTier::Pro => (100.0, 10.0),       // 100 burst, 10 req/sec
            SubscriptionTier::Team => (1000.0, 100.0),    // 1000 burst, 100 req/sec
            SubscriptionTier::Enterprise => (10000.0, 1000.0), 
        };

        let mut buckets = self.buckets.lock().await;
        let now = Instant::now();

        let bucket = buckets.entry(tenant_id.to_string()).or_insert((capacity, now));
        
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
