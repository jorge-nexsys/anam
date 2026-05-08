//! Authentication middleware for the AnamDB server.
//!
//! Provides API key validation against a pluggable storage backend.
//! In production, swap [`DummyAuthenticator`] for a real database-backed
//! implementation.

use crate::core::error::{AnamError, Result};

/// Represents an authenticated tenant session.
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// Unique tenant identifier.
    pub tenant_id: String,
    /// The raw API key used for this session.
    pub api_key: String,
    /// Subscription tier governing rate limits and feature access.
    pub tier: SubscriptionTier,
}

/// Subscription tiers that govern rate limits, feature access, and billing.
#[derive(Debug, Clone, PartialEq)]
pub enum SubscriptionTier {
    /// Free tier — solo developers, students, and OSS projects.
    Community,
    /// Paid tier — startups and small teams.
    Pro,
    /// Paid tier — mid-market teams with multi-user needs.
    Team,
    /// Custom contracts — regulated industries, dedicated infrastructure.
    Enterprise,
}

/// Trait for pluggable authentication backends.
#[async_trait::async_trait]
pub trait Authenticator: Send + Sync {
    /// Validate a bearer token and return the associated [`AuthContext`].
    async fn authenticate(&self, token: &str) -> Result<AuthContext>;
}

/// A placeholder authenticator for local development and testing.
///
/// Accepts `sk-admin-secret` as an enterprise key and any `sk-*` prefix
/// as a community key. **Never use in production.**
pub struct DummyAuthenticator;

#[async_trait::async_trait]
impl Authenticator for DummyAuthenticator {
    async fn authenticate(&self, token: &str) -> Result<AuthContext> {
        if token == "sk-admin-secret" {
            Ok(AuthContext {
                tenant_id: "tenant-admin".into(),
                api_key: token.to_string(),
                tier: SubscriptionTier::Enterprise,
            })
        } else if token.starts_with("sk-") {
            Ok(AuthContext {
                tenant_id: "tenant-demo".into(),
                api_key: token.to_string(),
                tier: SubscriptionTier::Community,
            })
        } else {
            Err(AnamError::Internal("Invalid API key".into()))
        }
    }
}
