//! Authentication middleware for the AnamDB server.
//!
//! Currently provides API key validation against a simple storage backend.
//! In the future, this will tie into a robust tenant/billing database.

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::core::error::{AnamError, Result};

/// Represents an authenticated tenant session.
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub tenant_id: String,
    pub api_key: String,
    pub tier: SubscriptionTier,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SubscriptionTier {
    Community,
    Pro,
    Team,
    Enterprise,
}

/// Simple authenticator trait.
#[async_trait::async_trait]
pub trait Authenticator: Send + Sync {
    async fn authenticate(&self, token: &str) -> Result<AuthContext>;
}

/// A dummy authenticator for MVP testing.
/// In production, this would query SQLite or a metadata DB.
pub struct DummyAuthenticator;

#[async_trait::async_trait]
impl Authenticator for DummyAuthenticator {
    async fn authenticate(&self, token: &str) -> Result<AuthContext> {
        // Placeholder logic:
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
