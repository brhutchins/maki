use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use tracing::debug;

use super::data::Meta;
use crate::AgentError;
use crate::providers::ResolvedAuth;

#[derive(Debug, Clone)]
pub(crate) struct CatalogProviderInfo {
    pub display_name: String,
    pub env_keys: Vec<String>,
    pub base_url: Option<String>,
    pub features: String,
    pub models: Vec<CatalogModelInfo>,
}

#[derive(Debug, Clone)]
pub(crate) struct CatalogModelInfo {
    pub id: String,
    pub meta: Meta,
}

type Registry = HashMap<String, CatalogProviderInfo>;

static CATALOG_REGISTRY: OnceLock<RwLock<Registry>> = OnceLock::new();

fn registry() -> &'static RwLock<Registry> {
    CATALOG_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

pub(crate) fn replace(infos: HashMap<String, CatalogProviderInfo>) {
    *registry().write().unwrap() = infos;
}

pub(crate) fn get(slug: &str) -> Option<CatalogProviderInfo> {
    registry().read().unwrap().get(slug).cloned()
}

pub(crate) fn all_slugs() -> Vec<String> {
    registry().read().unwrap().keys().cloned().collect()
}

pub(crate) fn contains(slug: &str) -> bool {
    registry().read().unwrap().contains_key(slug)
}

#[cfg(test)]
pub(crate) fn clear() {
    registry().write().unwrap().clear();
}

pub(crate) fn resolve_auth_for_catalog(
    slug: &str,
    env_keys: &[String],
) -> Result<ResolvedAuth, AgentError> {
    for var in env_keys {
        if let Ok(val) = std::env::var(var) {
            debug!(slug, var = %var, "catalog provider key resolved from env");
            return Ok(ResolvedAuth::bearer(&val));
        }
    }
    if let Some(dir) = maki_storage::StateDir::resolve().ok()
        && let Some(creds) = maki_storage::auth::load_provider_credentials(&dir, slug)
    {
        debug!(slug, "catalog provider key resolved from saved credentials");
        return Ok(ResolvedAuth::bearer(&creds.api_key));
    }
    Err(AgentError::Config {
        message: format!(
            "catalog provider '{slug}' not configured — set one of {:?} or run `maki auth login {slug}`",
            env_keys
        ),
    })
}
