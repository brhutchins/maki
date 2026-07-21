//! OpenCode-specific catalog helpers.
//!
//! Constants and key-resolution logic specific to the opencode catalogue.

use std::collections::HashMap;

use maki_storage::StateDir;
use tracing::debug;

use crate::providers::catalog::data::{
    CatalogIndex, CatalogProvider, ZEN_CATALOG_KEY, ZEN_MAKI_SLUG, maki_slug_for,
};

pub(super) const ALLOWED_NPM: &[&str] = &["@ai-sdk/openai-compatible", "@ai-sdk/anthropic"];

pub(super) const PUBLIC_KEY: &str = "public";
pub(crate) const GO_PROVIDER_ID: &str = "opencode-go";

fn saved_key(state_dir: &StateDir, slug: &str) -> Option<String> {
    maki_storage::auth::load_provider_credentials(state_dir, slug).map(|c| c.api_key)
}

fn resolve_provider_key(
    provider: &CatalogProvider,
    state_dir: &StateDir,
    saved_key_slug: &str,
    read_env: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    for var in &provider.env {
        if let Some(val) = read_env(var) {
            debug!(provider = %provider.name, var = %var, "api key resolved from env");
            return Some(val);
        }
        debug!(provider = %provider.name, var = %var, "env var not set");
    }
    if provider.env.iter().any(|v| v == "OPENCODE_API_KEY") {
        let key = saved_key(state_dir, saved_key_slug).or_else(|| {
            if saved_key_slug == ZEN_MAKI_SLUG {
                saved_key(state_dir, ZEN_CATALOG_KEY)
            } else {
                None
            }
        });
        if let Some(key) = key {
            debug!(provider = %provider.name, slug = saved_key_slug, "api key resolved from saved credentials");
            return Some(key);
        }
    }
    debug!(provider = %provider.name, "no api key available");
    None
}

pub(super) fn resolve_provider_keys(
    index: &CatalogIndex,
    state_dir: &StateDir,
) -> HashMap<String, Option<String>> {
    index
        .iter()
        .map(|(provider_id, provider)| {
            let slug = maki_slug_for(provider_id);
            let key = resolve_provider_key(provider, state_dir, slug, |v| std::env::var(v).ok());
            (provider_id.clone(), key)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use maki_storage::auth::ProviderCredentials;
    use tempfile::TempDir;

    fn empty_state_dir() -> (TempDir, StateDir) {
        let tmp = TempDir::new().unwrap();
        let dir = StateDir::from_path(tmp.path().to_path_buf());
        (tmp, dir)
    }

    fn state_dir_with_key(slug: &str, key: &str) -> (TempDir, StateDir) {
        let (tmp, dir) = empty_state_dir();
        maki_storage::auth::save_provider_credentials(
            &dir,
            slug,
            &ProviderCredentials {
                api_key: key.to_string(),
                host: None,
            },
        )
        .unwrap();
        (tmp, dir)
    }

    fn state_dir_with_opencode_key(key: &str) -> (TempDir, StateDir) {
        state_dir_with_key("opencode-zen", key)
    }

    #[test]
    fn resolve_provider_key_returns_none_when_env_unset() {
        let (_tmp, state_dir) = empty_state_dir();
        let provider = CatalogProvider {
            name: "Test".into(),
            env: vec!["MAKI_TEST_UNUSED_VAR_1".into()],
            npm: "@ai-sdk/openai".into(),
            api: None,
            models: HashMap::new(),
        };
        let key = resolve_provider_key(&provider, &state_dir, "opencode", |v| {
            assert_eq!(v, "MAKI_TEST_UNUSED_VAR_1");
            None
        });
        assert!(key.is_none());
    }

    #[test]
    fn resolve_provider_key_returns_saved_when_env_unset() {
        let (_tmp, state_dir) = state_dir_with_opencode_key("from-saved");
        let provider = CatalogProvider {
            name: "Opencode".into(),
            env: vec!["OPENCODE_API_KEY".into()],
            npm: "@ai-sdk/openai-compatible".into(),
            api: Some("https://opencode.ai/zen/v1".into()),
            models: HashMap::new(),
        };
        let key = resolve_provider_key(&provider, &state_dir, "opencode-zen", |_| None);
        assert_eq!(key, Some("from-saved".into()));
    }

    #[test]
    fn resolve_provider_key_reads_legacy_saved_credentials() {
        let (_tmp, state_dir) = state_dir_with_key("opencode", "from-legacy-saved");
        let provider = CatalogProvider {
            name: "Opencode".into(),
            env: vec!["OPENCODE_API_KEY".into()],
            npm: "@ai-sdk/openai-compatible".into(),
            api: Some("https://opencode.ai/zen/v1".into()),
            models: HashMap::new(),
        };
        let key = resolve_provider_key(&provider, &state_dir, "opencode-zen", |_| None);
        assert_eq!(key, Some("from-legacy-saved".into()));
    }

    #[test]
    fn resolve_provider_key_env_takes_priority_over_saved() {
        let (_tmp, state_dir) = state_dir_with_opencode_key("from-saved");
        let provider = CatalogProvider {
            name: "Opencode".into(),
            env: vec!["OPENCODE_API_KEY".into()],
            npm: "@ai-sdk/openai-compatible".into(),
            api: Some("https://opencode.ai/zen/v1".into()),
            models: HashMap::new(),
        };
        let key = resolve_provider_key(&provider, &state_dir, "opencode-zen", |v| {
            if v == "OPENCODE_API_KEY" {
                Some("from-env".into())
            } else {
                None
            }
        });
        assert_eq!(key, Some("from-env".into()));
    }

    #[test]
    fn resolve_provider_key_returns_none_when_nothing_available() {
        let (_tmp, state_dir) = empty_state_dir();
        let provider = CatalogProvider {
            name: "Opencode".into(),
            env: vec!["OPENCODE_API_KEY".into()],
            npm: "@ai-sdk/openai-compatible".into(),
            api: Some("https://opencode.ai/zen/v1".into()),
            models: HashMap::new(),
        };
        let key = resolve_provider_key(&provider, &state_dir, "opencode-zen", |_| None);
        assert!(key.is_none());
    }
}
