use maki_storage::StateDir;

use super::data::{CatalogProvider, EndpointType, maki_slug_for};

#[derive(Debug, Clone)]
pub struct ProviderData {
    pub slug: String,
    pub display_name: String,
    pub env_keys: Vec<String>,
    pub base_url: Option<String>,
    pub npm: String,
    pub api_format: EndpointType,
}

impl ProviderData {
    pub(crate) fn from_catalog_entry(provider_id: &str, provider: &CatalogProvider) -> Self {
        Self {
            slug: maki_slug_for(provider_id).to_string(),
            display_name: provider.name.clone(),
            env_keys: provider.env.clone(),
            base_url: provider.api.clone(),
            npm: provider.npm.clone(),
            api_format: EndpointType::from_npm(&provider.npm),
        }
    }

    pub fn load_key_from_storage(&self, state_dir: &StateDir) -> Option<String> {
        maki_storage::auth::load_provider_credentials(state_dir, &self.slug).map(|c| c.api_key)
    }

    pub fn env_key_set(&self) -> Option<&str> {
        self.env_keys
            .iter()
            .find(|v| std::env::var(v).is_ok())
            .map(|s| s.as_str())
    }

    pub fn resolve_api_key(&self, state_dir: &StateDir) -> Option<String> {
        self.env_keys
            .iter()
            .find_map(|v| std::env::var(v).ok())
            .or_else(|| self.load_key_from_storage(state_dir))
    }
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

    fn test_provider() -> ProviderData {
        ProviderData {
            slug: "test-provider".into(),
            display_name: "Test Provider".into(),
            env_keys: vec!["TEST_API_KEY".into()],
            base_url: Some("https://test.api/v1".into()),
            npm: "@ai-sdk/openai-compatible".into(),
            api_format: EndpointType::ChatCompletions,
        }
    }

    #[test]
    fn load_key_from_storage_returns_key_when_saved() {
        let (_tmp, state_dir) = state_dir_with_key("test-provider", "sk-test-123");
        let pd = test_provider();
        assert_eq!(
            pd.load_key_from_storage(&state_dir),
            Some("sk-test-123".into())
        );
    }

    #[test]
    fn load_key_from_storage_returns_none_when_not_saved() {
        let (_tmp, state_dir) = empty_state_dir();
        let pd = test_provider();
        assert!(pd.load_key_from_storage(&state_dir).is_none());
    }

    #[test]
    fn env_key_set_returns_env_var_name_when_set() {
        let var = format!("MAKI_TEST_ENV_KEY_{}", fastrand::u32(..));
        unsafe { std::env::set_var(&var, "dummy") };
        let pd = ProviderData {
            slug: "test".into(),
            display_name: "Test".into(),
            env_keys: vec![
                format!("MAKI_NONEXISTENT_{}", fastrand::u32(..)),
                var.clone(),
            ],
            base_url: None,
            npm: "@ai-sdk/openai-compatible".into(),
            api_format: EndpointType::ChatCompletions,
        };
        assert_eq!(pd.env_key_set(), Some(var.as_str()));
        unsafe { std::env::remove_var(&var) };
    }

    #[test]
    fn env_key_set_returns_none_when_no_env_vars_set() {
        let pd = ProviderData {
            slug: "test".into(),
            display_name: "Test".into(),
            env_keys: vec![format!("MAKI_NONEXISTENT_{}", fastrand::u32(..))],
            base_url: None,
            npm: "@ai-sdk/openai-compatible".into(),
            api_format: EndpointType::ChatCompletions,
        };
        assert!(pd.env_key_set().is_none());
    }

    #[test]
    fn resolve_api_key_prefers_env_over_storage() {
        let var = format!("MAKI_TEST_ENV_FIRST_{}", fastrand::u32(..));
        unsafe { std::env::set_var(&var, "from-env") };
        let (_tmp, state_dir) = state_dir_with_key("test-env-first", "from-storage");
        let pd = ProviderData {
            slug: "test-env-first".into(),
            display_name: "Test".into(),
            env_keys: vec![var.clone()],
            base_url: None,
            npm: "@ai-sdk/openai-compatible".into(),
            api_format: EndpointType::ChatCompletions,
        };
        assert_eq!(pd.resolve_api_key(&state_dir), Some("from-env".into()));
        unsafe { std::env::remove_var(&var) };
    }

    #[test]
    fn resolve_api_key_falls_back_to_storage() {
        let (_tmp, state_dir) = state_dir_with_key("test-fallback", "from-storage");
        let pd = ProviderData {
            slug: "test-fallback".into(),
            display_name: "Test".into(),
            env_keys: vec![format!("MAKI_NONEXISTENT_{}", fastrand::u32(..))],
            base_url: None,
            npm: "@ai-sdk/openai-compatible".into(),
            api_format: EndpointType::ChatCompletions,
        };
        assert_eq!(pd.resolve_api_key(&state_dir), Some("from-storage".into()));
    }

    #[test]
    fn resolve_api_key_returns_none_when_nothing_available() {
        let (_tmp, state_dir) = empty_state_dir();
        let pd = ProviderData {
            slug: "test-none".into(),
            display_name: "Test".into(),
            env_keys: vec![format!("MAKI_NONEXISTENT_{}", fastrand::u32(..))],
            base_url: None,
            npm: "@ai-sdk/openai-compatible".into(),
            api_format: EndpointType::ChatCompletions,
        };
        assert!(pd.resolve_api_key(&state_dir).is_none());
    }
}
