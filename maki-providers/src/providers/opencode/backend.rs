//! Unified Zen/Go catalog dispatch.
//!
//! Both providers turn a raw [`catalog::CatalogIndex`] into the same
//! [`Catalog`] shape (entries keyed by `(sub_provider, model_id)`, auths
//! keyed by sub_provider). They differ only in which entries they admit
//! and how they resolve a user-supplied model id.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tracing::debug;

use super::catalog::{ALLOWED_NPM, GO_PROVIDER_ID, PUBLIC_KEY};
use crate::AgentError;
use crate::model::{ModelInfo, ModelPricing};
use crate::providers::ResolvedAuth;
use crate::providers::catalog::data as cat_data;
use crate::providers::catalog::data::{
    CatalogIndex, CatalogProvider, EndpointType, Meta, ModelKey, ZEN_CATALOG_KEY, ZEN_MAKI_SLUG,
};
use crate::providers::catalog::registry::{
    self as cat_registry, CatalogModelInfo, CatalogProviderInfo,
};

pub(crate) struct Catalog {
    pub entries: HashMap<ModelKey, Meta>,
    pub auths: HashMap<String, ResolvedAuth>,
    pub anthropic_auths: HashMap<String, ResolvedAuth>,
    pub providers: HashMap<String, CatalogProvider>,
}

const BLOCKED_PROVIDERS: &[&str] = &["zai-coding-plan", "github-copilot"];

impl Catalog {
    pub(crate) fn empty() -> Self {
        Self {
            entries: HashMap::new(),
            auths: HashMap::new(),
            anthropic_auths: HashMap::new(),
            providers: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Backend {
    Zen,
    Go,
}

impl Backend {
    /// Whether auth override from a dynamic provider should replace the
    /// catalog auth. Zen only honours overrides for the "opencode" sub-provider
    /// (third-party sub-providers keep their catalog auth); Go honours any override.
    fn allow_auth_override(&self, meta: &Meta) -> bool {
        match self {
            Self::Zen => meta.provider_id == ZEN_CATALOG_KEY,
            Self::Go => true,
        }
    }

    pub(crate) fn build_catalog(
        &self,
        index: CatalogIndex,
        keys: &HashMap<String, Option<String>>,
        enable_free_models: bool,
    ) -> Catalog {
        match self {
            Self::Zen => build_zen_catalog(index, keys, enable_free_models),
            Self::Go => build_go_catalog(index, keys),
        }
    }

    pub(crate) fn lookup(
        &self,
        catalog: &Catalog,
        model_id: &str,
        auth_override: Option<&Arc<Mutex<ResolvedAuth>>>,
    ) -> Result<(Meta, ResolvedAuth), AgentError> {
        let key = self.lookup_key(model_id);
        let meta = catalog
            .entries
            .get(&key)
            .cloned()
            .ok_or_else(|| AgentError::Config {
                message: format!("model '{model_id}' not found in catalog"),
            })?;
        let catalog_auth = match meta.api_format {
            EndpointType::Messages => catalog.anthropic_auths.get(&meta.provider_id),
            EndpointType::ChatCompletions => catalog.auths.get(&meta.provider_id),
        }
        .cloned()
        .ok_or_else(|| AgentError::Config {
            message: format!(
                "auth for provider '{}' not found in catalog",
                meta.provider_id
            ),
        })?;
        let auth = match auth_override {
            Some(o) if self.allow_auth_override(&meta) => o.lock().unwrap().clone(),
            _ => catalog_auth,
        };
        Ok((meta, auth))
    }

    fn lookup_key(&self, model_id: &str) -> ModelKey {
        match self {
            // Zen: "nvidia/openai/gpt-oss-120b" -> ("nvidia", "openai/gpt-oss-120b")
            Self::Zen => model_id
                .split_once('/')
                .map(|(sub, rest)| (sub.to_string(), rest.to_string()))
                .unwrap_or_else(|| (ZEN_CATALOG_KEY.to_string(), model_id.to_string())),
            Self::Go => (GO_PROVIDER_ID.to_string(), model_id.to_string()),
        }
    }

    /// Strip the backend-specific prefix from a model id to get the bare
    /// upstream id.
    pub(crate) fn strip_prefix(&self, model_id: &str, prefix: &str) -> String {
        let needle = format!("{prefix}/");
        model_id
            .strip_prefix(&needle)
            .unwrap_or(model_id)
            .to_string()
    }

    /// Normalize a user-supplied id to catalog form.
    pub(crate) fn normalize_model_id(&self, model_id: &str) -> String {
        match self {
            Self::Zen if !model_id.contains('/') => format!("{ZEN_CATALOG_KEY}/{model_id}"),
            Self::Go => model_id
                .strip_prefix(&format!("{GO_PROVIDER_ID}/"))
                .unwrap_or(model_id)
                .to_string(),
            _ => model_id.to_string(),
        }
    }

    /// Display form of a catalog id: the default Zen sub-provider prefix is
    /// dropped, but third-party sub-providers keep their vendor segment so
    /// picker labels cannot collide.
    pub(crate) fn display_model_id<'a>(&self, model_id: &'a str) -> &'a str {
        match self {
            Self::Zen => match model_id.split_once('/') {
                Some((sub, rest)) if sub == ZEN_CATALOG_KEY => rest,
                _ => model_id,
            },
            Self::Go => model_id,
        }
    }

    pub(crate) fn all_models(&self, catalog: &Catalog) -> Vec<ModelInfo> {
        let mut models: Vec<ModelInfo> = catalog
            .entries
            .iter()
            .filter(|((provider, _), _)| match self {
                // Zen catalog's all_models only returns first-party models;
                // third-party models are listed by individual CatalogProvider instances.
                Self::Zen => provider == ZEN_CATALOG_KEY,
                Self::Go => true,
            })
            .map(|((provider, model_id), meta)| {
                let id = match self {
                    Self::Zen => format!("{provider}/{model_id}"),
                    Self::Go => model_id.clone(),
                };

                ModelInfo {
                    id,
                    context_window: Some(meta.context),
                    max_output_tokens: Some(meta.output),
                    pricing: Some(ModelPricing {
                        input: meta.input_price,
                        output: meta.output_price,
                        cache_read: meta.cache_read,
                        cache_write: meta.cache_write,
                        fast: None,
                    }),
                    supports_thinking: Some(meta.supports_thinking),
                    supports_vision: Some(meta.vision),
                    provider_info: None,
                }
            })
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }
}

fn admit_provider(provider: &CatalogProvider) -> bool {
    if !ALLOWED_NPM.contains(&provider.npm.as_str()) {
        debug!(npm = %provider.npm, "skipping provider: unsupported npm package");
        return false;
    }
    if provider.api.is_none() {
        debug!(provider = %provider.name, "skipping: no API URL in catalog");
        return false;
    }
    true
}

fn model_is_free(model: &cat_data::CatalogModel) -> bool {
    let cost = model.cost.as_ref();
    cost.and_then(|c| c.input).unwrap_or(0.0) == 0.0
        && cost.and_then(|c| c.output).unwrap_or(0.0) == 0.0
}

fn build_zen_catalog(
    index: CatalogIndex,
    keys: &HashMap<String, Option<String>>,
    enable_free_models: bool,
) -> Catalog {
    let mut catalog = Catalog::empty();
    let registry = build_zen_registry(&index, enable_free_models);

    for (provider_id, provider) in &index {
        // The Go provider has its own catalog; don't leak its models into Zen.
        if provider_id == GO_PROVIDER_ID {
            continue;
        }
        if provider_id != ZEN_CATALOG_KEY
            && (BLOCKED_PROVIDERS.contains(&provider_id.as_str())
                || maki_config::providers::builtin_provider(provider_id).is_some())
        {
            debug!(provider = %provider_id, "skipping provider supported by a builtin provider");
            continue;
        }
        if !admit_provider(provider) {
            continue;
        }
        catalog
            .providers
            .insert(provider_id.clone(), provider.clone());

        let is_opencode = provider_id == ZEN_CATALOG_KEY;
        let api_format = EndpointType::from_npm(&provider.npm);

        let key = keys.get(provider_id).and_then(|k| k.as_ref());
        let has_key = key.is_some();
        // Providers without a key are admitted only for the Zen catalog key,
        // which has a public fallback for free models.
        if !has_key && !is_opencode {
            continue;
        }
        let api_key = if is_opencode {
            key.cloned().unwrap_or_else(|| PUBLIC_KEY.to_string())
        } else {
            key.cloned().unwrap_or_default()
        };
        let auth = provider.build_auth(api_key.clone(), EndpointType::ChatCompletions);
        let anthropic_auth = provider.build_auth(api_key, EndpointType::Messages);

        let mut count = 0u32;
        for (model_id, model_data) in &provider.models {
            let is_free = model_is_free(model_data);
            if is_free && !enable_free_models {
                continue;
            }
            // Paid models in the opencode catalog require a real key.
            if !(has_key || is_opencode && is_free) {
                continue;
            }
            let meta = cat_data::model_meta(model_data, provider_id, api_format);
            catalog
                .entries
                .insert((provider_id.clone(), model_id.clone()), meta);
            count += 1;
        }
        if count > 0 {
            catalog.auths.insert(provider_id.clone(), auth);
            catalog
                .anthropic_auths
                .insert(provider_id.clone(), anthropic_auth);
            debug!(
                provider = %provider_id,
                models = count,
                "catalog provider registered",
            );
        }
    }

    cat_registry::replace(registry);
    catalog
}

fn build_zen_registry(
    index: &CatalogIndex,
    enable_free_models: bool,
) -> HashMap<String, CatalogProviderInfo> {
    let mut registry = HashMap::new();

    for (provider_id, provider) in index {
        if provider_id == GO_PROVIDER_ID {
            continue;
        }
        if provider_id != ZEN_CATALOG_KEY
            && (BLOCKED_PROVIDERS.contains(&provider_id.as_str())
                || maki_config::providers::builtin_provider(provider_id).is_some())
        {
            continue;
        }
        if !admit_provider(provider) {
            continue;
        }
        if provider_id == ZEN_CATALOG_KEY {
            continue;
        }

        let api_format = EndpointType::from_npm(&provider.npm);
        let cat_models: Vec<CatalogModelInfo> = provider
            .models
            .iter()
            .filter(|(_, model_data)| enable_free_models || !model_is_free(model_data))
            .map(|(model_id, model_data)| CatalogModelInfo {
                id: model_id.clone(),
                meta: cat_data::model_meta(model_data, provider_id, api_format),
            })
            .collect();

        let has_thinking = cat_models.iter().any(|m| m.meta.supports_thinking);
        let has_vision = cat_models.iter().any(|m| m.meta.vision);
        let mut caps = Vec::new();
        if has_thinking {
            caps.push("thinking support");
        }
        if has_vision {
            caps.push("vision support");
        }
        let features = if caps.is_empty() {
            format!("{} models", provider.name)
        } else {
            format!("{} models with {}", provider.name, caps.join(", "))
        };

        registry.insert(
            provider_id.clone(),
            CatalogProviderInfo {
                display_name: provider.name.clone(),
                env_keys: provider.env.clone(),
                base_url: provider.api.clone(),
                features,
                models: cat_models,
            },
        );
    }

    registry
}

fn build_go_catalog(index: CatalogIndex, keys: &HashMap<String, Option<String>>) -> Catalog {
    let mut catalog = Catalog::empty();
    let Some(provider) = index.get(GO_PROVIDER_ID) else {
        debug!("opencode-go entry not found in catalog");
        return catalog;
    };
    if !admit_provider(provider) {
        return catalog;
    }
    catalog
        .providers
        .insert(GO_PROVIDER_ID.to_string(), provider.clone());

    let key = keys.get(GO_PROVIDER_ID).and_then(|k| k.as_ref());
    if key.is_none() {
        return catalog;
    }
    let api_key = key.cloned().unwrap_or_default();
    let auth = provider.build_auth(api_key.clone(), EndpointType::ChatCompletions);
    let anthropic_auth = provider.build_auth(api_key, EndpointType::Messages);
    let api_format = EndpointType::from_npm(&provider.npm);
    let mut count = 0u32;
    for (model_id, model_data) in &provider.models {
        let meta = cat_data::model_meta(model_data, GO_PROVIDER_ID, api_format);
        catalog
            .entries
            .insert((GO_PROVIDER_ID.to_string(), model_id.clone()), meta);
        count += 1;
    }
    catalog.auths.insert(GO_PROVIDER_ID.to_string(), auth);
    catalog
        .anthropic_auths
        .insert(GO_PROVIDER_ID.to_string(), anthropic_auth);
    debug!(
        provider = GO_PROVIDER_ID,
        models = count,
        "opencode-go catalog provider registered",
    );
    catalog
}

/// Build a catalog from scratch, trying cache first then remote fetch.
pub(super) async fn build_catalog_async(backend: Backend) -> Catalog {
    let state_dir = maki_storage::StateDir::resolve().unwrap_or_else(|_| {
        maki_storage::StateDir::from_path(std::env::temp_dir().join("maki-fallback-state"))
    });

    let enable_free_models = match backend {
        Backend::Zen => {
            let providers = maki_config::providers::ProvidersConfig::load();
            providers
                .get(ZEN_MAKI_SLUG)
                .or_else(|| providers.get(ZEN_CATALOG_KEY))
                .and_then(|d| d.enable_free_models)
                .unwrap_or(false)
        }
        Backend::Go => false,
    };

    if let Some(index) = cat_data::load_cached_async().await {
        debug!(backend = ?backend, "using cached catalog");
        let keys = super::catalog::resolve_provider_keys(&index, &state_dir);
        let catalog = backend.build_catalog(index, &keys, enable_free_models);
        crate::model_registry::migrate_tier_overrides(&state_dir);
        return catalog;
    }

    let client = cat_data::http_client(match backend {
        Backend::Zen => "opencode-zen catalog",
        Backend::Go => "opencode-go catalog",
    });

    match cat_data::fetch_remote_async(&client).await {
        Ok(index) => {
            cat_data::save_cached_async(&index).await;
            let keys = super::catalog::resolve_provider_keys(&index, &state_dir);
            let catalog = backend.build_catalog(index, &keys, enable_free_models);
            crate::model_registry::migrate_tier_overrides(&state_dir);
            catalog
        }
        Err(e) => {
            warn_for(backend, &e);
            Catalog::empty()
        }
    }
}

fn warn_for(backend: Backend, err: &crate::AgentError) {
    match backend {
        Backend::Zen => tracing::warn!(error = %err, "catalog fetch failed, using empty catalog"),
        Backend::Go => {
            tracing::warn!(error = %err, "opencode-go catalog fetch failed, using empty catalog")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::catalog::data::{
        CatalogCost, CatalogLimits, CatalogModel, CatalogProvider,
    };
    use crate::providers::opencode::catalog::PUBLIC_KEY;
    use test_case::test_case;

    fn provider_with_models(
        name: &str,
        env: Vec<&str>,
        npm: &str,
        api: Option<&str>,
        models: HashMap<String, CatalogModel>,
    ) -> CatalogProvider {
        CatalogProvider {
            name: name.into(),
            env: env.into_iter().map(String::from).collect(),
            npm: npm.into(),
            api: api.map(String::from),
            models,
        }
    }

    fn paid_model() -> CatalogModel {
        CatalogModel {
            limit: None,
            cost: Some(CatalogCost {
                input: Some(5.0),
                output: Some(25.0),
                cache_read: None,
                cache_write: None,
            }),
            provider: None,
            attachment: None,
            modalities: None,
            reasoning: None,
        }
    }

    fn free_model() -> CatalogModel {
        CatalogModel {
            limit: None,
            cost: Some(CatalogCost {
                input: Some(0.0),
                output: Some(0.0),
                cache_read: None,
                cache_write: None,
            }),
            provider: None,
            attachment: None,
            modalities: None,
            reasoning: None,
        }
    }

    #[test]
    fn lookup_zen_splits_sub_provider_and_model() {
        let mut models = HashMap::new();
        models.insert("openai/gpt-oss-120b".into(), free_model());
        let mut index = HashMap::new();
        index.insert(
            "nvidia".into(),
            provider_with_models(
                "NVIDIA",
                vec!["MAKI_TEST_NVIDIA_DIR"],
                "@ai-sdk/openai-compatible",
                Some("https://nvapi.xyz/v1"),
                models,
            ),
        );
        let keys = HashMap::from([("nvidia".into(), Some("key".into()))]);
        let catalog = Backend::Zen.build_catalog(index, &keys, true);
        let (meta, _) = Backend::Zen
            .lookup(&catalog, "nvidia/openai/gpt-oss-120b", None)
            .unwrap();
        assert_eq!(meta.provider_id, "nvidia");
    }

    #[test]
    fn lookup_zen_nested_model_with_sub_provider() {
        let mut models = HashMap::new();
        models.insert("deepseek-ai/DeepSeek-R1".into(), free_model());
        let mut index = HashMap::new();
        index.insert(
            "fireworks".into(),
            provider_with_models(
                "Fireworks",
                vec!["MAKI_TEST_FIRE_DEEP"],
                "@ai-sdk/openai-compatible",
                Some("https://fireworks.ai/v1"),
                models,
            ),
        );
        let keys = HashMap::from([("fireworks".into(), Some("key".into()))]);
        let catalog = Backend::Zen.build_catalog(index, &keys, true);
        let (meta, _) = Backend::Zen
            .lookup(&catalog, "fireworks/deepseek-ai/DeepSeek-R1", None)
            .unwrap();
        assert_eq!(meta.provider_id, "fireworks");
    }

    #[test]
    fn lookup_zen_collisions_preserve_both_entries() {
        let mut models = HashMap::new();
        models.insert("shared-model".into(), free_model());
        let mut index = HashMap::new();
        index.insert(
            "opencode".into(),
            provider_with_models(
                "Opencode",
                vec![],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/v1"),
                models.clone(),
            ),
        );
        index.insert(
            "other-vendor".into(),
            provider_with_models(
                "Other",
                vec!["MAKI_TEST_OTHER_COLL"],
                "@ai-sdk/openai-compatible",
                Some("https://other.api/v1"),
                models,
            ),
        );
        let keys = HashMap::from([("other-vendor".into(), Some("key".into()))]);
        let catalog = Backend::Zen.build_catalog(index, &keys, true);
        assert!(
            catalog
                .entries
                .contains_key(&(ZEN_CATALOG_KEY.into(), "shared-model".into()))
        );
        assert!(
            catalog
                .entries
                .contains_key(&("other-vendor".into(), "shared-model".into()))
        );
        assert_eq!(catalog.entries.len(), 2);
        let (meta, _) = Backend::Zen
            .lookup(&catalog, "opencode/shared-model", None)
            .unwrap();
        assert_eq!(meta.provider_id, "opencode");
    }

    #[test]
    fn lookup_zen_filters_nonfree_without_key() {
        let mut models = HashMap::new();
        models.insert("paid-model".into(), paid_model());
        models.insert("free-model".into(), free_model());
        let mut index = HashMap::new();
        index.insert(
            "some-vendor".into(),
            provider_with_models(
                "Vendor",
                vec!["MAKI_TEST_VENDOR_FILTER"],
                "@ai-sdk/openai-compatible",
                Some("https://vendor.api/v1"),
                models,
            ),
        );
        let keys = HashMap::new();
        let catalog = Backend::Zen.build_catalog(index, &keys, true);
        assert!(catalog.entries.is_empty());
    }

    #[test]
    fn zen_catalog_excludes_builtin_and_blocked_providers() {
        let provider = |name: &str| {
            provider_with_models(
                name,
                vec!["MAKI_TEST_PROVIDER_KEY"],
                "@ai-sdk/openai-compatible",
                Some("https://provider.api/v1"),
                HashMap::from([(String::from("model"), paid_model())]),
            )
        };
        let index = HashMap::from([
            (String::from("zai"), provider("Z.AI")),
            (
                String::from("zai-coding-plan"),
                provider("Z.AI Coding Plan"),
            ),
            (String::from("github-copilot"), provider("GitHub Copilot")),
        ]);
        let keys = HashMap::from([
            (String::from("zai"), Some(String::from("zai-key"))),
            (
                String::from("zai-coding-plan"),
                Some(String::from("coding-plan-key")),
            ),
            (
                String::from("github-copilot"),
                Some(String::from("github-key")),
            ),
        ]);

        let catalog = Backend::Zen.build_catalog(index, &keys, true);

        assert!(catalog.entries.is_empty());
        assert!(catalog.providers.is_empty());
    }

    #[test]
    fn missing_model_cost_is_free() {
        let model = CatalogModel {
            limit: None,
            cost: None,
            provider: None,
            attachment: None,
            modalities: None,
            reasoning: None,
        };

        assert!(model_is_free(&model));
    }

    #[test]
    fn lookup_zen_includes_free_models_for_opencode_without_key() {
        let mut models = HashMap::new();
        models.insert("paid-model".into(), paid_model());
        models.insert("free-model".into(), free_model());
        let mut index = HashMap::new();
        index.insert(
            ZEN_CATALOG_KEY.into(),
            provider_with_models(
                "Opencode",
                vec!["MAKI_TEST_OPENCODE_FREE"],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/v1"),
                models,
            ),
        );
        let keys = HashMap::new();
        let catalog = Backend::Zen.build_catalog(index, &keys, true);
        assert!(
            catalog
                .entries
                .contains_key(&(ZEN_CATALOG_KEY.into(), "free-model".into()))
        );
        assert!(
            !catalog
                .entries
                .contains_key(&(ZEN_CATALOG_KEY.into(), "paid-model".into()))
        );
        assert!(catalog.auths.contains_key(ZEN_CATALOG_KEY));
    }

    #[test]
    fn lookup_zen_all_models_with_key() {
        let mut models = HashMap::new();
        models.insert("paid-model".into(), paid_model());
        models.insert("free-model".into(), free_model());
        let mut index = HashMap::new();
        index.insert(
            ZEN_CATALOG_KEY.into(),
            provider_with_models(
                "Opencode",
                vec!["MAKI_TEST_OPENCODE_ALL"],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/v1"),
                models,
            ),
        );
        let keys = HashMap::from([(ZEN_CATALOG_KEY.into(), Some("real-key".into()))]);
        let catalog = Backend::Zen.build_catalog(index, &keys, true);
        assert!(
            catalog
                .entries
                .contains_key(&(ZEN_CATALOG_KEY.into(), "free-model".into()))
        );
        assert!(
            catalog
                .entries
                .contains_key(&(ZEN_CATALOG_KEY.into(), "paid-model".into()))
        );
        assert!(catalog.auths.contains_key(ZEN_CATALOG_KEY));
    }

    #[test]
    fn lookup_zen_hides_free_models_when_disabled() {
        let mut models = HashMap::new();
        models.insert("paid-model".into(), paid_model());
        models.insert("free-model".into(), free_model());
        let mut index = HashMap::new();
        index.insert(
            ZEN_CATALOG_KEY.into(),
            provider_with_models(
                "Opencode",
                vec!["MAKI_TEST_OPENCODE_NOFREE"],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/v1"),
                models,
            ),
        );
        let keys = HashMap::from([(ZEN_CATALOG_KEY.into(), Some("real-key".into()))]);
        let catalog = Backend::Zen.build_catalog(index, &keys, false);
        assert!(
            !catalog
                .entries
                .contains_key(&(ZEN_CATALOG_KEY.into(), "free-model".into()))
        );
        assert!(
            catalog
                .entries
                .contains_key(&(ZEN_CATALOG_KEY.into(), "paid-model".into()))
        );
        assert!(catalog.auths.contains_key(ZEN_CATALOG_KEY));
    }

    #[test]
    fn lookup_zen_no_models_without_key_when_disabled() {
        let mut models = HashMap::new();
        models.insert("paid-model".into(), paid_model());
        models.insert("free-model".into(), free_model());
        let mut index = HashMap::new();
        index.insert(
            ZEN_CATALOG_KEY.into(),
            provider_with_models(
                "Opencode",
                vec!["MAKI_TEST_OPENCODE_EMPTY"],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/v1"),
                models,
            ),
        );
        let keys = HashMap::new();
        let catalog = Backend::Zen.build_catalog(index, &keys, false);
        assert!(catalog.entries.is_empty());
        assert!(!catalog.auths.contains_key(ZEN_CATALOG_KEY));
    }

    #[test]
    fn lookup_zen_skips_providers_without_api_url() {
        let mut index = HashMap::new();
        index.insert(
            "no-api".into(),
            provider_with_models(
                "No API",
                vec![],
                "@ai-sdk/openai-compatible",
                None,
                HashMap::new(),
            ),
        );
        let keys = HashMap::new();
        let catalog = Backend::Zen.build_catalog(index, &keys, true);
        assert!(catalog.entries.is_empty());
        assert!(catalog.auths.is_empty());
    }

    #[test]
    fn zen_free_key_admits_public_key_for_opencode() {
        let mut models = HashMap::new();
        models.insert("free-model".into(), free_model());
        let mut index = HashMap::new();
        index.insert(
            ZEN_CATALOG_KEY.into(),
            provider_with_models(
                "Opencode",
                vec!["OPENCODE_API_KEY"],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/v1"),
                models,
            ),
        );
        let keys = HashMap::new();
        let catalog = Backend::Zen.build_catalog(index, &keys, true);
        let auth = catalog.auths.get(ZEN_CATALOG_KEY).unwrap();
        assert_eq!(
            auth.headers,
            vec![("authorization".into(), format!("Bearer {}", PUBLIC_KEY))]
        );
    }

    #[test]
    fn zen_free_key_rejects_non_opencode_without_key() {
        let mut models = HashMap::new();
        models.insert("free-model".into(), free_model());
        let mut index = HashMap::new();
        index.insert(
            "other-vendor".into(),
            provider_with_models(
                "Other",
                vec!["MAKI_TEST_VENDOR_KEY"],
                "@ai-sdk/openai-compatible",
                Some("https://other.api/v1"),
                models,
            ),
        );
        let keys = HashMap::new();
        let catalog = Backend::Zen.build_catalog(index, &keys, true);
        assert!(catalog.entries.is_empty());
        assert!(catalog.auths.is_empty());
    }

    #[test]
    fn zen_catalog_excludes_go_provider() {
        let mut go_models = HashMap::new();
        go_models.insert(
            "deepseek-v4-flash".into(),
            CatalogModel {
                limit: None,
                cost: None,
                provider: None,
                attachment: None,
                modalities: None,
                reasoning: None,
            },
        );
        let mut zen_models = HashMap::new();
        zen_models.insert(
            "claude-sonnet".into(),
            CatalogModel {
                limit: None,
                cost: None,
                provider: None,
                attachment: None,
                modalities: None,
                reasoning: None,
            },
        );
        let mut index = HashMap::new();
        index.insert(
            ZEN_CATALOG_KEY.into(),
            provider_with_models(
                "Opencode",
                vec!["OPENCODE_API_KEY"],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/v1"),
                zen_models,
            ),
        );
        index.insert(
            GO_PROVIDER_ID.into(),
            provider_with_models(
                "Opencode Go",
                vec!["OPENCODE_API_KEY"],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/go/v1"),
                go_models,
            ),
        );
        let keys = HashMap::new();
        let catalog = Backend::Zen.build_catalog(index, &keys, true);
        let ids: Vec<String> = Backend::Zen
            .all_models(&catalog)
            .iter()
            .map(|m| m.id.clone())
            .collect();
        assert!(!ids.iter().any(|id| id.starts_with(GO_PROVIDER_ID)));
    }

    fn default_model() -> &'static str {
        use maki_config::providers::builtin_provider;
        builtin_provider("opencode-go")
            .expect("opencode-go builtin provider")
            .default_model
    }

    #[test]
    fn go_default_model_exists_in_catalog() {
        let default = default_model();
        let (_, model_id) = default.split_once('/').unwrap();
        let mut models = HashMap::new();
        models.insert(
            model_id.to_string(),
            CatalogModel {
                limit: Some(CatalogLimits {
                    context: Some(128_000),
                    input: None,
                    output: Some(64_000),
                }),
                cost: Some(CatalogCost {
                    input: Some(0.0),
                    output: Some(0.0),
                    cache_read: None,
                    cache_write: None,
                }),
                provider: None,
                attachment: None,
                modalities: None,
                reasoning: None,
            },
        );
        let mut index = HashMap::new();
        index.insert(
            GO_PROVIDER_ID.into(),
            provider_with_models(
                "Opencode Go",
                vec!["OPENCODE_API_KEY"],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/go/v1"),
                models,
            ),
        );
        let keys = HashMap::from([(GO_PROVIDER_ID.into(), Some("test-key".into()))]);
        let catalog = Backend::Go.build_catalog(index, &keys, false);
        let ids: Vec<String> = Backend::Go
            .all_models(&catalog)
            .iter()
            .map(|m| m.id.clone())
            .collect();
        assert!(
            ids.contains(&model_id.to_string()),
            "default model '{default}' not found in Go catalog models: {ids:?}"
        );
    }

    #[test]
    fn go_catalog_rebuild_with_credentials() {
        // Simulate: no credentials -> empty catalog, then add credentials -> models available.
        let mut index = HashMap::new();
        index.insert(
            GO_PROVIDER_ID.into(),
            provider_with_models(
                "Opencode Go",
                vec!["OPENCODE_API_KEY"],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/go/v1"),
                HashMap::from([("deepseek-v4-flash".into(), paid_model())]),
            ),
        );

        // Without keys -> empty catalog
        let no_keys = HashMap::new();
        let catalog = Backend::Go.build_catalog(index.clone(), &no_keys, false);
        assert!(
            catalog.entries.is_empty(),
            "expected empty catalog without credentials"
        );
        assert!(catalog.auths.is_empty());

        // With keys -> models available
        let with_keys = HashMap::from([(GO_PROVIDER_ID.into(), Some("new-key".into()))]);
        let catalog = Backend::Go.build_catalog(index.clone(), &with_keys, false);
        assert!(
            !catalog.entries.is_empty(),
            "expected non-empty catalog with credentials"
        );
        assert!(catalog.auths.contains_key(GO_PROVIDER_ID));
    }

    #[test]
    fn lookup_go_bare_id() {
        let mut models = HashMap::new();
        models.insert(
            "deepseek-v4-flash".into(),
            CatalogModel {
                limit: Some(CatalogLimits {
                    context: Some(128_000),
                    input: None,
                    output: Some(64_000),
                }),
                cost: Some(CatalogCost {
                    input: Some(0.0),
                    output: Some(0.0),
                    cache_read: None,
                    cache_write: None,
                }),
                provider: None,
                attachment: None,
                modalities: None,
                reasoning: None,
            },
        );
        let mut index = HashMap::new();
        index.insert(
            GO_PROVIDER_ID.into(),
            provider_with_models(
                "Opencode Go",
                vec!["OPENCODE_API_KEY"],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/go/v1"),
                models,
            ),
        );
        let keys = HashMap::from([(GO_PROVIDER_ID.into(), Some("go-key".into()))]);
        let catalog = Backend::Go.build_catalog(index, &keys, false);
        let (meta, auth) = Backend::Go
            .lookup(&catalog, "deepseek-v4-flash", None)
            .unwrap();
        assert_eq!(meta.provider_id, GO_PROVIDER_ID);
        assert_eq!(
            auth.base_url.as_deref(),
            Some("https://opencode.ai/zen/go/v1")
        );
    }

    #[test]
    fn lookup_go_includes_paid_with_key() {
        let mut models = HashMap::new();
        models.insert("paid-model".into(), paid_model());
        let mut index = HashMap::new();
        index.insert(
            GO_PROVIDER_ID.into(),
            provider_with_models(
                "Opencode Go",
                vec!["OPENCODE_API_KEY"],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/go/v1"),
                models,
            ),
        );
        let keys = HashMap::from([(GO_PROVIDER_ID.into(), Some("go-key".into()))]);
        let catalog = Backend::Go.build_catalog(index, &keys, false);
        assert!(catalog.auths.contains_key(GO_PROVIDER_ID));
        assert_eq!(catalog.entries.len(), 1);
    }

    #[test]
    fn lookup_go_skips_when_no_key() {
        let mut models = HashMap::new();
        models.insert(
            "any-model".into(),
            CatalogModel {
                limit: None,
                cost: None,
                provider: None,
                attachment: None,
                modalities: None,
                reasoning: None,
            },
        );
        let mut index = HashMap::new();
        index.insert(
            GO_PROVIDER_ID.into(),
            provider_with_models(
                "Opencode Go",
                vec!["OPENCODE_API_KEY"],
                "@ai-sdk/openai-compatible",
                Some("https://opencode.ai/zen/go/v1"),
                models,
            ),
        );
        let keys = HashMap::new();
        let catalog = Backend::Go.build_catalog(index, &keys, false);
        assert!(catalog.entries.is_empty());
        assert!(catalog.auths.is_empty());
    }

    #[test]
    fn lookup_go_ignores_other_providers() {
        let mut other_models = HashMap::new();
        other_models.insert(
            "other-model".into(),
            CatalogModel {
                limit: None,
                cost: None,
                provider: None,
                attachment: None,
                modalities: None,
                reasoning: None,
            },
        );
        let mut index = HashMap::new();
        index.insert(
            "other-provider".into(),
            provider_with_models(
                "Other",
                vec!["OPENCODE_API_KEY"],
                "@ai-sdk/openai-compatible",
                Some("https://other.api/v1"),
                other_models,
            ),
        );
        let keys = HashMap::from([("other-provider".into(), Some("key".into()))]);
        let catalog = Backend::Go.build_catalog(index, &keys, false);
        assert!(catalog.entries.is_empty());
        assert!(catalog.auths.is_empty());
    }

    #[test]
    fn lookup_go_anthropic_model() {
        let mut models = HashMap::new();
        models.insert(
            "anthropic-test-model".into(),
            CatalogModel {
                limit: Some(CatalogLimits {
                    context: Some(200_000),
                    input: None,
                    output: Some(16_000),
                }),
                cost: Some(CatalogCost {
                    input: Some(3.0),
                    output: Some(15.0),
                    cache_read: None,
                    cache_write: None,
                }),
                provider: None,
                attachment: None,
                modalities: None,
                reasoning: None,
            },
        );
        let mut index = HashMap::new();
        index.insert(
            GO_PROVIDER_ID.into(),
            provider_with_models(
                "Opencode Go Anthropic",
                vec!["OPENCODE_API_KEY"],
                "@ai-sdk/anthropic",
                Some("https://opencode.ai/zen/go/v1"),
                models,
            ),
        );
        let keys = HashMap::from([(GO_PROVIDER_ID.into(), Some("ant-key".into()))]);
        let catalog = Backend::Go.build_catalog(index, &keys, false);
        assert!(catalog.anthropic_auths.contains_key(GO_PROVIDER_ID));
        assert!(catalog.auths.contains_key(GO_PROVIDER_ID));
        let (meta, auth) = Backend::Go
            .lookup(&catalog, "anthropic-test-model", None)
            .unwrap();
        assert_eq!(meta.api_format, EndpointType::Messages);
        assert!(auth.headers.iter().any(|(k, _)| k == "x-api-key"));
    }

    #[test_case(Backend::Zen, "claude-sonnet-4-5", "opencode/claude-sonnet-4-5" ; "zen_bare_gains_default_prefix")]
    #[test_case(Backend::Zen, "opencode/claude-sonnet-4-5", "opencode/claude-sonnet-4-5" ; "zen_prefixed_unchanged")]
    #[test_case(Backend::Zen, "nvidia/openai/gpt-oss-120b", "nvidia/openai/gpt-oss-120b" ; "zen_third_party_unchanged")]
    #[test_case(Backend::Go, "opencode-go/deepseek-v4-flash", "deepseek-v4-flash" ; "go_legacy_prefix_stripped")]
    #[test_case(Backend::Go, "deepseek-v4-flash", "deepseek-v4-flash" ; "go_bare_unchanged")]
    fn normalize_model_id_cases(backend: Backend, id: &str, expected: &str) {
        assert_eq!(backend.normalize_model_id(id), expected);
    }

    #[test_case("opencode/claude-sonnet-4-5", "claude-sonnet-4-5" ; "default_subprovider_stripped")]
    #[test_case("nvidia/openai/gpt-oss-120b", "nvidia/openai/gpt-oss-120b" ; "third_party_keeps_vendor")]
    #[test_case("nvidia/shared-model", "nvidia/shared-model" ; "colliding_labels_stay_distinct")]
    fn zen_display_model_id_cases(id: &str, expected: &str) {
        assert_eq!(Backend::Zen.display_model_id(id), expected);
    }
}
