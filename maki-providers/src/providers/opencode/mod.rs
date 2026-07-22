//! Opencode Zen and Opencode Go providers.
//!
//! Both backends read the same models.dev catalog and dispatch each model
//! to either an OpenAI-compatible chat completions endpoint or an
//! Anthropic messages endpoint depending on the provider's `npm` package.
//!
//! They differ only in which entries they admit from the catalog and how
//! they resolve auth, both of which live in [`backend`].
//!
//! # Access asymmetry
//!
//! Every Go user has access to Zen, but not vice versa. The Zen catalog
//! contains all non-Go providers plus its own free-key opencode entry.
//! The Go catalog admits only its own entry. Both run against the same
//! fetched `CatalogIndex`; the only difference is the filter and the
//! `enable_free_models` config. Consequently, Zen is always checked first
//! in `catalog_provider`, `catalog_providers`, and
//! `catalog_providers_if_available`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Duration;

use flume::Sender;
use isahc::HttpClient;
use serde_json::Value;
use tracing::warn;

use crate::providers::catalog::data::{EndpointType, maki_slug_for};
pub use crate::providers::catalog::provider_data::ProviderData;
use crate::providers::catalog::registry;
use crate::providers::catalog::request as cat_request;

use maki_storage::id::SessionRef;

use backend::Catalog;
use request::{GO_CHAT, ZEN_CHAT};

use crate::model::Model;
use crate::model_registry::model_registry;
use crate::provider::{BoxFuture, Provider, ProviderKind};
use crate::providers::openai_compat::OpenAiCompatProvider;
use crate::providers::{ResolvedAuth, Timeouts, http_client};
use crate::{AgentError, Message, ProviderEvent, RequestOptions, StreamResponse};

mod backend;
mod catalog;
mod request;

pub(crate) use backend::Backend;
pub(crate) use backend::catalog_models_to_info;

inventory::submit!(maki_config::providers::BuiltInProvider {
    slug: "opencode-zen",
    display_name: "Opencode Zen",
    protocol: maki_config::providers::Protocol::Openai,
    default_base_url: "https://opencode.ai/zen/v1",
    default_api_key_env: "OPENCODE_API_KEY",
    default_model: "opencode-zen/opencode/claude-sonnet-4-5",
    plans: None,
    login_url: Some("https://opencode.ai/auth"),
    needs_url: false,
});

inventory::submit!(maki_config::providers::BuiltInProvider {
    slug: "opencode-go",
    display_name: "Opencode Go",
    protocol: maki_config::providers::Protocol::Openai,
    default_base_url: "https://opencode.ai/zen/go/v1",
    default_api_key_env: "OPENCODE_API_KEY",
    default_model: "opencode-go/deepseek-v4-flash",
    plans: None,
    login_url: Some("https://opencode.ai/auth"),
    needs_url: false,
});

/// One provider for both Zen and Go. Behavior is selected at construction
/// time by [`Backend`]; see [`Opencode::zen`] and [`Opencode::go`].
pub struct Opencode {
    backend: Backend,
    client: HttpClient,
    chat_compat: OpenAiCompatProvider,
    auth: Option<Arc<Mutex<ResolvedAuth>>>,
    system_prefix: Option<String>,
    stream_timeout: Duration,
}

static ZEN_CATALOG: OnceLock<RwLock<Catalog>> = OnceLock::new();
static GO_CATALOG: OnceLock<RwLock<Catalog>> = OnceLock::new();

static ZEN_CATALOG_REBUILD: AtomicBool = AtomicBool::new(false);
static GO_CATALOG_REBUILD: AtomicBool = AtomicBool::new(false);

impl Opencode {
    fn new_impl(
        backend: Backend,
        timeouts: Timeouts,
        auth: Option<Arc<Mutex<ResolvedAuth>>>,
    ) -> Self {
        let slot = catalog_for(backend);
        slot.get_or_init(|| RwLock::new(smol::block_on(backend::build_catalog_async(backend))));
        let kind = match backend {
            Backend::Zen => ProviderKind::OpencodeZen,
            Backend::Go => ProviderKind::OpencodeGo,
        };

        let empty = slot.get().unwrap().read().unwrap().entries.is_empty();
        if empty && claim_rebuild(backend) {
            let kind2 = kind.clone();
            warn!(
                ?backend,
                "opencode catalog is empty — triggering background rebuild"
            );
            smol::spawn(async move {
                let catalog = backend::build_catalog_async(backend).await;
                apply_rebuilt_catalog(backend, kind2, catalog);
            })
            .detach();
        }

        // Eagerly populate the discovered registry so tier 2 of
        // `supports_thinking()`/`supports_vision()` is available immediately, before the async
        // `fetch_all_models` completes. The later `fetch_all_models` call overwrites with the same
        // data; this is harmless. When the catalog seeded empty, `apply_rebuilt_catalog` re-seeds
        // the registry once the background rebuild is complete.
        let catalog_guard = slot.get().unwrap().read().unwrap();
        model_registry()
            .write()
            .unwrap()
            .set_known_models(kind, backend.all_models(&catalog_guard));
        drop(catalog_guard);

        // Eagerly seed third-party catalog providers so `available_model_specs`
        // returns them immediately without waiting on `fetch_all_models`.
        for slug in crate::providers::catalog::registry::all_slugs() {
            if let Some(info) = crate::providers::catalog::registry::get(&slug) {
                let catalog_kind = ProviderKind::Catalog(Arc::from(slug.as_str()));
                model_registry()
                    .write()
                    .unwrap()
                    .set_known_models(catalog_kind, backend::catalog_models_to_info(&info));
            }
        }

        let chat_compat = OpenAiCompatProvider::new(
            match backend {
                Backend::Zen => ZEN_CHAT,
                Backend::Go => GO_CHAT,
            },
            timeouts,
        );
        Self {
            backend,
            client: http_client(timeouts),
            chat_compat,
            auth,
            system_prefix: None,
            stream_timeout: timeouts.stream,
        }
    }

    pub fn zen(timeouts: Timeouts) -> Result<Self, AgentError> {
        Ok(Self::new_impl(Backend::Zen, timeouts, None))
    }

    pub fn go(timeouts: Timeouts) -> Result<Self, AgentError> {
        Ok(Self::new_impl(Backend::Go, timeouts, None))
    }

    pub(crate) fn with_auth(
        backend: Backend,
        auth: Arc<Mutex<ResolvedAuth>>,
        timeouts: Timeouts,
    ) -> Self {
        Self::new_impl(backend, timeouts, Some(auth))
    }

    pub(crate) fn with_system_prefix(mut self, prefix: Option<String>) -> Self {
        self.system_prefix = prefix;
        self
    }
}

fn catalog_for(backend: Backend) -> &'static OnceLock<RwLock<Catalog>> {
    match backend {
        Backend::Zen => &ZEN_CATALOG,
        Backend::Go => &GO_CATALOG,
    }
}

fn rebuild_flag_for(backend: Backend) -> &'static AtomicBool {
    match backend {
        Backend::Zen => &ZEN_CATALOG_REBUILD,
        Backend::Go => &GO_CATALOG_REBUILD,
    }
}

fn claim_rebuild_flag(flag: &AtomicBool) -> bool {
    flag.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}

fn claim_rebuild(backend: Backend) -> bool {
    claim_rebuild_flag(rebuild_flag_for(backend))
}

fn collect_providers(cats: &[&RwLock<Catalog>]) -> Vec<ProviderData> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();

    for cat in cats {
        let guard = cat.read().unwrap();
        for (provider_id, provider) in &guard.providers {
            let slug = maki_slug_for(provider_id).to_string();
            if seen.insert(slug) {
                result.push(ProviderData::from_catalog_entry(provider_id, provider));
            }
        }
    }

    result.sort_by(|a, b| a.slug.cmp(&b.slug));
    result
}

pub fn catalog_providers() -> Vec<ProviderData> {
    let zen_cat = ZEN_CATALOG
        .get_or_init(|| RwLock::new(smol::block_on(backend::build_catalog_async(Backend::Zen))));
    let go_cat = GO_CATALOG
        .get_or_init(|| RwLock::new(smol::block_on(backend::build_catalog_async(Backend::Go))));
    collect_providers(&[zen_cat, go_cat])
}

pub fn catalog_providers_if_available() -> Option<Vec<ProviderData>> {
    let zen = ZEN_CATALOG.get();
    let go = GO_CATALOG.get();
    let cats: Vec<&RwLock<Catalog>> = zen.into_iter().chain(go).collect();
    if cats.is_empty() {
        return None;
    }

    Some(collect_providers(&cats))
}

pub fn catalog_provider(slug: &str) -> Option<ProviderData> {
    let lookup = |cat: &RwLock<Catalog>| {
        let guard = cat.read().unwrap();
        guard.providers.iter().find_map(|(id, p)| {
            if maki_slug_for(id) == slug {
                Some(ProviderData::from_catalog_entry(id, p))
            } else {
                None
            }
        })
    };

    if let Some(cat) = ZEN_CATALOG.get()
        && let Some(result) = lookup(cat)
    {
        return Some(result);
    }
    if let Some(cat) = GO_CATALOG.get()
        && let Some(result) = lookup(cat)
    {
        return Some(result);
    }
    None
}

pub fn catalog_provider_slugs() -> Vec<String> {
    registry::all_slugs()
}

#[allow(dead_code)]
pub(crate) fn catalog_provider_info(slug: &str) -> Option<registry::CatalogProviderInfo> {
    registry::get(slug)
}

/// Swap in a rebuilt catalog and re-seed the discovered registry so
/// thinking/vision/pricing resolve from the fresh entries. The flag is
/// cleared when the rebuild produced entries; on an empty catalog the flag
/// is left set so subsequent constructions skip re-spawning this process.
fn apply_rebuilt_catalog(backend: Backend, kind: ProviderKind, catalog: Catalog) {
    let entries = catalog.entries.len();
    {
        let mut guard = catalog_for(backend).get().unwrap().write().unwrap();
        *guard = catalog;
        model_registry()
            .write()
            .unwrap()
            .set_known_models(kind, backend.all_models(&guard));
    }
    if entries > 0 {
        rebuild_flag_for(backend).store(false, Ordering::Release);
    } else {
        warn!(?backend, "opencode catalog rebuild came back empty, leaving rebuild flag set");
    }
    warn!(?backend, entries, "opencode catalog rebuild complete");
}

impl Provider for Opencode {
    fn stream_message<'a>(
        &'a self,
        model: &'a Model,
        messages: &'a [Message],
        system: &'a str,
        tools: &'a Value,
        event_tx: &'a Sender<ProviderEvent>,
        opts: RequestOptions,
        _session_id: Option<&'a SessionRef>,
    ) -> BoxFuture<'a, Result<StreamResponse, AgentError>> {
        Box::pin(async move {
            let model_for_stream = model.clone();
            let backend = self.backend;

            let (meta, auth) = {
                let guard = catalog_for(backend).get().unwrap().read().unwrap();
                backend.lookup(&guard, &model_for_stream.id, self.auth.as_ref())?
            };

            let mut buf = String::new();
            let system = super::with_prefix(&self.system_prefix, system, &mut buf);

            let actual_id = match backend {
                Backend::Zen => backend.strip_prefix(&model_for_stream.id, &meta.provider_id),
                Backend::Go => model_for_stream.id,
            };

            let model = Model {
                id: actual_id,
                max_output_tokens: Some(meta.output),
                context_window: meta.context,
                ..model_for_stream
            };

            match meta.api_format {
                EndpointType::ChatCompletions => {
                    cat_request::chat_completions(
                        &self.chat_compat,
                        &model,
                        messages,
                        system,
                        tools,
                        event_tx,
                        &auth,
                        &opts,
                    )
                    .await
                }
                EndpointType::Messages => {
                    cat_request::anthropic_messages(
                        &self.client,
                        self.stream_timeout,
                        &model,
                        messages,
                        system,
                        tools,
                        event_tx,
                        &auth,
                        &opts,
                    )
                    .await
                }
            }
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<crate::model::ModelInfo>, AgentError>> {
        Box::pin(async move {
            let guard = catalog_for(self.backend).get().unwrap().read().unwrap();
            Ok(self.backend.all_models(&guard))
        })
    }

    fn reload_auth(&self) -> BoxFuture<'_, Result<(), AgentError>> {
        let backend = self.backend;
        Box::pin(async move {
            let new_catalog = backend::build_catalog_async(backend).await;
            let kind = match backend {
                Backend::Zen => ProviderKind::OpencodeZen,
                Backend::Go => ProviderKind::OpencodeGo,
            };
            apply_rebuilt_catalog(backend, kind, new_catalog);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::catalog::data::{EndpointType, Meta};
    use crate::providers::opencode::catalog::GO_PROVIDER_ID;

    const TEST_MODEL_ID: &str = "rebuild-test-model";

    fn go_catalog_with_model() -> Catalog {
        let mut catalog = Catalog::empty();
        catalog.entries.insert(
            (GO_PROVIDER_ID.to_string(), TEST_MODEL_ID.to_string()),
            Meta {
                provider_id: GO_PROVIDER_ID.to_string(),
                api_format: EndpointType::ChatCompletions,
                context: 128_000,
                output: 64_000,
                input_price: 0.0,
                output_price: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                vision: false,
                supports_thinking: true,
            },
        );
        catalog
    }

    #[test]
    fn claim_rebuild_flag_allows_one_rebuild_at_a_time() {
        let flag = AtomicBool::new(false);
        assert!(claim_rebuild_flag(&flag));
        assert!(!claim_rebuild_flag(&flag));
        flag.store(false, Ordering::Release);
        assert!(claim_rebuild_flag(&flag));
    }

    #[test]
    fn apply_rebuilt_catalog_swaps_catalog_and_reseeds_registry() {
        let slot = catalog_for(Backend::Go);
        slot.get_or_init(|| RwLock::new(Catalog::empty()));
        *slot.get().unwrap().write().unwrap() = Catalog::empty();
        rebuild_flag_for(Backend::Go).store(true, Ordering::Release);

        apply_rebuilt_catalog(
            Backend::Go,
            ProviderKind::OpencodeGo,
            go_catalog_with_model(),
        );

        assert_eq!(slot.get().unwrap().read().unwrap().entries.len(), 1);

        let registry = model_registry().read().unwrap();
        let info = registry
            .discovered(&ProviderKind::OpencodeGo, TEST_MODEL_ID)
            .expect("rebuilt catalog should re-seed the registry");
        assert_eq!(info.supports_thinking, Some(true));

        assert!(!rebuild_flag_for(Backend::Go).load(Ordering::Acquire));
    }

    #[test]
    fn apply_rebuilt_catalog_empty_catalog_preserves_rebuild_flag() {
        let slot = catalog_for(Backend::Go);
        slot.get_or_init(|| RwLock::new(Catalog::empty()));
        *slot.get().unwrap().write().unwrap() = go_catalog_with_model();
        rebuild_flag_for(Backend::Go).store(true, Ordering::Release);

        apply_rebuilt_catalog(
            Backend::Go,
            ProviderKind::OpencodeGo,
            Catalog::empty(),
        );

        assert!(slot.get().unwrap().read().unwrap().entries.is_empty());
        assert!(
            rebuild_flag_for(Backend::Go).load(Ordering::Acquire),
            "empty catalog rebuild should not clear the rebuild flag"
        );
    }
}
