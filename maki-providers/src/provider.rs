use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use flume::Sender;
use serde_json::Value;
use tracing::{debug, warn};

use maki_storage::id::SessionRef;

use crate::model::{Model, ModelFamily, ModelInfo, models_for_provider};
use crate::providers::Timeouts;
use crate::providers::anthropic::Anthropic;
use crate::providers::anthropic::bedrock;
use crate::providers::copilot::Copilot;
use crate::providers::deepseek::DeepSeek;
use crate::providers::dynamic;
use crate::providers::google::Google;
use crate::providers::local::{LLAMACPP, LocalEndpoint, OLLAMA};
use crate::providers::mistral::Mistral;
use crate::providers::openai::OpenAi;
use crate::providers::opencode::{Backend, Opencode};
use crate::providers::openrouter::OpenRouter;
use crate::providers::synthetic::Synthetic;
use crate::providers::tensorx::TensorX;
use crate::providers::zai::Zai;
use crate::{AgentError, Message, ProviderEvent, ProviderUsage, RequestOptions, StreamResponse};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    Google,
    Copilot,
    Ollama,
    LlamaCpp,
    Mistral,
    Zai,
    DeepSeek,
    OpenRouter,
    Synthetic,
    TensorX,
    OpencodeZen,
    OpencodeGo,
    /// A dynamically-discovered third-party provider from models.dev.
    Catalog(Arc<str>),
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Catalog(slug) => f.write_str(slug),
            Self::Anthropic => f.write_str("anthropic"),
            Self::OpenAi => f.write_str("openai"),
            Self::Google => f.write_str("google"),
            Self::Copilot => f.write_str("copilot"),
            Self::Ollama => f.write_str("ollama"),
            Self::LlamaCpp => f.write_str("llama-cpp"),
            Self::Mistral => f.write_str("mistral"),
            Self::Zai => f.write_str("zai"),
            Self::DeepSeek => f.write_str("deepseek"),
            Self::OpenRouter => f.write_str("openrouter"),
            Self::Synthetic => f.write_str("synthetic"),
            Self::TensorX => f.write_str("tensorx"),
            Self::OpencodeZen => f.write_str("opencode-zen"),
            Self::OpencodeGo => f.write_str("opencode-go"),
        }
    }
}

pub const BUILTIN_KINDS: &[ProviderKind] = &[
    ProviderKind::Anthropic,
    ProviderKind::OpenAi,
    ProviderKind::Google,
    ProviderKind::Copilot,
    ProviderKind::Ollama,
    ProviderKind::LlamaCpp,
    ProviderKind::Mistral,
    ProviderKind::Zai,
    ProviderKind::DeepSeek,
    ProviderKind::OpenRouter,
    ProviderKind::Synthetic,
    ProviderKind::TensorX,
    ProviderKind::OpencodeZen,
    ProviderKind::OpencodeGo,
];

impl std::str::FromStr for ProviderKind {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "anthropic" => Ok(Self::Anthropic),
            "openai" => Ok(Self::OpenAi),
            "google" => Ok(Self::Google),
            "copilot" => Ok(Self::Copilot),
            "ollama" => Ok(Self::Ollama),
            "llama-cpp" => Ok(Self::LlamaCpp),
            "mistral" => Ok(Self::Mistral),
            "zai" => Ok(Self::Zai),
            "deepseek" => Ok(Self::DeepSeek),
            "openrouter" => Ok(Self::OpenRouter),
            "synthetic" => Ok(Self::Synthetic),
            "tensorx" => Ok(Self::TensorX),
            "opencode-zen" => Ok(Self::OpencodeZen),
            "opencode-go" => Ok(Self::OpencodeGo),
            "opencode" => {
                warn!(
                    slug = "opencode",
                    replacement = "opencode-zen",
                    "deprecated provider slug; update your config to use 'opencode-zen'"
                );
                Ok(Self::OpencodeZen)
            }
            _ => Err("unknown provider kind"),
        }
    }
}

impl ProviderKind {
    pub fn from_slug(slug: &str) -> Self {
        slug.parse().unwrap_or_else(|_| Self::Catalog(Arc::from(slug)))
    }

    pub fn slug(&self) -> Option<&str> {
        match self {
            Self::Catalog(slug) => Some(slug.as_ref()),
            _ => None,
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            Self::Anthropic => "Anthropic".to_string(),
            Self::OpenAi => "OpenAI".to_string(),
            Self::Google => "Google".to_string(),
            Self::Copilot => "Copilot".to_string(),
            Self::Ollama => "Ollama".to_string(),
            Self::LlamaCpp => "LlamaCpp".to_string(),
            Self::Mistral => "Mistral".to_string(),
            Self::Zai => "Z.AI".to_string(),
            Self::DeepSeek => "DeepSeek".to_string(),
            Self::OpenRouter => "OpenRouter".to_string(),
            Self::Synthetic => "Synthetic".to_string(),
            Self::TensorX => "TensorX".to_string(),
            Self::OpencodeZen => "OpencodeZen".to_string(),
            Self::OpencodeGo => "OpencodeGo".to_string(),
            Self::Catalog(slug) => {
                crate::providers::opencode::registry::get(slug)
                    .map(|info| info.display_name)
                    .unwrap_or_else(|| slug.to_string())
            }
        }
    }

    pub fn api_key_env(&self) -> String {
        match self {
            Self::Anthropic => "ANTHROPIC_API_KEY".into(),
            Self::OpenAi => "OPENAI_API_KEY".into(),
            Self::Google => "GEMINI_API_KEY".into(),
            Self::Copilot => "GH_COPILOT_TOKEN".into(),
            Self::Ollama => "OLLAMA_API_KEY".into(),
            Self::LlamaCpp => "LLAMA_CPP_API_KEY".into(),
            Self::Mistral => "MISTRAL_API_KEY".into(),
            Self::Zai => "ZHIPU_API_KEY".into(),
            Self::DeepSeek => "DEEPSEEK_API_KEY".into(),
            Self::OpenRouter => "OPENROUTER_API_KEY".into(),
            Self::Synthetic => "SYNTHETIC_API_KEY".into(),
            Self::TensorX => "TENSORX_API_KEY".into(),
            Self::OpencodeZen => "OPENCODE_API_KEY".into(),
            Self::OpencodeGo => "OPENCODE_API_KEY".into(),
            Self::Catalog(slug) => {
                crate::providers::opencode::registry::get(slug)
                    .map(|info| info.env_keys.join(","))
                    .unwrap_or_default()
            }
        }
    }

    pub fn base_url(&self) -> String {
        match self {
            Self::Anthropic => "https://api.anthropic.com/v1/messages".into(),
            Self::OpenAi => "https://api.openai.com/v1".into(),
            Self::Google => "https://generativelanguage.googleapis.com/v1beta".into(),
            Self::Copilot => "https://api.githubcopilot.com (or GraphQL-discovered Copilot API endpoint)".into(),
            Self::Ollama => "http://localhost:11434/v1".into(),
            Self::LlamaCpp => "http://localhost:8080/v1".into(),
            Self::Mistral => "https://api.mistral.ai/v1".into(),
            Self::Zai => "https://api.z.ai/api/paas/v4".into(),
            Self::DeepSeek => "https://api.deepseek.com".into(),
            Self::OpenRouter => "https://openrouter.ai/api/v1".into(),
            Self::Synthetic => "https://api.synthetic.new/openai/v1".into(),
            Self::TensorX => "https://api.tensorx.ai/v1".into(),
            Self::OpencodeZen => "https://opencode.ai/zen/v1".into(),
            Self::OpencodeGo => "https://opencode.ai/zen/go/v1".into(),
            Self::Catalog(slug) => {
                crate::providers::opencode::registry::get(slug)
                    .and_then(|info| info.base_url)
                    .unwrap_or_default()
            }
        }
    }

    pub fn supports_thinking(&self) -> bool {
        match self {
            Self::Catalog(_) => false,
            _ => matches!(
                self,
                Self::Anthropic
                    | Self::Google
                    | Self::Mistral
                    | Self::DeepSeek
                    | Self::Synthetic
                    | Self::OpenAi
                    | Self::OpenRouter
                    | Self::LlamaCpp
                    | Self::TensorX
                    | Self::OpencodeZen
                    | Self::OpencodeGo
            ),
        }
    }

    pub fn features(&self) -> Option<String> {
        match self {
            Self::Anthropic => Some("Prompt caching, thinking mode (adaptive/budgeted), advanced tool use".to_string()),
            Self::OpenAi => Some("ChatGPT, GPT-4, DALL-E, Whisper, and more".to_string()),
            Self::Google => Some("Native Gemini API with thinking support".to_string()),
            Self::Copilot => Some("Native Copilot Chat HTTP API with model endpoint discovery".to_string()),
            Self::Ollama => Some("Local or remote inference via OLLAMA_HOST, cloud fallback via OLLAMA_API_KEY".to_string()),
            Self::LlamaCpp => Some("Local or remote inference via LLAMA_CPP_HOST".to_string()),
            Self::Mistral => Some("Le Chat and Le Chat Enterprise".to_string()),
            Self::Zai => Some("Zhipu AI GLM models".to_string()),
            Self::DeepSeek => Some("Thinking mode toggle (on/off), open-weight models".to_string()),
            Self::Synthetic => Some("Reasoning effort support (low/medium/high), open-weight models".to_string()),
            Self::TensorX => Some("Open-weight models, zero data retention, prompt caching".to_string()),
            Self::OpenRouter => Some("300+ models from all providers, prompt caching, provider routing".to_string()),
            Self::OpencodeZen => Some("Dynamically discovered models via [models.dev](https://models.dev/) + all the models provided by Opencode Zen API".to_string()),
            Self::OpencodeGo => Some("Dynamically discovered models via [models.dev](https://models.dev/)".to_string()),
            Self::Catalog(slug) => {
                Some(crate::providers::opencode::registry::get(slug)
                    .map(|info| info.features)
                    .unwrap_or_else(|| "Third-party provider".to_string()))
            }
        }
    }

    pub fn family(&self) -> ModelFamily {
        match self {
            Self::Anthropic => ModelFamily::Claude,
            Self::OpenAi => ModelFamily::Gpt,
            Self::Google => ModelFamily::Gemini,
            Self::Copilot => ModelFamily::Generic,
            Self::Ollama => ModelFamily::Generic,
            Self::LlamaCpp => ModelFamily::Generic,
            Self::Mistral => ModelFamily::Generic,
            Self::Zai => ModelFamily::Glm,
            Self::DeepSeek => ModelFamily::Generic,
            Self::OpenRouter => ModelFamily::Generic,
            Self::Synthetic => ModelFamily::Synthetic,
            Self::TensorX => ModelFamily::Generic,
            Self::OpencodeZen => ModelFamily::Generic,
            Self::OpencodeGo => ModelFamily::Generic,
            Self::Catalog(_) => ModelFamily::Generic,
        }
    }

    pub fn accepts_arbitrary_models(&self) -> bool {
        match self {
            Self::Catalog(_) => true,
            _ => matches!(
                self,
                Self::Ollama
                    | Self::LlamaCpp
                    | Self::Google
                    | Self::Copilot
                    | Self::OpenRouter
                    | Self::TensorX
                    | Self::Mistral
                    | Self::OpencodeZen
                    | Self::OpencodeGo
            ),
        }
    }

    /// `None` when we honestly don't know the output window: llama.cpp
    /// serves whatever model the user loaded, and TensorX rejects explicit
    /// max_tokens (see tensorx.rs). Unknown means "don't limit", never
    /// "assume small"; a `0` sentinel here once silently capped llama.cpp
    /// thinking budgets at the floor.
    pub fn fallback_max_output(&self) -> Option<u32> {
        match self {
            Self::Anthropic => Some(128_000),
            Self::OpenAi => Some(100_000),
            Self::Google => Some(65_536),
            Self::Copilot => Some(100_000),
            Self::Ollama => Some(16_384),
            Self::LlamaCpp => None,
            Self::Mistral => Some(32_000),
            Self::Zai => Some(16_000),
            Self::DeepSeek => Some(384_000),
            Self::OpenRouter => Some(128_000),
            Self::Synthetic => Some(32_000),
            Self::TensorX => None,
            Self::OpencodeZen => Some(128_000),
            Self::OpencodeGo => Some(128_000),
            Self::Catalog(_) => None,
        }
    }

    pub fn fallback_context_window(&self) -> u32 {
        match self {
            Self::Anthropic => 200_000,
            Self::OpenAi => 200_000,
            Self::Google => 1_000_000,
            Self::Copilot => 200_000,
            Self::Ollama => 128_000,
            Self::LlamaCpp => 128_000,
            Self::Mistral => 128_000,
            Self::Zai => 128_000,
            Self::DeepSeek => 1_000_000,
            Self::OpenRouter => 200_000,
            Self::Synthetic => 128_000,
            Self::TensorX => 200_000,
            Self::OpencodeZen => 256_000,
            Self::OpencodeGo => 256_000,
            Self::Catalog(_) => 128_000,
        }
    }

    pub fn create(&self, timeouts: Timeouts) -> Result<Box<dyn Provider>, AgentError> {
        match self {
            Self::Anthropic => {
                if bedrock::is_enabled() {
                    Ok(Box::new(bedrock::Bedrock::new(timeouts)?))
                } else {
                    Ok(Box::new(Anthropic::new(timeouts)?))
                }
            }
            Self::OpenAi => Ok(Box::new(OpenAi::new(timeouts)?)),
            Self::Google => Ok(Box::new(Google::new(timeouts)?)),
            Self::Copilot => Ok(Box::new(Copilot::new(timeouts)?)),
            Self::Ollama => Ok(Box::new(LocalEndpoint::new(&OLLAMA, timeouts)?)),
            Self::LlamaCpp => Ok(Box::new(LocalEndpoint::new(&LLAMACPP, timeouts)?)),
            Self::Mistral => Ok(Box::new(Mistral::new(timeouts)?)),
            Self::Zai => Ok(Box::new(Zai::new(timeouts)?)),
            Self::DeepSeek => Ok(Box::new(DeepSeek::new(timeouts)?)),
            Self::OpenRouter => Ok(Box::new(OpenRouter::new(timeouts)?)),
            Self::Synthetic => Ok(Box::new(Synthetic::new(timeouts)?)),
            Self::TensorX => Ok(Box::new(TensorX::new(timeouts)?)),
            Self::OpencodeZen => Ok(Box::new(Opencode::zen(timeouts)?)),
            Self::OpencodeGo => Ok(Box::new(Opencode::go(timeouts)?)),
            Self::Catalog(slug) => {
                let info = crate::providers::opencode::registry::get(slug)
                    .ok_or_else(|| AgentError::Config {
                        message: format!("unknown catalog provider '{slug}'"),
                    })?;
                let auth = crate::providers::opencode::registry::resolve_auth_for_catalog(slug, &info.env_keys)?;
                Ok(Box::new(crate::providers::opencode::cat_provider::CatalogProvider::new(
                    Arc::clone(slug), info, auth, timeouts,
                )))
            }
        }
    }

    /// Canonicalize a model id before registry lookups and spec rendering
    /// (legacy opencode id shapes); most providers use the id unchanged.
    pub fn normalize_model_id(&self, model_id: &str) -> String {
        match self {
            Self::OpencodeZen => Backend::Zen.normalize_model_id(model_id),
            Self::OpencodeGo => Backend::Go.normalize_model_id(model_id),
            Self::Catalog(_) => model_id.to_string(),
            _ => model_id.to_string(),
        }
    }

    /// Id form shown in pickers: opencode drops the redundant default
    /// sub-provider prefix; other providers use the id unchanged.
    pub fn display_model_id<'a>(&self, model_id: &'a str) -> &'a str {
        match self {
            Self::OpencodeZen => Backend::Zen.display_model_id(model_id),
            Self::Catalog(_) => model_id,
            _ => model_id,
        }
    }

    pub fn is_available(&self) -> bool {
        match self {
            // Opencode construction cannot fail and has global side effects
            // (blocking catalog build, registry seeding); an availability
            // probe must not trigger them.
            Self::OpencodeZen | Self::OpencodeGo => true,
            Self::Catalog(_) => self.create(Timeouts::default()).is_ok(),
            _ => self.create(Timeouts::default()).is_ok(),
        }
    }
}

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub trait Provider: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn stream_message<'a>(
        &'a self,
        model: &'a Model,
        messages: &'a [Message],
        system: &'a str,
        tools: &'a Value,
        event_tx: &'a Sender<ProviderEvent>,
        opts: RequestOptions,
        session_id: Option<&'a SessionRef>,
    ) -> BoxFuture<'a, Result<StreamResponse, AgentError>>;

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>>;

    /// Fetch provider-side usage quota (remaining percentage / reset times).
    /// `Ok(None)` means the provider does not expose a programmatic usage endpoint.
    fn fetch_usage(&self) -> BoxFuture<'_, Result<Option<ProviderUsage>, AgentError>> {
        Box::pin(async { Ok(None) })
    }

    fn refresh_auth(&self) -> BoxFuture<'_, Result<(), AgentError>> {
        Box::pin(async { Ok(()) })
    }

    fn reload_auth(&self) -> BoxFuture<'_, Result<(), AgentError>> {
        Box::pin(async { Ok(()) })
    }

    fn rotate_key(&self) -> BoxFuture<'_, Result<bool, AgentError>> {
        Box::pin(async { Ok(false) })
    }

    fn adjust_model(&self, _model: &mut Model) {}
}

fn provider_for_slug(slug: &str, timeouts: Timeouts) -> Result<Box<dyn Provider>, AgentError> {
    if dynamic::display_name(slug).is_some() {
        dynamic::create(slug, timeouts)
    } else {
        crate::providers::custom::create(slug, timeouts)
    }
}

pub fn from_model(model: &mut Model, timeouts: Timeouts) -> Result<Box<dyn Provider>, AgentError> {
    if let Some(slug) = &model.dynamic_slug {
        debug!(slug, model = %model.id, "slug provider created");
        return provider_for_slug(slug, timeouts);
    }
    let provider = model.provider.create(timeouts)?;
    provider.adjust_model(model);
    debug!(provider = %model.provider, model = %model.id, "provider created");
    Ok(provider)
}

pub fn from_model_fallback(model: &mut Model, timeouts: Timeouts) -> Box<dyn Provider> {
    match from_model(model, timeouts) {
        Ok(provider) => provider,
        Err(e) => {
            warn!(error = %e, "provider creation failed, using unconfigured provider");
            Box::new(UnconfiguredProvider)
        }
    }
}

struct UnconfiguredProvider;

const NOT_CONFIGURED: &str = "no provider configured — run /login or `maki auth login`";

impl Provider for UnconfiguredProvider {
    fn stream_message<'a>(
        &'a self,
        _model: &'a Model,
        _messages: &'a [Message],
        _system: &'a str,
        _tools: &'a Value,
        _event_tx: &'a Sender<ProviderEvent>,
        _opts: RequestOptions,
        _session_id: Option<&'a SessionRef>,
    ) -> BoxFuture<'a, Result<StreamResponse, AgentError>> {
        Box::pin(async {
            Err(AgentError::Config {
                message: NOT_CONFIGURED.to_string(),
            })
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
        Box::pin(async {
            Err(AgentError::Config {
                message: NOT_CONFIGURED.to_string(),
            })
        })
    }
}

pub async fn from_model_async(
    model: &mut Model,
    timeouts: Timeouts,
) -> Result<Box<dyn Provider>, AgentError> {
    let slug = model.dynamic_slug.clone();
    let kind = model.provider.clone();
    let id = model.id.clone();
    let kind2 = kind.clone();
    let provider = smol::unblock(move || {
        if let Some(slug) = &slug {
            provider_for_slug(slug, timeouts)
        } else {
            kind2.create(timeouts)
        }
    })
    .await?;
    if model.dynamic_slug.is_none() {
        provider.adjust_model(model);
    }
    debug!(provider = %kind, model = %id, "provider created");
    Ok(provider)
}

pub struct ModelBatch {
    pub models: Vec<String>,
    pub warnings: Vec<String>,
}

/// Offline version of model discovery: returns specs from static tables
/// and configured dynamic providers. See [`fetch_all_models`] for live lookups.
pub fn available_model_specs() -> Vec<String> {
    let mut specs: Vec<String> = BUILTIN_KINDS
        .iter()
        .filter(|kind| kind.is_available())
        .flat_map(|kind| {
            models_for_provider(kind)
                .iter()
                .flat_map(|entry| entry.prefixes.iter())
                .map(move |p| format!("{kind}/{p}"))
        })
        .collect();
    for slug in dynamic::discovered_slugs() {
        specs.extend(dynamic::dynamic_model_specs_for(slug));
    }
    for spec in crate::providers::custom::declared_model_specs() {
        if !specs.contains(&spec) {
            specs.push(spec);
        }
    }
    // Catalog providers: use known_models seeded by Opencode::new_impl or fetch_all_models
    // instead of creating providers synchronously (which risks panicking via smol::block_on).
    let registry = crate::model_registry::model_registry().read().unwrap();
    for slug in crate::providers::opencode::registry::all_slugs() {
        let kind = ProviderKind::Catalog(Arc::from(slug));
        if let Some(models) = registry.discovered_models(&kind) {
            for m in models {
                specs.push(format!("{kind}/{}", m.id));
            }
        }
    }
    specs
}

pub async fn fetch_all_models(
    mut on_ready: impl FnMut(ModelBatch),
    on_done: Option<Box<dyn FnOnce() + Send>>,
) {
    let (tx, rx) = flume::unbounded();
    let timeouts = Timeouts::default();

    for kind in BUILTIN_KINDS.iter() {
        let kind = kind.clone();
        let kind2 = kind.clone();
        let Ok(provider) = smol::unblock(move || kind2.create(timeouts)).await else {
            warn!(provider = %kind, "failed to create provider, skipping");
            continue;
        };
        let tx = tx.clone();
        smol::spawn(async move {
            let batch = match provider.list_models().await {
                Ok(models) => {
                    if kind.accepts_arbitrary_models() {
                        crate::model_registry::model_registry()
                            .write()
                            .unwrap()
                            .set_known_models(kind.clone(), models.clone());
                    }
                    let mut specs: Vec<String> =
                        models.iter().map(|m| format!("{kind}/{}", m.id)).collect();
                    for entry in models_for_provider(&kind) {
                        for prefix in entry.prefixes {
                            let spec = format!("{kind}/{prefix}");
                            if !specs.contains(&spec) {
                                specs.push(spec);
                            }
                        }
                    }
                    ModelBatch {
                        models: specs,
                        warnings: Vec::new(),
                    }
                }
                Err(e) => {
                    warn!(provider = %kind, error = %e, "failed to list models, using static fallback");
                    let fallback: Vec<String> = models_for_provider(&kind)
                        .iter()
                        .flat_map(|entry| entry.prefixes.iter())
                        .map(|p| format!("{kind}/{p}"))
                        .collect();
                    ModelBatch {
                        models: fallback,
                        warnings: vec![format!(
                            "{}: {e} (using static fallback)",
                            kind.display_name()
                        )],
                    }
                }
            };
            let _ = tx.send_async(batch).await;
        })
        .detach();
    }

    // Spawn catalog provider fetches
    let config = maki_config::providers::ProvidersConfig::load();
    for slug in crate::providers::opencode::registry::all_slugs() {
        if dynamic::display_name(&slug).is_some()
            || config.get(&slug).is_some()
        {
            continue;
        }
        let slug = Arc::<str>::from(slug);
        let tx = tx.clone();
        smol::spawn(async move {
            match ProviderKind::Catalog(Arc::clone(&slug)).create(timeouts) {
                Ok(provider) => {
                    match provider.list_models().await {
                        Ok(models) => {
                            let specs: Vec<String> = models
                                .iter()
                                .map(|m| format!("{slug}/{}", m.id))
                                .collect();
                            crate::model_registry::model_registry()
                                .write()
                                .unwrap()
                                .set_known_models(ProviderKind::Catalog(Arc::clone(&slug)), models);
                            let _ = tx.send_async(ModelBatch { models: specs, warnings: vec![] }).await;
                        }
                        Err(e) => {
                            warn!(slug = %slug, error = %e, "catalog provider list_models failed");
                        }
                    }
                }
                Err(_) => { /* no auth → skip */ }
            }
        }).detach();
    }

    for slug in dynamic::discovered_slugs() {
        let tx = tx.clone();
        let slug = slug.to_string();
        smol::spawn(async move {
            let static_fallback = |reason: String| {
                warn!(
                    slug,
                    error = reason,
                    "dynamic model listing failed, using static fallback"
                );
                ModelBatch {
                    models: dynamic::dynamic_model_specs_for(&slug),
                    warnings: vec![format!("{slug}: {reason} (using static fallback)")],
                }
            };
            let batch = match dynamic::create(&slug, timeouts) {
                Ok(provider) => match provider.list_models().await {
                    Ok(models) => ModelBatch {
                        models: models.iter().map(|m| format!("{slug}/{}", m.id)).collect(),
                        warnings: Vec::new(),
                    },
                    Err(e) => static_fallback(e.to_string()),
                },
                Err(e) => static_fallback(e.to_string()),
            };
            let _ = tx.send_async(batch).await;
        })
        .detach();
    }

    let custom_timeouts = timeouts;
    let tx_custom = tx.clone();
    smol::spawn(async move {
        let declared = crate::providers::custom::declared_model_specs();
        if !declared.is_empty() {
            let _ = tx_custom
                .send_async(ModelBatch {
                    models: declared,
                    warnings: Vec::new(),
                })
                .await;
        }
        let custom_specs =
            smol::unblock(move || crate::providers::custom::discover_models(custom_timeouts)).await;
        if !custom_specs.is_empty() {
            let _ = tx_custom
                .send_async(ModelBatch {
                    models: custom_specs,
                    warnings: Vec::new(),
                })
                .await;
        }
    })
    .detach();

    drop(tx);

    while let Ok(batch) = rx.recv_async().await {
        on_ready(batch);
    }
    if let Some(done) = on_done {
        done();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    const UNKNOWN_PROVIDER_ERR: &str = "unknown provider kind";

    #[test_case("opencode", ProviderKind::OpencodeZen ; "from_str_opencode_deprecated")]
    #[test_case("opencode-zen", ProviderKind::OpencodeZen ; "from_str_opencode_zen")]
    #[test_case("opencode-go", ProviderKind::OpencodeGo ; "from_str_opencode_go")]
    fn from_str_ok(slug: &str, expected: ProviderKind) {
        assert_eq!(slug.parse::<ProviderKind>().unwrap(), expected);
    }

    #[test_case("nonexistent" ; "from_str_unknown")]
    fn from_str_err(slug: &str) {
        assert_eq!(
            slug.parse::<ProviderKind>().unwrap_err(),
            UNKNOWN_PROVIDER_ERR
        );
    }

    #[test]
    fn builtin_kinds_includes_all_compile_time_variants() {
        assert_eq!(BUILTIN_KINDS.len(), 14);
    }

    #[test]
    fn from_slug_parses_builtin_or_creates_catalog() {
        assert_eq!(ProviderKind::from_slug("anthropic"), ProviderKind::Anthropic);
        assert_eq!(ProviderKind::from_slug("opencode-zen"), ProviderKind::OpencodeZen);
        assert_eq!(ProviderKind::from_slug("nvidia"), ProviderKind::Catalog(Arc::from("nvidia")));
    }

    #[test]
    fn catalog_slug_accessor() {
        let kind = ProviderKind::Catalog(Arc::from("nvidia"));
        assert_eq!(kind.slug(), Some("nvidia"));
        assert_eq!(ProviderKind::Anthropic.slug(), None);
    }

    #[test]
    fn catalog_round_trips_through_display_and_from_slug() {
        let kind = ProviderKind::Catalog(Arc::from("nvidia"));
        assert_eq!(kind.to_string(), "nvidia");
        assert_eq!(ProviderKind::from_slug(&kind.to_string()), kind);
    }
}
