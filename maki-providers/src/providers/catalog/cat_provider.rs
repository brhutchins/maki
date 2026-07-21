use std::sync::{Arc, Mutex};
use std::time::Duration;

use flume::Sender;
use isahc::HttpClient;
use serde_json::Value;

use super::data::EndpointType;
use super::registry::{CatalogProviderInfo, resolve_auth_for_catalog};
use super::request;
use crate::model::{Model, ModelInfo, ModelPricing};
use crate::provider::{BoxFuture, Provider};
use crate::providers::openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider};
use crate::providers::{ResolvedAuth, Timeouts, http_client};
use crate::{AgentError, Message, ProviderEvent, RequestOptions, StreamResponse};

/// Static config used for all catalog providers. `api_key_env` and `base_url`
/// are empty because auth is resolved at construction time from env vars or
/// saved credentials and passed via `CatalogProvider::new` -> `auth` field.
static CAT_CHAT_COMPAT: &OpenAiCompatConfig = &OpenAiCompatConfig {
    api_key_env: "",
    base_url: "",
    max_tokens_field: "max_tokens",
    include_stream_usage: true,
    provider_name: "catalog",
};

pub(crate) struct CatalogProvider {
    slug: Arc<str>,
    info: CatalogProviderInfo,
    auth: Arc<Mutex<ResolvedAuth>>,
    chat_compat: OpenAiCompatProvider,
    client: HttpClient,
    stream_timeout: Duration,
}

impl CatalogProvider {
    pub(crate) fn new(
        slug: Arc<str>,
        info: CatalogProviderInfo,
        auth: ResolvedAuth,
        timeouts: Timeouts,
    ) -> Self {
        let auth = ResolvedAuth {
            base_url: auth.base_url.clone().or(info.base_url.clone()),
            ..auth
        };
        Self {
            slug,
            info,
            auth: Arc::new(Mutex::new(auth)),
            chat_compat: OpenAiCompatProvider::new(CAT_CHAT_COMPAT, timeouts),
            client: http_client(timeouts),
            stream_timeout: timeouts.stream,
        }
    }
}

impl Provider for CatalogProvider {
    fn stream_message<'a>(
        &'a self,
        model: &'a Model,
        messages: &'a [Message],
        system: &'a str,
        tools: &'a Value,
        event_tx: &'a Sender<ProviderEvent>,
        opts: RequestOptions,
        _session_id: Option<&'a maki_storage::id::SessionRef>,
    ) -> BoxFuture<'a, Result<StreamResponse, AgentError>> {
        Box::pin(async move {
            let model_meta = self.info.models.iter().find(|m| m.id == model.id);
            let meta = match model_meta {
                Some(m) => &m.meta,
                None => {
                    return Err(AgentError::Config {
                        message: format!("model '{}' not found in catalog provider '{}'", model.id, self.slug),
                    })
                }
            };
            let auth = self.auth.lock().unwrap().clone();
            let actual_id = model.id.clone();
            let model_for_stream = Model {
                id: actual_id,
                max_output_tokens: Some(meta.output),
                context_window: meta.context,
                ..model.clone()
            };

            match meta.api_format {
                EndpointType::ChatCompletions => {
                    request::chat_completions(
                        &self.chat_compat,
                        &model_for_stream,
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
                    request::anthropic_messages(
                        &self.client,
                        self.stream_timeout,
                        &model_for_stream,
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

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
        let models: Vec<ModelInfo> = self
            .info
            .models
            .iter()
            .map(|m| ModelInfo {
                id: m.id.clone(),
                context_window: Some(m.meta.context),
                max_output_tokens: Some(m.meta.output),
                pricing: Some(ModelPricing {
                    input: m.meta.input_price,
                    output: m.meta.output_price,
                    cache_read: m.meta.cache_read,
                    cache_write: m.meta.cache_write,
                    fast: None,
                }),
                supports_thinking: Some(m.meta.supports_thinking),
                supports_vision: Some(m.meta.vision),
                provider_info: None,
            })
            .collect();
        Box::pin(async { Ok(models) })
    }

    fn reload_auth(&self) -> BoxFuture<'_, Result<(), AgentError>> {
        let slug = self.slug.clone();
        let env_keys = self.info.env_keys.clone();
        let info_base_url = self.info.base_url.clone();
        let auth = self.auth.clone();
        Box::pin(async move {
            let new_auth = resolve_auth_for_catalog(&slug, &env_keys)?;
            *auth.lock().unwrap() = ResolvedAuth {
                base_url: new_auth.base_url.or(info_base_url),
                ..new_auth
            };
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::catalog::registry::CatalogProviderInfo;

    fn info_with_base_url(base_url: Option<&str>) -> CatalogProviderInfo {
        CatalogProviderInfo {
            display_name: "Neuralwatt".into(),
            env_keys: vec![],
            base_url: base_url.map(String::from),
            features: "Neuralwatt models".into(),
            models: Vec::new(),
        }
    }

    #[test]
    fn new_propagates_info_base_url_to_auth() {
        let info = info_with_base_url(Some("https://api.neuralwatt.com/v1"));
        let auth = ResolvedAuth::bearer("token");
        assert!(auth.base_url.is_none());

        let provider = CatalogProvider::new(Arc::from("neuralwatt"), info, auth, Timeouts::default());

        assert_eq!(
            provider.auth.lock().unwrap().base_url.as_deref(),
            Some("https://api.neuralwatt.com/v1")
        );
    }

    #[test]
    fn new_preserves_existing_auth_base_url_over_info_base_url() {
        let info = info_with_base_url(Some("https://api.neuralwatt.com/v1"));
        let mut auth = ResolvedAuth::bearer("token");
        auth.base_url = Some("https://override.example/v1".into());

        let provider = CatalogProvider::new(Arc::from("neuralwatt"), info, auth, Timeouts::default());

        assert_eq!(
            provider.auth.lock().unwrap().base_url.as_deref(),
            Some("https://override.example/v1")
        );
    }
}
