//! OpenCode-specific chat-compat configs.
//!
//! The actual request building functions live in [`crate::providers::catalog::request`].

use crate::providers::openai_compat::OpenAiCompatConfig;

pub(super) const ZEN_CHAT: &OpenAiCompatConfig = &OpenAiCompatConfig {
    api_key_env: "",
    base_url: "",
    max_tokens_field: "max_tokens",
    include_stream_usage: true,
    provider_name: "Opencode Zen",
};

pub(super) const GO_CHAT: &OpenAiCompatConfig = &OpenAiCompatConfig {
    api_key_env: "",
    base_url: "",
    max_tokens_field: "max_tokens",
    include_stream_usage: true,
    provider_name: "Opencode Go",
};
