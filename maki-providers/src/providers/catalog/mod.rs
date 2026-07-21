//! Shared catalog infrastructure for the models.dev provider index.
//!
//! Provides the raw data model, HTTP loading, provider descriptor registry,
//! per-slug `CatalogProvider` implementation, request building, and the
//! `ProviderData` display facade. Backends (e.g. opencode) use these building
//! blocks to construct and serve their specific catalog instances.

pub(crate) mod data;
pub(crate) mod request;
pub mod registry;
pub mod cat_provider;
pub mod provider_data;
