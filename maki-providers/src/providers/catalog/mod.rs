//! Shared catalog infrastructure for the models.dev provider index.
//!
//! Provides the raw data model, HTTP loading, provider descriptor registry,
//! per-slug `CatalogProvider` implementation, request building, and the
//! `ProviderData` display facade. Backends (e.g. opencode) use these building
//! blocks to construct and serve their specific catalog instances.

pub mod catalog_provider;
pub(crate) mod data;
pub mod provider_data;
pub mod registry;
pub(crate) mod request;
