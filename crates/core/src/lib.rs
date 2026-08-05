//! Core leaf crate: shared primitives and the public runtime configuration.
//!
//! Downstream crates should depend on this crate under an alias, for example
//! `foundation = { package = "mrefrust-core", path = "../core" }`, so that references
//! to it are not confused with the standard library `core` crate in the
//! extern prelude.

pub mod chunk_sizing;
pub mod error;
pub mod prefill;
pub mod primitives;
pub mod runtime_config;

pub use chunk_sizing::{resolve_automatic_chunk_size, InputLength};
pub use error::Error;
pub use prefill::{
    prefill_chunk_spans, PrefillChunkCommitState, PrefillChunkSpan, PrefillError, PrefillMode,
    PrefillRuntimeConfig,
};
pub use primitives::{LogitValue, LogitsView, TokenId};
pub use runtime_config::{
    AttentionStrategy, CacheReplacement, HeadProjection, RuntimeConfig, RuntimeConfigBuilder,
    ALLOWED_CACHE_SLOTS, ALLOWED_CHUNK_SIZES, DEFAULT_CACHE_SLOTS, DEFAULT_CHUNK_SIZE,
};
