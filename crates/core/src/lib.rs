//! Core leaf crate: shared primitives and the public runtime configuration.
//!
//! Downstream crates should depend on this crate under an alias, for example
//! `foundation = { package = "turbospark-core", path = "../core" }`, so that references
//! to it are not confused with the standard library `core` crate in the
//! extern prelude.

/// Prefill chunk sizing and automatic chunk size resolution.
pub mod chunk_sizing;
/// Error types for foundation operations.
pub mod error;
/// Prefill chunking strategies and runtime prefill configuration.
pub mod prefill;
/// Core scalar primitives, type aliases, and logit view definitions.
pub mod primitives;
/// Runtime engine configuration and options builder.
pub mod runtime_config;
/// The directional-steering edit's mode, shared by the CPU reference and the
/// Metal dispatch.
pub mod steering;

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
pub use steering::{SteeringMode, STEERING_MODE_NAMES};
