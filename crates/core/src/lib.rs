//! Core leaf crate: shared primitives and the allowed runtime-knob sets.
//!
//! Downstream crates should depend on this crate under an alias, for example
//! `foundation = { package = "turbospark-core", path = "../core" }`, so that references
//! to it are not confused with the standard library `core` crate in the
//! extern prelude.

#![forbid(unsafe_code)]

/// Prefill chunk sizing and automatic chunk size resolution.
pub mod chunk_sizing;
/// Prefill chunking primitives: span planning and commit-state tracking.
pub mod prefill;
/// Core scalar primitives, type aliases, and logit view definitions.
pub mod primitives;
/// Allowed numeric sets and documented defaults for runtime knobs.
pub mod runtime_config;
/// The directional-steering edit's mode, shared by the CPU reference and the
/// Metal dispatch.
pub mod steering;

pub use chunk_sizing::{resolve_automatic_chunk_size, InputLength};
pub use prefill::{
    prefill_chunk_spans, PrefillChunkCommitState, PrefillChunkSpan, PrefillError, MAX_CHUNK_TOKENS,
};
pub use primitives::{LogitValue, LogitsView, TokenId};
pub use runtime_config::{
    ALLOWED_CACHE_SLOTS, ALLOWED_CHUNK_SIZES, ALLOWED_SPECULATION_BLOCKS, DEFAULT_CACHE_SLOTS,
    DEFAULT_CHUNK_SIZE, DEFAULT_MAX_CONTEXT,
};
pub use steering::{SteeringMode, STEERING_MODE_NAMES};
