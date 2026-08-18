//! The raw-completion prefill+decode loop, wiring the `selection` sampler,
//! the `tokenizer` crate's streaming detokenizer and stop matcher, and a
//! pluggable [`LogitProducer`] into one token generation loop. Ported from
//! `Runtime/Generation/RawCompletion.swift` and `LogitProducer.swift`.
#![forbid(unsafe_code)]

mod config;
mod context_policy;
mod error;
mod expert_cache_policy;
#[cfg(target_os = "macos")]
mod families;
#[cfg(target_os = "macos")]
mod ffn_hist;
mod pacing;
mod power;
mod producer;
mod raw_completion;
#[cfg(target_os = "macos")]
mod real_forward;
#[cfg(target_os = "macos")]
mod real_forward_dispatch;
#[cfg(target_os = "macos")]
mod real_forward_init;
#[cfg(target_os = "macos")]
mod real_forward_layout;
#[cfg(target_os = "macos")]
mod real_forward_types;
#[cfg(target_os = "macos")]
mod real_forward_utils;
#[cfg(target_os = "macos")]
mod router_hist;

pub use config::GenerationConfig;
pub use context_policy::{
    committed_bytes, kv_bytes_for_context, largest_context_within, resolve_max_context,
    ContextPlan, ContextTooLarge, MaxContext, CONTEXT_BUDGET_FRACTION, CONTEXT_GRANULARITY,
    CONTEXT_RESERVE_BYTES, MAX_SUPPORTED_CONTEXT,
};
pub use error::RuntimeError;
pub use expert_cache_policy::{ExpertCacheSlots, HEADROOM_FRACTION, HEADROOM_RESERVE_BYTES};
pub use power::{
    low_power_mode_enabled, physical_memory, rate_control_for, resolve_profile, stepped_cap,
    thermal_level, PowerProfile, RateControl, ThermalLevel, CRITICAL_TOK_PER_SEC,
    READING_SPEED_TOK_PER_SEC, SERIOUS_TOK_PER_SEC,
};
pub use producer::{ChunkedPrefillRunner, LogitProducer, ScriptedLogitProducer};
pub use raw_completion::{
    run_raw_completion, run_raw_completion_chunked, RawDecodeProgress, RawDecodeResult, StopReason,
};
#[cfg(target_os = "macos")]
pub use real_forward::{
    dispatch_profile_report, PhaseCounters, RealForwardError, RealForwardRunner, RollbackPoint,
};

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
