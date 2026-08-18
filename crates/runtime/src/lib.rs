//! The raw-completion prefill+decode loop, wiring the `selection` sampler,
//! the `tokenizer` crate's streaming detokenizer and stop matcher, and a
//! pluggable [`LogitProducer`] into one token generation loop. Ported from
//! `Runtime/Generation/RawCompletion.swift` and `LogitProducer.swift`.
#![forbid(unsafe_code)]

mod config;
mod error;
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
// The two sizing policies LIVE IN `model_io` and are re-exported here.
//
// They moved there when `crates/catalog` needed the same arithmetic to answer
// "would this install fit before I spend twenty minutes streaming it". Both
// are pure functions of an `ArchConfig` and a machine size, neither touches
// `gpu`, and `model_io` is the lowest crate that already owns `ArchConfig` --
// so the alternative was a second copy of the KV formula in a crate that
// could not see this one, which is the drift this repo has paid for before.
//
// Re-exported rather than relocated in the callers' source because every
// consumer (`crates/cli`, `crates/server`, `crates/ffi`, `crates/bench`)
// reaches them as `runtime::MaxContext` and `runtime::ExpertCacheSlots`, and
// this is where a reader of the decode engine expects to find them named.
#[cfg(target_os = "macos")]
pub use model_io::{
    committed_bytes, kv_bytes_for_context, largest_context_within, resolve_max_context,
    ContextPlan, ContextTooLarge, ExpertCacheSlots, MaxContext, CONTEXT_BUDGET_FRACTION,
    CONTEXT_GRANULARITY, CONTEXT_RESERVE_BYTES, HEADROOM_FRACTION, HEADROOM_RESERVE_BYTES,
    MAX_SUPPORTED_CONTEXT,
};

pub use error::RuntimeError;
pub use power::{
    low_power_mode_enabled, physical_memory, rate_control_for, resolve_profile, stepped_cap,
    thermal_level, PowerProfile, RateControl, ThermalLevel, CRITICAL_TOK_PER_SEC,
    READING_SPEED_TOK_PER_SEC, SERIOUS_TOK_PER_SEC,
};
pub use producer::{ChunkedPrefillRunner, LogitProducer, ScriptedLogitProducer};
pub use raw_completion::{
    run_raw_completion, run_raw_completion_cancellable, run_raw_completion_chunked,
    run_raw_completion_chunked_cancellable, CancelFlag, RawDecodeProgress, RawDecodeResult,
    StopReason,
};
#[cfg(target_os = "macos")]
pub use real_forward::{
    dispatch_profile_report, PhaseCounters, RealForwardError, RealForwardRunner, RollbackPoint,
};

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
