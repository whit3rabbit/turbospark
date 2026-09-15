//! The raw-completion prefill+decode loop, wiring the `selection` sampler,
//! the `tokenizer` crate's streaming detokenizer and stop matcher, and a
//! pluggable [`LogitProducer`] into one token generation loop. Ported from
//! `Runtime/Generation/RawCompletion.swift` and `LogitProducer.swift`.
#![forbid(unsafe_code)]

mod config;
#[cfg(target_os = "macos")]
pub mod encoder;
mod error;
mod expert_layout_validation;
#[cfg(target_os = "macos")]
mod families;
#[cfg(target_os = "macos")]
mod ffn_hist;
#[cfg(target_os = "macos")]
mod kv_prefix;
#[cfg(target_os = "macos")]
mod kv_write;
#[cfg(target_os = "macos")]
mod moe_prefill_pipeline;
mod pacing;
mod power;
mod producer;
mod raw_completion;
mod raw_completion_chunked;
#[cfg(target_os = "macos")]
mod real_forward;
#[cfg(target_os = "macos")]
mod real_forward_api;
#[cfg(target_os = "macos")]
mod real_forward_dispatch;
#[cfg(target_os = "macos")]
mod real_forward_dispatch_moe;
#[cfg(target_os = "macos")]
mod real_forward_init;
#[cfg(target_os = "macos")]
mod real_forward_layout;
#[cfg(target_os = "macos")]
mod real_forward_open;
#[cfg(target_os = "macos")]
mod real_forward_rollback;
#[cfg(target_os = "macos")]
mod real_forward_traits;
#[cfg(target_os = "macos")]
mod real_forward_types;
#[cfg(target_os = "macos")]
mod real_forward_utils;
#[cfg(target_os = "macos")]
mod real_forward_vision_api;
#[cfg(target_os = "macos")]
mod resid_capture;
#[cfg(target_os = "macos")]
mod router_hist;
#[cfg(target_os = "macos")]
mod session_pool;
#[cfg(target_os = "macos")]
mod speculation_policy;
mod speculative;
#[cfg(target_os = "macos")]
mod steering;
mod token_sink;
mod turn_stream;
#[cfg(target_os = "macos")]
pub mod vision;

pub use config::GenerationConfig;
#[cfg(target_os = "macos")]
pub use encoder::{cosine_similarity, EncoderRunner};
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
    committed_breakdown, committed_bytes, gdn_state_bytes, kv_bytes_for_context,
    kv_bytes_for_context_with, largest_context_within, largest_context_within_with,
    resolve_max_context, resolve_max_context_with, session_pool_bytes, session_pool_bytes_with,
    CommittedBytes, ContextCap, ContextFloorUnmet, ContextOverCap, ContextPlan, ContextRefused,
    ContextTooLarge, ExpertCacheSlots, ExpertResidency, GuardBudget, KvQuant, LoadGuard,
    LoadPolicy, MaxContext, ResolvedExpertResidency, CONTEXT_BUDGET_FRACTION, CONTEXT_GRANULARITY,
    CONTEXT_RESERVE_BYTES, HEADROOM_FRACTION, HEADROOM_RESERVE_BYTES, MAX_SUPPORTED_CONTEXT,
};

pub use error::RuntimeError;
#[cfg(target_os = "macos")]
pub use families::qwen::{
    install_has_dflash, install_has_mtp_head, DflashDraftPolicy, DraftPolicies, MtpDraftPolicy,
    DFLASH_BLOCK, DFLASH_SERVING_BLOCK, MOE_SPECULATION_BLOCKER_MARKER,
    VISION_SPECULATION_BLOCKER_MARKER,
};
pub use power::{
    low_power_mode_enabled, memory_cap, memory_pressure, physical_memory, rate_control_for,
    recommended_max_working_set, resolve_profile, stepped_cap, thermal_cap, thermal_level,
    MemoryPressure, PowerProfile, RateControl, ThermalLevel, CRITICAL_TOK_PER_SEC,
    READING_SPEED_TOK_PER_SEC, SERIOUS_TOK_PER_SEC,
};
pub use producer::{
    ChunkedPrefillRunner, LogitProducer, ScriptedLogitProducer, SpeculativeProducer,
};
pub use raw_completion::{
    run_raw_completion, run_raw_completion_cancellable, run_raw_completion_chunked,
    run_raw_completion_chunked_cancellable, CancelFlag, RawDecodeProgress, RawDecodeResult,
    StopReason,
};
#[cfg(target_os = "macos")]
pub use real_forward::{
    dispatch_profile_report, PhaseCounters, RealForwardError, RealForwardRunner, RollbackPoint,
};
#[cfg(target_os = "macos")]
pub use real_forward_init::resolve_expert_residency;
// The drafter/speculation policy, shared by `turbospark-check` and
// `turbospark-server`. It lived in the CLI until the server needed the same
// three decisions in the same order; see `speculation_policy`'s own header
// for why it is here rather than copied.
#[cfg(target_os = "macos")]
pub use speculation_policy::{
    draft_policies, resolve_drafter, resolve_speculation, DrafterChoice, Speculation,
    SpeculationPlan, SpeculativeDrafter,
};
pub use speculative::{
    run_raw_completion_speculative, run_raw_completion_speculative_cancellable,
    DEFAULT_SPECULATION_BLOCK,
};
#[cfg(target_os = "macos")]
pub use steering::{SteeringPolicy, SteeringVector, MAX_STEER_ROWS};
pub use turn_stream::{TurnEvent, TurnSplitter};

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
