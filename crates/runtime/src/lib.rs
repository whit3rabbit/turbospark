//! The raw-completion prefill+decode loop, wiring the `selection` sampler,
//! the `tokenizer` crate's streaming detokenizer and stop matcher, and a
//! pluggable [`LogitProducer`] into one token generation loop. Ported from
//! `Runtime/Generation/RawCompletion.swift` and `LogitProducer.swift`.
#![forbid(unsafe_code)]

mod config;
mod error;
mod producer;
mod raw_completion;
#[cfg(target_os = "macos")]
mod real_forward;
#[cfg(target_os = "macos")]
mod real_forward_gemma4;
#[cfg(target_os = "macos")]
mod real_forward_qwen;
#[cfg(target_os = "macos")]
mod real_forward_qwen_attn;
#[cfg(target_os = "macos")]
mod real_forward_qwen_state;

pub use config::GenerationConfig;
pub use error::RuntimeError;
pub use producer::{ChunkedPrefillRunner, LogitProducer, ScriptedLogitProducer};
pub use raw_completion::{
    run_raw_completion, run_raw_completion_chunked, RawDecodeProgress, RawDecodeResult, StopReason,
};
#[cfg(target_os = "macos")]
pub use real_forward::{
    dispatch_profile_report, PhaseCounters, RealForwardError, RealForwardRunner,
};

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
