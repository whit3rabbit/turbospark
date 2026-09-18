//! Invocation crate: pure translation of an ordered list of command-line
//! argument tokens into a validated invocation request, a help
//! short-circuit, or one of six distinguishable typed failures.
//!
//! This crate performs no filesystem, network, environment, or process
//! access, and holds no state between calls. Every claim here is testable
//! by supplying a token list and asserting on the returned outcome.

/// Exit status and stream routing diagnostics for CLI invocations.
pub mod diagnostics;
/// Typed parse failures returned when argument parsing fails.
pub mod failure;
/// CLI option declarations and options metadata table.
pub mod options;
/// Invocation command-line argument parser and outcome types.
pub mod parser;
/// Validated invocation request structures and configuration options.
pub mod request;
/// Help text and usage message formatting.
pub mod usage;

pub use diagnostics::{exit_status, stream_routing, ExitStatus, StreamRouting};
pub use failure::ParseFailure;
pub use options::{OptionDecl, OPTIONS};
pub use parser::{parse, ParseOutcome};
pub use request::{
    steering_knob, ExpertCacheSlots, ExpertResidency, InvocationRequest, KvBits, LoadGuard,
    MaxContext, Mode, PowerProfile, PrefillChunk, ReadAheadMode, ReasoningEffort, Speculation,
    SpeculativeDrafter, SteeringMode, ALLOWED_SPECULATION_BLOCKS,
};
pub use usage::{render_usage, render_version, VERSION};
