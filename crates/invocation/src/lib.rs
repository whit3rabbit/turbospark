//! Invocation crate: pure translation of an ordered list of command-line
//! argument tokens into a validated invocation request, a help
//! short-circuit, or one of six distinguishable typed failures.
//!
//! This crate performs no filesystem, network, environment, or process
//! access, and holds no state between calls. Every claim here is testable
//! by supplying a token list and asserting on the returned outcome.

pub mod diagnostics;
pub mod failure;
pub mod options;
pub mod parser;
pub mod request;
pub mod usage;

pub use diagnostics::{exit_status, stream_routing, ExitStatus, StreamRouting};
pub use failure::ParseFailure;
pub use options::{OptionDecl, OPTIONS};
pub use parser::{parse, ParseOutcome};
pub use request::{InvocationRequest, Mode, PrefillChunk, ReadAheadMode};
pub use usage::render_usage;
