//! Pure mapping from a parse outcome to a process exit status and a stream
//! routing decision.
//!
//! This module emits nothing and terminates nothing; the mapping is plain
//! data that a process entry point (outside this crate) applies.

use crate::failure::ParseFailure;
use crate::parser::ParseOutcome;
use crate::usage::render_usage;

/// The two distinct process exit statuses this unit ever produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitStatus {
    /// A validated invocation or a help request.
    Success,
    /// Any parsing failure, distinct from success.
    InvalidInvocation,
}

impl ExitStatus {
    /// The documented raw process exit code for this status.
    pub fn code(self) -> i32 {
        match self {
            ExitStatus::Success => 0,
            ExitStatus::InvalidInvocation => 2,
        }
    }
}

/// Where an outcome's text belongs: the primary stream, the diagnostic
/// stream, or nowhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamRouting {
    /// Text for the primary stream, if any.
    pub primary: Option<String>,
    /// Text for the diagnostic stream, if any.
    pub diagnostic: Option<String>,
}

/// Map a parse outcome to its exit status.
///
/// A validated invocation and a help request both map to the success
/// status; every parsing failure maps to the distinct invalid-invocation
/// status.
pub fn exit_status(outcome: &ParseOutcome) -> ExitStatus {
    match outcome {
        ParseOutcome::Success(_) | ParseOutcome::Help | ParseOutcome::Version => {
            ExitStatus::Success
        }
        ParseOutcome::Failure(_) => ExitStatus::InvalidInvocation,
    }
}

/// Map a parse outcome to its stream routing.
///
/// A help request's usage text goes on the primary stream. A failure's
/// diagnostic plus usage text goes on the diagnostic stream, with the
/// primary stream left empty. A validated invocation writes nothing here;
/// what it triggers next is outside this crate's responsibility.
pub fn stream_routing(outcome: &ParseOutcome) -> StreamRouting {
    match outcome {
        ParseOutcome::Success(_) => StreamRouting {
            primary: None,
            diagnostic: None,
        },
        ParseOutcome::Help => StreamRouting {
            primary: Some(render_usage()),
            diagnostic: None,
        },
        ParseOutcome::Version => StreamRouting {
            primary: Some(crate::usage::render_version()),
            diagnostic: None,
        },
        ParseOutcome::Failure(failure) => StreamRouting {
            primary: None,
            diagnostic: Some(format!("{}\n{}", describe(failure), render_usage())),
        },
    }
}

/// A non-contractual, human-readable description of a failure. Exact
/// wording is never asserted on; only the fact that some diagnostic text is
/// produced.
fn describe(failure: &ParseFailure) -> String {
    format!("{failure:?}")
}
