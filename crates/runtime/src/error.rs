//! Failure modes for the raw-completion loop. Ported from the
//! `GeneratorError` cases exercised by `Runtime/Generation/RawCompletion.swift`.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeError {
    EmptyPrompt,
    ContextOverflow {
        prompt: usize,
        max_new: u32,
        max_context: u32,
    },
    Selection(selection::SelectionError),
    Producer(String),
    ZeroBudget,
    /// Speculative decoding was asked for under a configuration it cannot
    /// serve losslessly. REFUSED rather than silently falling back to the
    /// sequential loop: a caller that asked for speculation and quietly got
    /// none would measure the non-speculative engine and report it as the
    /// speculative one (AGENTS.md Gotcha 35, and `crates/runtime`'s own
    /// argument for `TURBOSPARK_MTP_DRAFT` erroring on a headless install).
    SpeculationUnavailable(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::EmptyPrompt => write!(f, "prompt must not be empty"),
            RuntimeError::ContextOverflow {
                prompt,
                max_new,
                max_context,
            } => write!(
                f,
                "prompt ({prompt}) + max_new ({max_new}) exceeds max_context ({max_context})"
            ),
            RuntimeError::Selection(e) => write!(f, "{e}"),
            RuntimeError::Producer(detail) => write!(f, "logit producer failed: {detail}"),
            RuntimeError::ZeroBudget => write!(f, "max_new_tokens must be greater than 0"),
            RuntimeError::SpeculationUnavailable(detail) => {
                write!(f, "speculative decoding unavailable: {detail}")
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<selection::SelectionError> for RuntimeError {
    fn from(e: selection::SelectionError) -> Self {
        RuntimeError::Selection(e)
    }
}
