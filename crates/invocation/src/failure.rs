//! The closed set of six distinguishable typed parsing failures.
//!
//! Each variant is distinguishable by shape and carries the offending
//! option name (and, for a value failure, the offending text) so a caller
//! never has to inspect a message string. Message wording is not
//! contractual; only the shape and the structured payload are.

/// One of six distinguishable outcomes when a token list fails to parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseFailure {
    /// A token matched none of the documented option spellings.
    UnknownOption {
        /// The exact offending token.
        token: String,
    },
    /// A value-taking option had no token after it.
    MissingValue {
        /// The option that was missing its value.
        option: &'static str,
    },
    /// A supplied value failed its option's type, range, or membership
    /// check. Also reused for the interactive-only leading-message
    /// violation and the sampling cross-check violation, each with a fixed
    /// descriptive payload in place of caller text.
    InvalidValue {
        /// The option whose value was rejected.
        option: &'static str,
        /// The offending text, or a fixed descriptive payload.
        value: String,
    },
    /// The required model-path option was absent.
    MissingRequired {
        /// The absent required option.
        option: &'static str,
    },
    /// Two mode-selecting options were present together.
    MutuallyExclusive {
        /// The first offending option, in the fixed checked order.
        first: &'static str,
        /// The second offending option, in the fixed checked order.
        second: &'static str,
    },
    /// No mode-selecting option was present at all.
    NoModeSelected,
}
