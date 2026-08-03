//! Root error taxonomy extended by downstream engine areas.
//!
//! Downstream crates define their own, richer error enums for their public
//! categories. This module provides a small shared root for common cases that
//! callers can bubble up generically. It depends on no external crate.

use std::fmt;

/// Shared root error.
#[derive(Debug)]
pub enum Error {
    /// A caller-supplied argument was invalid.
    InvalidArgument(String),
    /// An internal invariant was violated.
    Internal(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Error::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    /// Build an invalid-argument error from any string-like value.
    pub fn invalid_argument(msg: impl Into<String>) -> Self {
        Error::InvalidArgument(msg.into())
    }

    /// Build an internal error from any string-like value.
    pub fn internal(msg: impl Into<String>) -> Self {
        Error::Internal(msg.into())
    }
}
