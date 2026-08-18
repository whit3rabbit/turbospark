//! Usage-text rendering, derived from the single option declaration table.
//!
//! A new option cannot be added to [`crate::options::OPTIONS`] without
//! appearing here. The exact wording is not contractual; only structural
//! completeness (every declared option present, each with a default or an
//! allowed-value description) is part of the public contract.

use crate::options::OPTIONS;

/// The workspace version, from cargo rather than a literal.
///
/// Every crate here inherits `version.workspace = true`, so this crate's
/// `CARGO_PKG_VERSION` is also the version of whichever binary linked it --
/// which is what lets one pure library answer `--version` for all of them.
/// A hand-maintained constant would be a second place to forget on a
/// release, and the failure mode is a binary confidently reporting the
/// wrong version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Render the version line. Names the ENGINE rather than a binary: the
/// three entry points share one version and this crate cannot know which
/// of them linked it.
pub fn render_version() -> String {
    format!("turbospark {VERSION}\n")
}

/// Render the usage text enumerating every declared option together with
/// its default or its allowed-value description.
pub fn render_usage() -> String {
    let mut out = String::from("usage:\n");
    for opt in OPTIONS {
        out.push_str("  ");
        out.push_str(opt.flag);
        out.push_str("  ");
        out.push_str(opt.usage_hint);
        out.push('\n');
    }
    out
}
