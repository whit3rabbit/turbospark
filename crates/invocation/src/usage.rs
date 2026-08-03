//! Usage-text rendering, derived from the single option declaration table.
//!
//! A new option cannot be added to [`crate::options::OPTIONS`] without
//! appearing here. The exact wording is not contractual; only structural
//! completeness (every declared option present, each with a default or an
//! allowed-value description) is part of the public contract.

use crate::options::OPTIONS;

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
