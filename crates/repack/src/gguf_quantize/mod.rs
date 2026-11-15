//! Recipe-driven GGUF quantizer (Phase A: float sources to F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0).
//!
//! Reads a GGUF whose tensors are F32/F16/BF16, applies a [`Recipe`] that picks
//! a target type per tensor, and streams a new GGUF to disk chunk by chunk, so
//! the largest tensor never has to fit in memory.
//!
//! What this is NOT: a reimplementation of Unsloth's "Dynamic" recipes. Those
//! are not public. A recipe here is whatever the caller writes. See
//! `docs/QUANTIZE.md` for the scope and what is verified.
//!
//! Layout rules the writer keeps:
//! - Metadata is copied key for key. `general.file_type` is rewritten only when
//!   the recipe default is a single known type. An empty metadata array loses
//!   its declared element type (the parser does not keep it); readers see a
//!   zero-length array either way.
//! - 1-D tensors (norms, biases) and tensors whose row length is not a whole
//!   number of blocks are never block-quantized. The latter use the recipe's
//!   `fallback` type.
//! - A tensor that is already quantized is copied verbatim when it keeps its
//!   type. Re-quantizing it would stack error on error, so an explicit rule that
//!   asks for a different type is refused by name.

mod imatrix;
mod iq1;
mod iq2;
mod iq3;
mod iq4;
mod iq_kgrid_tables;
mod iqgrid;
mod kquants;
mod libc_sort;
mod provenance;
mod quantizers;
mod recipe;
mod reference;
mod writer;

#[cfg(test)]
mod tests;

pub use imatrix::{Imatrix, ImatrixError};
pub use iq1::{set_tie_order, TieOrder};
pub use provenance::{
    check_freshness, read_provenance, write_provenance, Finding, HfFile, HfHub, Hub, InputRecord,
    Provenance, RepoFile, RepoListing, Status, GGML_PARITY_COMMIT,
};
pub use quantizers::Target;
pub use recipe::{Choice, Recipe, RecipeError, Rule};
pub use reference::recipe_from_gguf;
pub use writer::{
    quantize_gguf, quantize_gguf_ctl, quantize_gguf_opts, quantize_gguf_with, Control, Progress,
    QuantizeError, QuantizeReport,
};
