#![cfg(target_os = "macos")]
//! The quality gate for the dense `llama` half, against a real Mistral 7B
//! `.gturbo` install (ROADMAP P4.2's dense-llama clause). Same body and same
//! measurements as `quality_gate.rs` (see `quality_common`); only the install
//! and the rows differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process, the
//! same rule the memory oracles are split under. The family's memory row has
//! been frozen since 2026-09-10 (`mistral_memory_oracle.rs`); this file is
//! the quality half the family shipped without.
//!
//! Not run by default (needs a ~4 GB install; use --release):
//!
//!   TURBOSPARK_MISTRAL_INSTALL_DIR=~/models/mistral7b-dense.gturbo \
//!     cargo test -p turbospark-bench --test mistral_quality_gate --release -- --ignored --nocapture
//!
//! **NO ASSISTANT PREFIX**, for the same reason the qwen38 and ternary gates
//! give: this family's chat template opens an assistant turn and then says
//! words, so the reference answer already lands in the answer position
//! (`quality_common` Gotcha 13). Read the rendered prompt before changing
//! this; `crates/tokenizer`'s `installed_template` target prints it.
//!
//! The corpus and the digest prompt are the frozen protocol's, chosen for
//! Gemma 4 and reused verbatim, so this measures a Mistral on a
//! Gemma-shaped text. Compare each row only against its own past.

mod quality_common;

/// Per-chip rows for the dense Mistral 7B install, MOST SPECIFIC SUBSTRING
/// FIRST. Empty until the first run on this machine freezes them, which is
/// `quality_common`'s documented first-run mode: the gate asserts
/// determinism (two fresh processes agreeing), prints the measured
/// perplexity and digests, and refuses to assert against a row that does
/// not exist yet -- the digests are a property of the device's reduce
/// order, so there is nothing to write ahead of a run.
const BASELINES: &[quality_common::ChipQuality] = &[];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_MISTRAL_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real Mistral 7B dense .gturbo install (TURBOSPARK_MISTRAL_INSTALL_DIR)"]
fn real_mistral_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "mistral_quality_gate: TURBOSPARK_MISTRAL_INSTALL_DIR is not set; skipping. \
             Point it at a streamed Mistral 7B .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
