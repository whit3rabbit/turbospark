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
/// FIRST. Frozen 2026-09-15 from the first gate run on this machine
/// (release, 16 slots, `mistral7b-dense.gturbo`), after the determinism
/// arm had asserted two fresh processes agree: the gate's own output was
/// transcribed into this row, then this file was re-run to ASSERT it.
///
/// 9.3971 against `qwen38`'s 4.9432 and Ternary's 6.8350 is a model
/// difference, not a quality finding: Mistral 7B is a much smaller, older
/// dense model scored on the frozen protocol's Gemma-shaped corpus, and
/// the row is a sentinel against this install's own past. The constrained
/// arm's digest was byte-identical at first run (dense model, no expert
/// cache, so the arms differ only in order), which is the assertion that
/// matters there.
const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 9.3971,
    greedy_digest: "522026e684c711fdffc33a0e915e88b985d1a83e68c219bd15d83cac608b88b7",
    sampled_digest: "2a98d7607d9fe6bf60d0f25141e3ff4e3d89888544b2d6a9333743ec86f2c5cd",
    source: "this port, 2026-09-15, Apple M4 Max, 16 slots",
}];

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
