#![cfg(target_os = "macos")]
//! The quality gate for Qwen3-30B-A3B, against a REAL `qwen3moe` `.gturbo`
//! install. Same body and same measurements as `quality_gate.rs` (see
//! `quality_common`); only the install and the rows differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process,
//! the same rule the memory oracles are split under.
//!
//! Not run by default (needs a ~17 GB install; use --release):
//!
//!   TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
//!     cargo test -p turbospark-bench --test qwen3moe_quality_gate --release -- --ignored --nocapture
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, which was
//! chosen for Gemma 4 and is reused verbatim here -- deliberately, so the
//! families' rows describe the same workload, but it does mean this measures
//! Qwen3 on a Gemma-shaped text. Compare each row only against its own past;
//! the absolute perplexities are not a ranking across families, because each
//! family's chat template puts the reference answer in a different position
//! (`qwen36_quality_gate.rs` has the worked example).

mod quality_common;

/// Per-chip rows for Qwen3-30B-A3B, MOST SPECIFIC SUBSTRING FIRST.
/// `quality_gate.rs` explains why a row cannot be written ahead of a run.
const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Measured 2026-08-10 on AC, release, 16 expert-cache slots, the session
    // that streamed the real checkpoint, then reproduced in a second process
    // at the same commit -- perplexity to the last digit and both digests to
    // the last hex character.
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 14.7576,
        greedy_digest: "65f57a1aa684c3acd34c821fb5701ddbbefd0fc1b2f285835d7e8e92853e1bd1",
        sampled_digest: "c76573959430ace79441edc52afbf2aadc34655cd141a0b57c7bdfc53e81683d",
        source: "this port, 2026-08-10, Apple M4 Max, AC, 16 slots",
    },
];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_QWEN3MOE_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~17 GB qwen3moe .gturbo install (TURBOSPARK_QWEN3MOE_INSTALL_DIR)"]
fn real_qwen3moe_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "qwen3moe_quality_gate: TURBOSPARK_QWEN3MOE_INSTALL_DIR is not set; skipping. \
             Point it at a streamed Qwen3-30B-A3B .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
