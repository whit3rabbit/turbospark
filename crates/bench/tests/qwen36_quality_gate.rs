#![cfg(target_os = "macos")]
//! The quality gate for Qwen 3.6 35B-A3B, against a REAL `.gturbo`
//! install. Same body and same three measurements as `quality_gate.rs`
//! (see `quality_common`); only the install and the rows differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process,
//! the same rule the two memory oracles are split under.
//!
//! Not run by default (needs an ~18 GB install; use --release):
//!
//!   TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
//!     cargo test -p turbospark-bench --test qwen36_quality_gate --release -- --ignored --nocapture
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, which
//! was chosen for Gemma 4 and is reused verbatim here -- deliberately, so
//! the two families' rows describe the same workload, but it does mean
//! this measures Qwen on a Gemma-shaped text.

mod quality_common;

/// Per-chip rows for Qwen 3.6 35B-A3B, MOST SPECIFIC SUBSTRING FIRST.
/// `quality_gate.rs` explains why a row cannot be written ahead of a run.
const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Two full runs in separate processes, 2026-08-07, on AC, release,
    // 16 expert-cache slots, at the commit that introduced this gate.
    // Both agreed on all three values exactly.
    //
    // 6.25 against Gemma 4's 37.31 on the SAME passage is not a quality
    // ranking of the two models. Gemma's chat template opens a
    // `<|channel>thought` block before the assistant slot, so its number
    // scores the reference answer as internal reasoning; Qwen's template
    // opens no channel, so its number scores it as a reply. Compare each
    // row only against its own past.
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 6.2536,
        greedy_digest: "c5b52f776861277ae37c54b22978a771150bb60d8b4c354a783dfb9560d92c40",
        sampled_digest: "525cadbc918786b3b7f89f97361c1d350ed5ef71aa28d3636f8769c24fca8eff",
        source: "this port, 2026-08-07, Apple M4 Max, AC, 16 slots",
    },
];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_QWEN36_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~18 GB Qwen 3.6 .gturbo install (TURBOSPARK_QWEN36_INSTALL_DIR)"]
fn real_qwen36_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "qwen36_quality_gate: TURBOSPARK_QWEN36_INSTALL_DIR is not set; skipping. \
             Point it at a repacked Qwen 3.6 .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
