#![cfg(target_os = "macos")]
//! The quality gate for Gemma 4 26B-A4B, against a REAL `.gturbo`
//! install. ROADMAP Phase Q's in-repo half: teacher-forced perplexity on
//! the frozen corpus, plus greedy and sampled golden digests.
//! `quality_common` has the reasoning behind every part of it.
//!
//! A SEPARATE TARGET from `qwen36_quality_gate.rs` for the same reason the
//! memory oracles are split: one real model per process.
//!
//! Not run by default (needs a ~14.6 GB install; use --release or the
//! perplexity pass takes ten times as long as it needs to):
//!
//!   MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!     cargo test -p mrefrust-bench --test quality_gate --release -- --ignored --nocapture

mod quality_common;

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST -- the lookup takes the
/// first `contains` hit, so a future bare "Apple M4" row must come after
/// "Apple M4 Max".
///
/// A row cannot be invented: the value of a golden digest is that it was
/// observed on a build known to generate coherent text. On a new chip, run
/// the gate once, read the printed values, and paste them in with the date
/// and power source.
const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Two full runs in separate processes, 2026-08-07, on AC, release,
    // 16 expert-cache slots, at the commit that introduced this gate.
    // Both agreed on ALL THREE values to the last digit and the last hex
    // character, which is the property being frozen: the run order in
    // `quality_common` (warmup, measure, measure, warmup, measure) makes
    // a fresh process reproduce a warm one exactly.
    //
    // 37.31 is high for teacher-forced English, and the reason is
    // structural rather than a quality signal: Gemma 4's chat template
    // opens a `<|channel>thought` block before the assistant slot, so
    // the reference answer is being scored as if it were the model's
    // internal reasoning rather than its reply. Qwen 3.6, whose template
    // opens no channel, scores 6.25 on the same passage. The two numbers
    // are therefore NOT comparable across families; each is only
    // comparable to its own past.
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 37.3105,
        greedy_digest: "4f5cba92159ca76006c39ed6aab76e15da862a36d34bae0f85a82d42690bc38e",
        sampled_digest: "cde6012aae80021227ab3ff775b7517f9e6976df1ab031ba96559a22544c0027",
        // DIFFERENT from greedy_digest above, and expected to be: Gemma's
        // routed slots are ordered misses-first, so an 8-slot cache
        // reduces phase 2 in a different order (quality_common's module
        // doc). Qwen's row has the same value in both fields because its
        // flow does not reorder.
        constrained_digest: "a50ed69d05849d799870f82a44712bf0ea6fce8302cbd920b7c6432febb7f864",
        source: "this port, 2026-08-07, Apple M4 Max, AC, 16 slots",
    },
];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("MREFRUST_GEMMA4_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~14.6 GB Gemma 4 .gturbo install (MREFRUST_GEMMA4_INSTALL_DIR)"]
fn real_gemma4_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "quality_gate: MREFRUST_GEMMA4_INSTALL_DIR is not set; skipping. \
             Point it at a repacked Gemma 4 .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
