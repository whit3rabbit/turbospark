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
    // RE-FROZEN 2026-08-11 on AC, release, 16 expert-cache slots, reproduced
    // in two separate processes -- perplexity to the last digit and both
    // digests to the last hex character. The prior row was 14.7576 /
    // 65f57a1a... / c7657395..., measured 2026-08-10.
    //
    // WHY IT MOVED, and why this is a re-freeze rather than a regression:
    // chat framing now comes from the checkpoint's own template instead of
    // the special-token dialect (AGENTS.md Gotcha 41), and THIS FAMILY IS
    // THE ONLY ONE OF THE FOUR WHOSE BYTES CHANGE. The protocol prompt is a
    // file ending in `\n`; the dialect renderer called `raw.trim()`
    // unconditionally, while a template trims only if it says so. Gemma's
    // and Qwen 3.6's say `| trim`, so those three rows are untouched to the
    // last hex character. Qwen3-30B-A3B's does not, so one trailing newline
    // now reaches the model -- which is what HF, vLLM and llama.cpp send it,
    // and what the port was wrong to strip.
    //
    // The move is ONE TOKEN of prompt, and the numbers say so: perplexity
    // -1.08% (inside the 2% tolerance, and in the improving direction),
    // greedy and sampled digests both changed, constrained-vs-16-slot
    // byte-identity still holds. `crates/tokenizer/tests/installed_template.rs`
    // pins the trim behaviour per family so this cannot move again unnoticed.
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 14.5988,
        greedy_digest: "b9211e34142a2e9c6fd60feca13600d1a2e668885d3dadedeeaa25debd5cc848",
        sampled_digest: "92c294a3bb2ec623583e76f412c61d705381d9aa122ebace8d8d97be14a1c874",
        source: "this port, 2026-08-11, Apple M4 Max, AC, 16 slots",
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
