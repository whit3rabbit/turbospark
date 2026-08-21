#![cfg(target_os = "macos")]
//! The quality gate for `ornith-ai/Ornith-1.5-9B`, against a REAL `qwen35`
//! `.gturbo` install. Same body and same measurements as `quality_gate.rs`
//! (see `quality_common`); only the install and the rows differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process,
//! the same rule the memory oracles are split under.
//!
//! Not run by default (needs a ~9 GB install; use --release):
//!
//!   TURBOSPARK_ORNITH9B_INSTALL_DIR=~/models/ornith9b.gturbo \
//!     cargo test -p turbospark-bench --test ornith9b_quality_gate --release -- --ignored --nocapture
//!
//! Build the install with `tests/ornith_install_network.rs` in
//! `turbospark-repack`, which streams the published Q8_0 GGUF and drops the
//! tokenizer sidecars in beside it.
//!
//! **THE FIRST GGUF-DERIVED `qwen35` ROW.** Every other row for this family
//! is an MLX-affine install (`qwen38_quality_gate`, `ternary_quality_gate`),
//! so this one sentinels a combination nothing else covers: the DENSE half of
//! `families/qwen/` reading GGUF Q8_0 block quants. The two axes are
//! independent and both were new when it landed.
//!
//! **NO ASSISTANT PREFIX**, for the reason `qwen38_quality_gate`'s header
//! spells out at length: `apply_chat_template` renders with
//! `enable_thinking: false`, and this template's non-thinking branch emits a
//! CLOSED, EMPTY `<think>` block, so the reference answer lands in the ANSWER
//! position with nothing spliced in. Adding a prefix here would write a
//! SECOND close and measure the model's surprise at that. A healthy
//! single-digit perplexity beside coherent generations is what says the
//! framing is right; the failure this guards against reads in the tens of
//! thousands (gpt-oss read 148,421.76 before its prefix landed).
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, chosen
//! for Gemma 4 and reused verbatim. Compare each row only against its own
//! past; the absolute perplexities are not a ranking across families.

mod quality_common;

/// Per-chip rows for `ornith-ai/Ornith-1.5-9B` at Q8_0, MOST SPECIFIC
/// SUBSTRING FIRST. `quality_gate.rs` explains why a row cannot be written
/// ahead of a run: the digests are a property of the device's reduce order,
/// so there is nothing to write until this target has been run on that
/// device.
const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen 2026-08-20 on AC, release, 16 expert-cache slots.
    //
    // 6.0503 is a healthy single-digit number beside coherent 400-token
    // generations, which is what says the reference answer landed in the
    // ANSWER position -- the failure the header warns about reads in the
    // tens of thousands. It is NOT comparable to another family's row: the
    // corpus is Gemma's and each template puts the answer somewhere else.
    //
    // The constrained arm reads 1.00x with a digest IDENTICAL to the 16-slot
    // one, and both halves of that are the mechanism rather than luck. The
    // digest identity is Gotcha 27's fix (routed slots dispatch in router
    // rank, so output cannot depend on cache state); the 1.00x is that a
    // DENSE model has no routed experts at all, so `--expert-cache-slots`
    // sizes a cache nothing reads and the two arms differ only in noise.
    // Its MoE sibling `ornith35b_quality_gate` reads 0.91x on the same day
    // and the same prompt, which is the contrast that makes this row's
    // reading a statement rather than an absence.
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 6.0503,
        greedy_digest: "37c9bbafd8c2d2733b66a88a36dd9bb6eb23b885ebb05dec8c4b062b4efd3a8f",
        sampled_digest: "db6538f61580d040a6b7d7be59faec1b41d999f86d5f157b4deb3c2a47f6dd4e",
        source: "this port, 2026-08-20, Apple M4 Max, AC, 16 slots",
    },
];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_ORNITH9B_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~9 GB Ornith-1.5-9B .gturbo install (TURBOSPARK_ORNITH9B_INSTALL_DIR)"]
fn real_ornith9b_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "ornith9b_quality_gate: TURBOSPARK_ORNITH9B_INSTALL_DIR is not set; skipping. \
             Point it at a streamed Ornith-1.5-9B .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
