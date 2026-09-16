#![cfg(target_os = "macos")]
//! The quality gate for TinyLlama-1.1B-Chat (Q6_K GGUF intake), against a
//! REAL dense `llama`-flow `.gturbo` install. Same body and same
//! measurements as `quality_gate.rs` (see `quality_common`); only the
//! install and the rows differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process.
//!
//! Not run by default (needs a ~1 GB install; use --release):
//!
//!   TURBOSPARK_DENSE_LLAMA_INSTALL_DIR=~/models/tinyllama-dense.gturbo \
//!     cargo test -p turbospark-bench --test tinyllama_quality_gate --release -- --ignored --nocapture
//!
//! Build the install with `tests/gguf_mixtral_install_network.rs` in
//! `turbospark-repack` (the `repacks_a_real_dense_llama_gguf` case).
//!
//! **NO ASSISTANT PREFIX**: this checkpoint's chat framing is Zephyr
//! (`<|user|>` / `<|assistant|>`, the framing that made Gotcha 41's point
//! that the checkpoint's own template wins over the dialect), which opens
//! an assistant turn and then says words -- there is no structured slot to
//! prefix (crate Gotcha 13's "empty for the other families" case).
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, chosen
//! for Gemma 4 and reused verbatim, so this measures a 1.1B 2023 checkpoint
//! on a Gemma-shaped text; the absolute perplexity will be high and is a
//! property of the pairing, not a defect. Compare each row only against
//! its own past. The trained context is 2,048 and the gate's window is the
//! llama flow's 8,192: positions past training degrade fluency, so judge
//! coherence accordingly and lean on the digest (deterministic settings)
//! rather than the prose.

mod quality_common;

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST. Frozen 2026-09-16 from
/// the first run on this machine (two fresh processes agreeing), then
/// re-run to ASSERT. 17.6084 is high for the reasons the module header
/// gives -- a 1.1B 2023 checkpoint on the frozen protocol's Gemma-shaped
/// corpus -- and is a sentinel against this install's own past, not a
/// ranking. The constrained arm's digest is byte-identical (dense, no
/// expert cache).
const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 17.6084,
    greedy_digest: "0f4d46c395249d53373d172d63200814c0ec6713d78576cb33bd7268f186f474",
    sampled_digest: "3a0bc65a0ffb5ed557c080b46b7238c0d7c4b4f64a2fc2ae86bb47188c365a4b",
    source: "this port, 2026-09-16, Apple M4 Max, 16 slots",
}];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_DENSE_LLAMA_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real TinyLlama dense install via TURBOSPARK_DENSE_LLAMA_INSTALL_DIR"]
fn real_tinyllama_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "tinyllama_quality_gate: TURBOSPARK_DENSE_LLAMA_INSTALL_DIR is not set; \
             skipping. Point it at a streamed TinyLlama .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
