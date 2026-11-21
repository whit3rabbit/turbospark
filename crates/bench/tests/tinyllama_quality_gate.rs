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

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST. A sentinel against this
/// install's own past, not a ranking. The constrained arm's digest is
/// byte-identical (dense, no expert cache).
///
/// RE-FROZEN 2026-11-18 because the first row recorded a bug. The
/// 2026-09-16 row (17.6084, greedy `0f4d46c3...`, sampled `3a0bc65a...`)
/// was taken while GGUF intake left a `llama` install's `attention_scale` at
/// the Mixtral baseline's 128^-0.5, so this 64-wide head attended at 0.71x
/// the right scale. Its high perplexity was put down to the corpus at the
/// time. Evidence for the new row, all on this machine on AC power:
/// - against llama.cpp on the same Q6_K bytes, the old scale reads 0.339
///   mean nats at 73.3% top-1 (perplexity 17.61 against llama.cpp's 12.52),
///   and the fixed scale reads 6.0e-6 at 99.55% (12.519), under a 1.8e-3
///   backend floor (`docs/verification/tinyllama-q6k-attention-scale-*`);
/// - the old scale, restored by hand on the same install, reproduces the
///   old row exactly (perplexity and both digests);
/// - three fresh processes on the fixed install agree byte for byte, and
///   greedy chat output is coherent.
const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 12.5191,
    greedy_digest: "93e1519d01a2e94e82b073165347b74f2678bd1ae224188df9678759684e6a48",
    sampled_digest: "d080acc0a3b0acfb1127e2642bafca5de2cf7e3d3334b15d23bdd66806bafb0a",
    source: "this port, 2026-11-18, Apple M4 Max, 16 slots",
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
