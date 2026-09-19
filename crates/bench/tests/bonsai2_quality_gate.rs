#![cfg(target_os = "macos")]
//! The quality gate for `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (the
//! HADAMARD-FOLDED Bonsai-2 line, `docs/BONSAI2.md`), against a REAL
//! `qwen35` `.gturbo` install. Same body and same measurements as
//! `ternary_quality_gate.rs` (see `quality_common`); only the install, the
//! rows, and what the install is PROOF OF differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process,
//! the same rule the memory oracles are split under.
//!
//! Not run by default (needs an ~8 GB install; use --release):
//!
//!   TURBOSPARK_BONSAI2_INSTALL_DIR=~/.turbospark/models/text/bonsai2.gturbo \
//!     cargo test -p turbospark-bench --test bonsai2_quality_gate --release -- --ignored --nocapture
//!
//! Build the install with `tests/bonsai2_checkpoint_network.rs` in
//! `turbospark-repack`, which streams the 8.6 GB checkpoint, carries the
//! Hadamard contract into the install, and drops the tokenizer sidecars
//! beside it.
//!
//! **A HEALTHY ROW HERE IS PROOF THE ACTIVATION TRANSFORMS ARE RIGHT.** This
//! checkpoint cannot be read as a plain affine one: its weights live in a
//! signed block-Hadamard basis, and the engine runs `fwht` dispatches at
//! eleven call sites per layer pair (embedding inverse, both norm outputs,
//! `o_proj`, `out_proj`, `down_proj` inputs, and the head). A missing or
//! wrong transform does not error anywhere -- it produces a fluent-looking
//! model with a catastrophic perplexity, exactly the failure shape
//! `6.25 -> 255,409` taught this repo on the Gemma sandwich norms. The
//! no-assistant-prefix rule carries over from the ternary gate for the same
//! measured reason (this family's non-thinking branch emits a closed, empty
//! `<think>` block), and the corpus is still the frozen Gemma-shaped text,
//! so compare each row only against its own past.

mod quality_common;

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST. `quality_gate.rs` explains
/// why a row cannot be written ahead of a run: the digests are a property of
/// the device's reduce order, so there is nothing to write until this target
/// has been run on that device.
const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen 2026-09-19 on AC, release, 16 expert-cache slots (inert on a
    // dense model), from the first stream of the checkpoint.
    //
    // **5.8134 SITS BETWEEN THE ARCHITECTURE'S OTHER THREE WIDTHS AND IS THE
    // PROOF THE ACTIVATION TRANSFORMS ARE RIGHT.** Same protocol, same
    // corpus: qwen38 (4-bit, mlx-community) 4.9432, this checkpoint (2-bit,
    // folded) 5.8134, ternary27b (2-bit QAT) 6.8350, bonsai27b (1-bit)
    // 8.3554. prism's "98.2% of bf16 retention" claim is exactly the gap to
    // the 4-bit original, and the number is a full point UNDER the first-gen
    // ternary at the same width -- but Bonsai-2 is a new training as well as
    // a new recipe, so read the inter-checkpoint gaps as indicative and the
    // row as a sentinel against its own past. What the number categorically
    // rules out is the family's characteristic silent failure: a missing or
    // misordered Hadamard transform does not error anywhere, it produces a
    // fluent-looking model with a perplexity in the hundreds or thousands.
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 5.8134,
        greedy_digest: "55ad97a5a53b99b9851d70bb002857fbf6d9d95cd64f1254c6ad05df467606de",
        sampled_digest: "ee1d1bf7120d87d4f16536504b859b132ec0b81c2c8252e0b8fd0093fd6494c7",
        source: "this port, 2026-09-19, Apple M4 Max, AC, 16 slots",
    },
];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_BONSAI2_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~8 GB Hadamard-folded Bonsai-2 .gturbo install (TURBOSPARK_BONSAI2_INSTALL_DIR)"]
fn real_bonsai2_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "bonsai2_quality_gate: TURBOSPARK_BONSAI2_INSTALL_DIR is not set; skipping. \
             Point it at a streamed Ternary-Bonsai-2-27B .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
