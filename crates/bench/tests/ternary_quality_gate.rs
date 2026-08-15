#![cfg(target_os = "macos")]
//! The quality gate for `prism-ml/Ternary-Bonsai-27B-mlx-2bit`, against a REAL
//! `qwen35` `.gturbo` install (ROADMAP's ternary entry). Same body and same
//! measurements as `quality_gate.rs` (see `quality_common`); only the install
//! and the rows differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process, the
//! same rule the memory oracles are split under.
//!
//! Not run by default (needs an ~8 GB install; use --release):
//!
//!   TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
//!     cargo test -p turbospark-bench --test ternary_quality_gate --release -- --ignored --nocapture
//!
//! Build the install with `tests/ternary_checkpoint_network.rs` in
//! `turbospark-repack`, which streams the 8.49 GB checkpoint and drops the
//! tokenizer sidecars in beside it.
//!
//! **THE THIRD CHECKPOINT OF ONE ARCHITECTURE, AND THE FIRST CONTROLLED
//! QUANTIZATION PAIR THIS REPO CAN SCORE.** `qwen38_quality_gate.rs` calls the
//! Bonsai/Qwen3.8 pair a controlled comparison for THROUGHPUT and explicitly
//! declines to compare their perplexities, because those are two different
//! trainings. This one is different: Bonsai-27B and Ternary-Bonsai-27B are the
//! same publisher's same base model at two widths, so the perplexity gap
//! between them is a quantization result rather than a model one -- as far as
//! the two QAT recipes are the same, which nothing here establishes. Read the
//! gap as indicative and the row as a sentinel against its own past.
//!
//! **NO ASSISTANT PREFIX**, for `qwen38_quality_gate.rs`'s measured reason:
//! this family's template renders with `enable_thinking: false` and its
//! non-thinking branch emits a CLOSED, EMPTY `<think>` block, so the reference
//! answer already lands in the answer position. Adding the `\n</think>\n\n`
//! the thinking branch would need writes a SECOND close and measures the
//! model's surprise at that. Read the rendered prompt before changing this;
//! `crates/tokenizer`'s `installed_template` target prints it.
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, chosen for
//! Gemma 4 and reused verbatim, so this measures a ternary Qwen on a
//! Gemma-shaped text. Compare each row only against its own past.

mod quality_common;

/// Per-chip rows for `prism-ml/Ternary-Bonsai-27B-mlx-2bit`, MOST SPECIFIC
/// SUBSTRING FIRST. `quality_gate.rs` explains why a row cannot be written
/// ahead of a run: the digests are a property of the device's reduce order, so
/// there is nothing to write until this target has been run on that device.
const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen 2026-08-15 on AC, release, 16 expert-cache slots, from TWO fresh
    // processes that agreed on the perplexity to the last digit and on both
    // digests to the last hex character.
    //
    // **6.8350 AGAINST QWEN3.8-27B's 4.9432 IS THE FIRST SAME-ARCHITECTURE
    // QUANTIZATION COMPARISON IN THIS REPO, and it is not a clean ablation.**
    // Both installs run `families/qwen/`'s dense half at the same shapes on
    // the same corpus with the same framing, so the perplexity gap is not a
    // framing artifact -- but Ternary-Bonsai is prism-ml's QAT checkpoint and
    // Qwen3.8-27B is Qwen's own release quantized by mlx-community, so what
    // separates the two numbers is a TRAINING as well as a width. Read the
    // gap as indicative of nothing in particular and the row as a sentinel
    // against its own past.
    //
    // What the number DOES rule out is this family's characteristic failure:
    // a structured assistant slot scored in the wrong position reads in the
    // tens of thousands (gpt-oss read 148,421.76 before its prefix landed),
    // so a healthy single-digit figure beside coherent generations says the
    // framing is right -- and this checkpoint needs no prefix for the reason
    // the module header gives.
    //
    // The constrained arm reads 0.89x, which is expert-cache noise on a model
    // that HAS no expert cache: `--expert-cache-slots` sizes a routed-expert
    // cache and a dense model has none, so the two arms differ only in run
    // order and thermal state. Its digest is byte-identical to the 16-slot
    // one, which is the assertion that matters (crate Gotcha 9).
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 6.8350,
        greedy_digest: "6a99d87023f889002c21f97b447783c122cb14194f7cfd97e691608a1f010239",
        sampled_digest: "7ea1f8d93c63ab38159b514a9b85697dba23fea5a5effa3742173e4d8dce9523",
        source: "this port, 2026-08-15, Apple M4 Max, AC, 16 slots",
    },
];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_TERNARY_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~8 GB Ternary-Bonsai-27B .gturbo install (TURBOSPARK_TERNARY_INSTALL_DIR)"]
fn real_ternary_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "ternary_quality_gate: TURBOSPARK_TERNARY_INSTALL_DIR is not set; skipping. \
             Point it at a streamed Ternary-Bonsai-27B .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
