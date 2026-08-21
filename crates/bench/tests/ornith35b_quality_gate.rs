#![cfg(target_os = "macos")]
//! The quality gate for `ornith-ai/Ornith-1.5-35B-A3B`, against the REAL
//! INT4-affine `.gturbo` install. Same body and same measurements as
//! `quality_gate.rs` (see `quality_common`); only the install and the rows
//! differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process,
//! the same rule the memory oracles are split under.
//!
//! Not run by default (needs an ~18 GB install; use --release):
//!
//!   TURBOSPARK_ORNITH35B_INSTALL_DIR=~/models/ornith35b.gturbo \
//!     cargo test -p turbospark-bench --test ornith35b_quality_gate --release -- --ignored --nocapture
//!
//! Build the install with `tests/ornith_mlx_install_network.rs` in
//! `turbospark-repack`, which streams `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`.
//!
//! **THE INT4 INSTALL AND NOT THE Q8_0 ONE, DELIBERATELY.** Both exist on the
//! development machine and they are the same model, but the INT4 one is
//! 1.63-1.67x faster at half the expert stride and is the artifact anything
//! future depends on (it is the only one that clears
//! `MtpState::speculation_blocker`'s dtype arm). Freezing a row for the Q8_0
//! install too would double the gate's cost to sentinel a path nothing
//! builds on; what that install is FOR is the quantization A/B recorded in
//! `docs/BENCHMARKS.md`, which is a throughput measurement and needs no
//! frozen digest.
//!
//! **NO ASSISTANT PREFIX**, for the reason `qwen38_quality_gate`'s header
//! spells out: `apply_chat_template` renders with `enable_thinking: false`
//! and this template's non-thinking branch emits a CLOSED, EMPTY `<think>`
//! block, so the reference answer lands in the ANSWER position with nothing
//! spliced in. A healthy single-digit perplexity beside coherent generations
//! is what says the framing is right.
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, chosen
//! for Gemma 4 and reused verbatim. Compare each row only against its own
//! past; the absolute perplexities are not a ranking across families.

mod quality_common;

/// Per-chip rows for `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`, MOST SPECIFIC
/// SUBSTRING FIRST. `quality_gate.rs` explains why a row cannot be written
/// ahead of a run: the digests are a property of the device's reduce order,
/// so there is nothing to write until this target has been run on that
/// device.
const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen 2026-08-20 on AC, release, 16 expert-cache slots.
    //
    // 6.2298 is a healthy single-digit number beside coherent generations,
    // which is what says the reference answer landed in the ANSWER position.
    // It sits beside Qwen 3.6's 6.2536 on the SAME corpus and the SAME
    // ChatML framing, and that is worth noticing but is not a ranking: this
    // model is a retrain of that architecture, so the two rows are as close
    // to comparable as any pair here, and still each is only a sentinel
    // against its own past.
    //
    // The constrained arm reads 0.91x with a digest IDENTICAL to the 16-slot
    // one. The identity is Gotcha 27's fix -- routed slots dispatch in router
    // rank, so output cannot depend on cache state -- and 0.91x sits in the
    // 0.85-0.94x band the other MoE families show, against the DENSE 9B
    // sibling's 1.00x measured the same day on the same prompt. That
    // contrast is the evidence the ratio is measuring the expert cache and
    // not the weather.
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 6.2298,
        greedy_digest: "8bab701357fc2ebd235906e6200d36512bf8ca756d21ef7847b9a8669c97846f",
        sampled_digest: "8ff05e36c85a5172933b86a07cbde02608c1b944f9f5bf21572883bd64c8e40a",
        source: "this port, 2026-08-20, Apple M4 Max, AC, 16 slots",
    },
];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_ORNITH35B_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~18 GB Ornith-1.5-35B INT4 .gturbo install (TURBOSPARK_ORNITH35B_INSTALL_DIR)"]
fn real_ornith35b_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "ornith35b_quality_gate: TURBOSPARK_ORNITH35B_INSTALL_DIR is not set; skipping. \
             Point it at a streamed Ornith-1.5-35B INT4 .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
