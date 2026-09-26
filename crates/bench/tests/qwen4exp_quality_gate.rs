#![cfg(target_os = "macos")]
//! The quality gate for `qwen4_exp` (Qwen3.8-Flash-Next), against a REAL
//! `qwen4exp` `.gturbo` install. Same body and same measurements as
//! `quality_gate.rs` (see `quality_common`); only the install, the window,
//! and the pressure arm differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process,
//! the same rule the memory oracles are split under.
//!
//! Not run by default (needs a ~68 GB install; use --release):
//!
//!   TURBOSPARK_QWEN4EXP_INSTALL_DIR=~/.turbospark/models/qwen4-reap288.gturbo \
//!     cargo test -p turbospark-bench --test qwen4exp_quality_gate --release -- --ignored --nocapture
//!
//! **THE DETERMINISM BUG IS FIXED (2026-09-04).** This gate used to be
//! left failing on purpose at arm 3: two back-to-back warm greedy
//! generations of the identical prompt, on the same open runner, produced
//! DIFFERENT SHA-256 digests, while cross-process generation reproduced
//! exactly. Root cause (`docs/QWEN4_EXP.md`'s "The quality gate is
//! BLOCKED" section has the full investigation): `RealQwen4State::reset`
//! rewound the GDN chain and the PLE n-gram context but never zeroed
//! `ple_conv_tail`, the dilated conv's own recurrent tail
//! (`families/qwen4/state.rs`). A fresh process always starts that buffer
//! at the zeros `RealQwen4State::build` writes, so cross-process runs
//! agreed; a second within-process generation after `reset()` started PLE's
//! conv from the first generation's leftover history instead, diverging the
//! wide residual from the first PLE-layer token onward and cascading into a
//! different greedy digest. Fixed by zeroing `ple_conv_tail` in `reset()`
//! alongside the GDN state and the n-gram context. Verified on two fresh
//! processes below, and the gate's own two-generations-per-process arm
//! (arm 3) now passes.
//!
//! **RUNS AT 2,048 CONTEXT, NOT THE SHARED 4,096.** This is the context of
//! the frozen quality row, not a current runtime limit: QSA supports larger
//! windows, and `qwen4exp_qsa_probe` checks its above-budget path on a real
//! install. Keep this regression sentinel at 2,048 so its values remain
//! comparable with their recorded baseline. The gate's own corpus -- the
//! `short-explanation` prompt (62 tokens) plus the reference answer (512
//! tokens) -- fits comfortably within that window.
//!
//! **NO CONSTRAINED-WORKING-SET ARM.** This checkpoint routes top-10 of 288
//! experts, and `PRESSURE_EXPERT_CACHE_SLOTS` (8) is below `top_k`: a cache
//! smaller than `top_k` cannot hold even one token's active experts, so
//! `ExpertCache::plan` aborts rather than merely losing throughput. The next
//! legal value in `ALLOWED_CACHE_SLOTS` above 8 is 16, which is the SAME as
//! the baseline `PROTOCOL_EXPERT_CACHE_SLOTS`, so there is no smaller legal
//! cache size left to demonstrate anything at today's granularity. See
//! `quality_common::run_quality_gate_full`'s header.
//!
//! **NO ASSISTANT PREFIX.** Checked directly against this install with
//! `apply_chat_template` before writing this gate (the same check
//! `qwen38_quality_gate.rs`'s header describes): the rendered prompt ends
//! `<|im_start|>assistant\n<think>\n\n</think>\n\n`, an already-CLOSED, EMPTY
//! think block -- the same non-thinking-mode shape `qwen38-27b`'s template
//! renders. The reference answer lands in the ANSWER position with nothing
//! spliced in, so no prefix is needed.
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, chosen
//! for Gemma 4 and reused verbatim, so this measures `qwen4_exp` on a
//! Gemma-shaped text. Compare each row only against its own past; the
//! absolute perplexity is not a ranking across families, because each
//! family's chat template puts the reference answer in a different position.

mod quality_common;

/// Per-chip rows for `qwen4-reap288`. Frozen 2026-09-04 on AC, release, 16
/// expert-cache slots, from TWO independent fresh processes (this family has
/// no smaller legal expert-cache size to run a constrained arm at -- see the
/// module header) that agreed on the perplexity to the last digit and on
/// both digests to the last hex character.
const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 8.7224,
        greedy_digest: "9f9ed49203dda1aad1b379eb503a3dd8b565d53ccc38ff417fb35e43ed9e0795",
        sampled_digest: "4cf6da5560936e306df09fa09bca7bd19950429e910cee2b66c93c8ed0335772",
        source: "this port, 2026-09-04, Apple M4 Max, AC, 16 slots",
    },
];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_QWEN4EXP_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~68 GB Qwen3.8-Flash-Next-REAP-288 .gturbo install (TURBOSPARK_QWEN4EXP_INSTALL_DIR)"]
fn real_qwen4exp_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "qwen4exp_quality_gate: TURBOSPARK_QWEN4EXP_INSTALL_DIR is not set; skipping. \
             Point it at a streamed Qwen3.8-Flash-Next-REAP-288 .gturbo install to run the gate."
        );
        return;
    };
    let max_context =
        turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::Qwen4Exp)
            .max_context;
    quality_common::run_quality_gate_full(&dir, BASELINES, "", max_context, None);
}
