#![cfg(target_os = "macos")]
//! The quality gate for `Qwen/Qwen3.8-27B`, against a REAL `qwen35` `.gturbo`
//! install. Same body and same measurements as `quality_gate.rs` (see
//! `quality_common`); only the install and the rows differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process,
//! the same rule the memory oracles are split under.
//!
//! Not run by default (needs a ~16 GB install; use --release):
//!
//!   TURBOSPARK_QWEN38_INSTALL_DIR=~/models/qwen38-27b.gturbo \
//!     cargo test -p turbospark-bench --test qwen38_quality_gate --release -- --ignored --nocapture
//!
//! Build the install with `tests/qwen38_checkpoint_network.rs` in
//! `turbospark-repack`, which streams `mlx-community/Qwen3.8-27B-4bit` and
//! drops the tokenizer sidecars in beside it.
//!
//! **THE FIRST QUALITY ROW THIS FAMILY HAS EVER HAD.** `qwen35` shipped with
//! Bonsai-27B and got neither a gate nor an oracle row, so until now nothing
//! could see a numerics regression in `families/qwen/`'s DENSE half at all --
//! only the MoE half (`qwen36_quality_gate.rs`) was sentineled, and the two
//! share every line of the flow except the FFN.
//!
//! **NO ASSISTANT PREFIX, AND THAT IS A MEASURED ANSWER RATHER THAN A
//! DEFAULT.** This checkpoint's assistant slot IS structured -- its template
//! opens a `<think>` block in the generation prompt -- which is exactly the
//! shape that made gpt-oss read 148,421.76 without
//! `HARMONY_ASSISTANT_PREFIX` (crate Gotcha 13). It needs none here because
//! `apply_chat_template` renders with `enable_thinking: false`, and this
//! template's own non-thinking branch emits a CLOSED, EMPTY block:
//!
//! ```text
//! <|im_start|>assistant\n<think>\n\n</think>\n\n
//! ```
//!
//! So the slot is already closed when the reference answer arrives, and it
//! lands in the ANSWER position with nothing spliced in. Adding the
//! `\n</think>\n\n` that the thinking branch would have needed writes a
//! SECOND close and measures the model's surprise at that -- the same class
//! of error as omitting one, from the other side. Read the rendered prompt
//! before changing this; `crates/tokenizer`'s `installed_template` target
//! prints it.
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, chosen
//! for Gemma 4 and reused verbatim, so this measures Qwen3.8 on a
//! Gemma-shaped text. Compare each row only against its own past; the
//! absolute perplexities are not a ranking across families, because each
//! family's chat template puts the reference answer in a different position.

mod quality_common;

/// Per-chip rows for `Qwen/Qwen3.8-27B` at INT4, MOST SPECIFIC SUBSTRING
/// FIRST. `quality_gate.rs` explains why a row cannot be written ahead of a
/// run: the digests are a property of the device's reduce order, so there is
/// nothing to write until this target has been run on that device.
const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen 2026-08-14 on AC, release, 16 expert-cache slots, from TWO
    // fresh processes that agreed on the perplexity to the last digit and on
    // both digests to the last hex character. Only the tok/s moved between
    // them (20.516 then 20.279 at 16 slots), and no tok/s is frozen here.
    //
    // **4.9432 IS THE LOWEST PERPLEXITY OF ANY FAMILY IN THIS REPO** (against
    // Qwen 3.6's 6.2536, gpt-oss's 12.0801, qwen3moe's 14.5988, Gemma's
    // 37.4176 and the IQ3 install's 38.3753) and that is NOT a claim that it
    // is the best model. The corpus is the frozen protocol's, chosen for
    // Gemma and reused verbatim, and each family's template puts the
    // reference answer in a different position; the number is a sentinel
    // against its own past and nothing else. What it DOES rule out is the
    // failure this family was most exposed to: a structured assistant slot
    // scored in the wrong position reads in the tens of thousands (gpt-oss
    // read 148,421.76 before its prefix landed), so a healthy single-digit
    // number beside coherent generations says the framing is right.
    //
    // The constrained arm reads 0.98x and 1.00x rather than the 0.85-0.94x
    // the MoE families show, which is the mechanism rather than a surprise:
    // `--expert-cache-slots` sizes a routed-expert cache, and a DENSE model
    // has no routed experts, so the two arms differ only in noise.
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 4.9432,
        greedy_digest: "c3df0095926caeffd24439a04f71ca3f7bff80df4e71813418625d0590187c0a",
        sampled_digest: "f272437c378f65225abe9de1bfb1df4c8fd54eb238bb6de400109161a53de145",
        source: "this port, 2026-08-14, Apple M4 Max, AC, 16 slots",
    },
];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_QWEN38_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~16 GB Qwen3.8-27B .gturbo install (TURBOSPARK_QWEN38_INSTALL_DIR)"]
fn real_qwen38_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "qwen38_quality_gate: TURBOSPARK_QWEN38_INSTALL_DIR is not set; skipping. \
             Point it at a streamed Qwen3.8-27B .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
