#![cfg(target_os = "macos")]
//! The quality gate for `gpt-oss-20b`, against a REAL `gptOss` `.gturbo`
//! install (ROADMAP M5). Same body and same measurements as
//! `quality_gate.rs` (see `quality_common`); only the install and the rows
//! differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process,
//! the same rule the memory oracles are split under.
//!
//! Not run by default (needs a ~12 GB install; use --release):
//!
//!   TURBOSPARK_GPTOSS_INSTALL_DIR=~/models/gptoss-20b.gturbo \
//!     cargo test -p turbospark-bench --test gptoss_quality_gate --release -- --ignored --nocapture
//!
//! **THIS IS THE FIRST GATE HERE WHOSE PROMPT READS A CLOCK, and without the
//! pin below it would expire at midnight.** Harmony's template writes
//! `Current date: ` into its system preamble via transformers'
//! `strftime_now`, so the rendered prompt -- and therefore both frozen
//! digests and the perplexity taken over it -- is a function of the day it
//! was measured. A digest that changes daily is worse than no digest: it
//! reads as a numerics regression on the first morning after it was frozen.
//!
//! The pin lives HERE rather than in the renderer because the renderer's job
//! is to match what transformers, vLLM and llama.cpp send, which is the real
//! date. Only the measurement needs determinism, so only the measurement asks
//! for it (`jinja_chat_template::CHAT_DATE_ENV`).
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, chosen
//! for Gemma 4 and reused verbatim, so this measures gpt-oss on a
//! Gemma-shaped text. Compare each row only against its own past; the
//! absolute perplexities are not a ranking across families, because each
//! family's chat template puts the reference answer in a different position.

mod quality_common;

/// The date every measurement of this family renders under. Arbitrary but
/// FIXED; see the module header for why it exists at all.
const PINNED_CHAT_DATE: &str = "2026-01-01";

/// What a Harmony assistant turn opens with, after the generation prompt's
/// trailing `<|start|>assistant`.
///
/// The reference answer is a finished ANSWER, so it belongs in the `final`
/// channel; the model would otherwise reason first. Without this the gate
/// scores the model's surprise that an assistant turn began with prose
/// instead of a channel marker and reads 148,421.76 -- see
/// `quality_common::reference_perplexity`.
const HARMONY_ASSISTANT_PREFIX: &str = "<|channel|>final<|message|>";

/// Per-chip rows for `gpt-oss-20b`, MOST SPECIFIC SUBSTRING FIRST.
/// `quality_gate.rs` explains why a row cannot be written ahead of a run.
const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen 2026-08-12 on AC, release, 16 expert-cache slots.
    //
    // THE PERPLEXITY IS ONLY MEANINGFUL WITH `HARMONY_ASSISTANT_PREFIX`, and
    // the difference is not marginal: 12.0801 with it, 148,421.76 without.
    // The second number looks exactly like the genuinely broken Qwen of
    // `5279c88` (255,409 against a frozen 6.2536), and the thing that tells
    // them apart is that this model's generations were coherent throughout --
    // a model that cannot predict its own output does not write fluent prose.
    // See `quality_common::reference_perplexity`.
    //
    // 12.0801 sits inside the band the other four families occupy (6.25 to
    // 38.38) but is NOT comparable to them as a ranking: the corpus is the
    // frozen protocol's, chosen for Gemma, and each family's template puts
    // the reference answer in a different position. Compare this row only
    // against its own past.
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 12.0801,
        greedy_digest: "7f7672aee4cdf9752bfb4865295ffdeacf7fb51fe497ce7449316c1a0160be95",
        sampled_digest: "e81441b289c8f1c88382ed8cdb9ea1f598e12771f8af1ef3141c6d5d3fc71a2d",
        source: "this port, 2026-08-12, Apple M4 Max, AC, 16 slots",
    },
];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_GPTOSS_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~12 GB gpt-oss .gturbo install (TURBOSPARK_GPTOSS_INSTALL_DIR)"]
fn real_gpt_oss_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "gptoss_quality_gate: TURBOSPARK_GPTOSS_INSTALL_DIR is not set; skipping. \
             Point it at a streamed gpt-oss-20b .gturbo install to run the gate."
        );
        return;
    };
    // Before ANY render. One model per process, so this is the whole process's
    // clock and there is nothing else in it to disturb.
    std::env::set_var(tokenizer::CHAT_DATE_ENV, PINNED_CHAT_DATE);
    eprintln!("gptoss_quality_gate: chat template date pinned to {PINNED_CHAT_DATE}");
    quality_common::run_quality_gate_with_assistant_prefix(
        &dir,
        BASELINES,
        HARMONY_ASSISTANT_PREFIX,
    );
}
