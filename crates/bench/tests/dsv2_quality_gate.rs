#![cfg(target_os = "macos")]
//! The quality gate for `deepseek2` (DeepSeek-V2-Lite-Chat), against a REAL
//! `.gturbo` install streamed from `mradermacher/DeepSeek-V2-Lite-Chat-GGUF`
//! Q8_0. Same body and same measurements as `quality_gate.rs` (see
//! `quality_common`); only the install and the rows differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`: one real model per process,
//! the same rule the memory oracles are split under.
//!
//! Not run by default (needs a ~17 GB install; use --release):
//!
//!   TURBOSPARK_DSV2_INSTALL_DIR=~/.turbospark/models/dsv2lite-16b.gturbo \
//!     cargo test -p turbospark-bench --test dsv2_quality_gate --release -- --ignored --nocapture
//!
//! **NO ASSISTANT PREFIX, unlike spark and gpt-oss.** The V2 chat template's
//! generation prompt ends at `Assistant:` and the model's next emission is
//! the answer's prose directly (`docs/DEEPSEEK2_PHASE0.md`'s tokenizer
//! section) -- no think frame, no channel marker -- so a reference answer
//! spliced in raw is exactly what the checkpoint was trained to see. A
//! prefix here would score the model's surprise at markup it never emits.
//!
//! NOTE the corpus and the digest prompt are the frozen protocol's, which was
//! chosen for Gemma 4 and is reused verbatim here -- deliberately, so the
//! families' rows describe the same workload. Compare each row only against
//! its own past; the absolute perplexities are not a ranking across families.

mod quality_common;

/// Per-chip rows for DeepSeek-V2-Lite-Chat Q8_0, MOST SPECIFIC SUBSTRING
/// FIRST. `quality_gate.rs` explains why a row cannot be written ahead of a
/// run.
///
/// No Swift row at any chip and there will not be one: the Swift original
/// has no `deepseek2` support at all, so every row here is this port
/// measuring itself.
///
/// Frozen 2026-09-17 from TWO fresh processes on AC, release, 16
/// expert-cache slots: perplexity and both digests agreed to the last digit
/// and hex character, and the 8-slot constrained digest EQUALS the 16-slot
/// one in both. A third process then ran GREEN against the frozen row.
///
/// **14.3688 IS HEALTHY HERE, and the two-digit reading is not a quality
/// concern**: the absolute perplexities are not a ranking across families
/// because each template puts the reference answer somewhere else
/// (`quality_common`'s module doc has the worked example), and this corpus
/// is Gemma-shaped prose under a GPT2-BPE vocab. What the number IS evidence
/// for: the answer landed in the plain answer slot the no-prefix comment
/// above predicts, and the greedy generations beside it are coherent.
///
/// The constrained-arm throughput RATIO was 0.73x in the freeze's second
/// process (the neighbours' shape: Gemma 0.90, Qwen 0.74) but 1.51x in the
/// first, where the 16-slot decode read 13.1 tok/s against a second-process
/// 24.4 on identical work. That first reading is recorded as machine-state
/// noise on a shared machine, not as a cache-size effect; the assertion is
/// the 0.5 floor, not the direction.
const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 14.3688,
    greedy_digest: "8a0e247593e9487004eeb152723edea46f1e645626d4b496945eef598fab44e9",
    sampled_digest: "d212a8365d648b285f8de6fd19225b802eb0e23d6ab9c0f1789e97b72db99d56",
    source: "this port, 2026-09-17, Apple M4 Max, AC, 16 slots",
}];

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_DSV2_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~17 GB deepseek2 .gturbo install (TURBOSPARK_DSV2_INSTALL_DIR)"]
fn real_deepseek2_install_quality_holds() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "dsv2_quality_gate: TURBOSPARK_DSV2_INSTALL_DIR is not set; skipping. \
             Point it at a streamed DeepSeek-V2-Lite-Chat .gturbo install to run the gate."
        );
        return;
    };
    quality_common::run_quality_gate(&dir, BASELINES);
}
