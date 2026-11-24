#![cfg(target_os = "macos")]
//! The quality gate for Inkling 2 Mini (`inkling2_loop`), unsloth's dynamic UD-Q4_K_XL GGUF (`inkling2-mini-ud-q4kxl`, the original NVFP4 experts with per-expert `.scale` on layers 0-19, Q8_0 on 20-23), against a
//! REAL `.gturbo` install. Same body and measurements as `quality_gate.rs`
//! (`quality_common`): reference-answer perplexity in the assistant slot,
//! a greedy and a sampled digest, and the 8-slot constrained arm whose
//! digest must equal the 16-slot one. One target per install, one real
//! model per process.
//!
//! **THE ASSISTANT PREFIX IS `<|message_model|><|content_text|>`.** The
//! Inkling template writes no generation prompt: the prompt ends after the
//! user segment and the model writes `<|message_model|>` itself, then a
//! `<|content_thinking|>` segment or a `<|content_text|>` answer segment
//! (`tokenizer::ChatDialect::Inkling2`). The reference is a finished answer,
//! so it goes in the text segment, as gpt-oss's goes in Harmony's `final`
//! channel. Without the prefix the slot would score the model's surprise at
//! prose where it writes a role marker.
//!
//! Streamed at the protocol's 16 slots (`open_model_runner_with_context`
//! pins `Streamed`), at the family's 8,192 window. Top-8 routing, so the
//! 8-slot pressure arm is legal.
//!
//!   TURBOSPARK_INKLING2_UD_Q4KXL_INSTALL_DIR=~/.turbospark/models/text/inkling2-mini-ud-q4kxl.gturbo \
//!     cargo test -p turbospark-bench --test inkling2_ud_q4kxl_quality_gate --release -- --ignored --nocapture

mod quality_common;

/// The assistant-slot markup (see the module doc).
const INKLING2_ASSISTANT_PREFIX: &str = "<|message_model|><|content_text|>";

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST. No Swift row exists:
/// every row here is this port measuring itself.
///
/// Frozen 2026-11-23 from TWO fresh processes on AC, release, 16 slots
/// streamed, 8,192 context: perplexity 4.6962 and both digests agreed to the
/// last digit and hex character, and the 8-slot digest EQUALS the 16-slot
/// one in both (0.97x, 0.95x decode). 0.13% under the original's 4.7024 on
/// the same corpus: the experts are the original's NVFP4 bytes on layers
/// 0-19 (with per-expert `.scale`), Q8_0 on 20-23 where the original is
/// MXFP8, and the dense weights are block-quantized.
const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 4.6962,
    greedy_digest: "082414f0e5376a0b3effbcf7a9a5639246612f30d29a00e889dcc0d37c3e2b63",
    sampled_digest: "ec3a4bc569f58aacf5c805552bf3136662f170ab4af9629253b9af0102b13204",
    source: "this port, 2026-11-23, Apple M4 Max, AC, 16 slots streamed, 8192 context",
}];

#[test]
#[ignore = "needs a real Inkling 2 .gturbo install (TURBOSPARK_INKLING2_UD_Q4KXL_INSTALL_DIR)"]
fn real_inkling2_ud_q4kxl_install_quality_holds() {
    let Some(dir) = std::env::var_os("TURBOSPARK_INKLING2_UD_Q4KXL_INSTALL_DIR") else {
        eprintln!("TURBOSPARK_INKLING2_UD_Q4KXL_INSTALL_DIR is not set; skipping. Point it at the inkling2-mini-ud-q4kxl install.");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    assert_eq!(
        repack::peek_manifest_arch(&dir).unwrap(),
        model_io::inkling2_mini(),
        "the gate measures the witnessed Inkling 2 Mini shape"
    );
    // The prefix must be the two special tokens, or the answer is scored in
    // the wrong slot.
    let tokenizer = tokenizer::MfTokenizer::load_from_dir(&dir).expect("tokenizer loads");
    let prefix = tokenizer.encode(INKLING2_ASSISTANT_PREFIX, false);
    assert_eq!(
        prefix.len(),
        2,
        "{INKLING2_ASSISTANT_PREFIX} must encode as two special tokens, got {prefix:?}"
    );
    drop(tokenizer);
    let params =
        turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::Inkling2Loop);
    quality_common::run_quality_gate_full(
        &dir,
        BASELINES,
        INKLING2_ASSISTANT_PREFIX,
        params.max_context,
        Some(8),
    );
}
