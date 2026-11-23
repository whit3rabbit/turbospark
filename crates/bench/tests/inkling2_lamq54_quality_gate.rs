#![cfg(target_os = "macos")]
//! The quality gate for Inkling 2 Mini (`inkling2_loop`), the LookAtMyQuant 5.4 GB release (`inkling2-mini-lamq54`), against a
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
//!   TURBOSPARK_INKLING2_LAMQ54_INSTALL_DIR=~/.turbospark/models/text/inkling2-mini-lamq54.gturbo \
//!     cargo test -p turbospark-bench --test inkling2_lamq54_quality_gate --release -- --ignored --nocapture

mod quality_common;

/// The assistant-slot markup (see the module doc).
const INKLING2_ASSISTANT_PREFIX: &str = "<|message_model|><|content_text|>";

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST. No Swift row exists:
/// every row here is this port measuring itself.
///
/// Re-frozen 2026-11-23 from TWO fresh processes on AC, release, 16 slots
/// streamed, 8,192 context, after decode's trellis GEMVs moved to
/// `lamq_gemv_rows` (the kernel the multi-row passes share; the same sums in
/// another order, within about 1e-6 of a row's RMS): perplexity 5.7391 and
/// both digests agreed to the last digit and hex character, and the 8-slot
/// digest EQUALS the 16-slot one in both (0.95x, 0.96x decode). The
/// 2026-11-22 row, from the one-row kernel, was 5.7382 with greedy digest
/// `47ac81ef...` and sampled `b3608a52...`; the move is +0.016%. 22% above
/// the original's 4.7024 on the same corpus, the release's quality loss in
/// this gate's terms (`docs/LAMQ.md`, step 5: about 80% next-token
/// agreement, mean KL 0.25 to 0.31).
const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 5.7391,
    greedy_digest: "90fe18b1690d4548c26e1c946aabb8ee63991ebb14d6aaf98397366bc1fad4aa",
    sampled_digest: "ab0ffe261dfe294f2e0be7da3d586be6db23bfa60a9b9371dd820fee81fd36b9",
    source: "this port, 2026-11-23, Apple M4 Max, AC, 16 slots streamed, 8192 context, lamq_gemv_rows decode",
}];

#[test]
#[ignore = "needs a real Inkling 2 .gturbo install (TURBOSPARK_INKLING2_LAMQ54_INSTALL_DIR)"]
fn real_inkling2_lamq54_install_quality_holds() {
    let Some(dir) = std::env::var_os("TURBOSPARK_INKLING2_LAMQ54_INSTALL_DIR") else {
        eprintln!("TURBOSPARK_INKLING2_LAMQ54_INSTALL_DIR is not set; skipping. Point it at the inkling2-mini-lamq54 install.");
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
