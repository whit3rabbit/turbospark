#![cfg(target_os = "macos")]
//! `lfm3` quality and determinism gate for the pinned GGUF Q8_0 artifact
//! (`LiquidAI/LFM3-1.7B-GGUF` @ b3f2ae5d). Empty assistant prefix, for the
//! reason `lfm3_quality_gate` records.
//!
//!   TURBOSPARK_LFM3_Q8_INSTALL_DIR=~/models/lfm3-1.7b-q8 \
//!     cargo test -p turbospark-bench --test lfm3_q8_quality_gate --release -- --ignored --nocapture
mod lfm3_common;
mod quality_common;

/// Frozen after two fresh processes agreed on every digit and hex character.
/// Higher than the MLX 4-bit row's 14.7455 although Q8_0 is the finer
/// quantization: the two are different converters' artifacts, this one's
/// greedy output matches llama.cpp token for token on the same bytes, and
/// the gap is recorded rather than explained.
///
/// RE-FROZEN 2026-10-28 for the GQA-shared decode attention partial, this
/// family's default since then. Perplexity 15.8615 -> 15.8614; greedy digest
/// unchanged, so the llama.cpp token-for-token match stands; the sampled
/// text keeps its first 453 of ~1185 characters, then reads "the peak flow
/// that reaches inland neighborhoods" where it read "the peak flood height
/// and lengthening the duration". Two fresh processes agreed on all three
/// values. The old row still reproduces exactly under
/// `TURBOSPARK_ATTN_DECODE_KERNEL=barrier`: perplexity 15.8615, sampled
/// 6e5a9eb128142e2d7f010bb92d459d45ed44efeb034b6f0b8e94dc1a38e35001.
const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 15.8614,
    greedy_digest: "eb3508651827e9d0aed96bee2554176e1c436e6c01d1ae37541cfb83d62277db",
    sampled_digest: "e458a1a37276d5e460e6e4ae10098f7cf8e92eb8d950653988e689f73c30db16",
    source: "this port, 2026-10-28, re-frozen 2026-10-28 for GQA decode attention, LFM3-1.7B GGUF Q8_0, M4 Max, AC, 8192 context",
}];

#[test]
#[ignore = "needs the pinned LFM3-1.7B Q8_0 install via TURBOSPARK_LFM3_Q8_INSTALL_DIR"]
fn lfm3_q8_quality_and_determinism() {
    quality_common::run_quality_gate_full(
        &lfm3_common::q8_install_dir(),
        BASELINES,
        "",
        8192,
        None,
    );
}
