#![cfg(target_os = "macos")]
//! `lfm3` quality and determinism gate for the pinned MLX 4-bit artifact
//! (`LiquidAI/LFM3-1.7B-MLX-4bit` @ d0ca73dc).
//!
//!   TURBOSPARK_LFM3_INSTALL_DIR=~/models/lfm3-1.7b-mlx-4bit \
//!     cargo test -p turbospark-bench --test lfm3_quality_gate --release -- --ignored --nocapture
//!
//! The assistant prefix is EMPTY: our render runs with thinking off, so the
//! generation prompt already ends `<think></think>`, which is exactly where
//! the template places the answer's prose. Dense, so the constrained
//! expert-cache arm does not apply.
mod lfm3_common;
mod quality_common;

/// Frozen after two fresh processes agreed on every digit and hex character.
const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 14.7455,
    greedy_digest: "16cc1e9cfafd99f88724ea76fad0460ef760f04ecea8e6f042fd9e185fe208a6",
    sampled_digest: "58d4c54fc228db127854ea254ff4125f73840f781433036f34a2449ba5fbcf4b",
    source: "this port, 2026-10-27, LFM3-1.7B MLX 4-bit, M4 Max, AC, 8192 context",
}];

#[test]
#[ignore = "needs the pinned LFM3-1.7B MLX 4-bit install via TURBOSPARK_LFM3_INSTALL_DIR"]
fn lfm3_quality_and_determinism() {
    quality_common::run_quality_gate_full(&lfm3_common::install_dir(), BASELINES, "", 8192, None);
}
