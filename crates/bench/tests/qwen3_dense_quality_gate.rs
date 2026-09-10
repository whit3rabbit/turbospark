#![cfg(target_os = "macos")]
//! Dense Qwen3 per-head normalization and tied-head regression measurement.
mod quality_common;
mod qwen3_dense_common;

// Two fresh processes on AC reproduced all three values on 2026-09-10.
// This row belongs to the pinned 0.6B Q8_0 artifact, not other Qwen3 sizes.
const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 33.5298,
    greedy_digest: "d108df5120d92b47419e9d8096a6bc54120e0f1eb18422295b08ecf13e6a1637",
    sampled_digest: "73e57a2bacd00b20be5febce9289d8f781806faaf5bdacf128d9bf63752d36d0",
    source: "this port, 2026-09-10, Qwen3-0.6B Q8_0, M4 Max, AC, 8192 context",
}];
#[test]
#[ignore = "needs pinned Qwen3-0.6B Q8_0 install and release build"]
fn qwen3_dense_quality_and_determinism() {
    // The checkpoint opens the assistant slot without a forced thought prefix.
    // Dense models allocate no expert slots, so the pressure arm is inapplicable.
    quality_common::run_quality_gate_full(
        &qwen3_dense_common::install_dir(),
        BASELINES,
        "",
        8192,
        None,
    );
}
