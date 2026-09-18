#![cfg(target_os = "macos")]
//! Dense Qwen2/Qwen2.5 quality and determinism gate for the pinned MLX INT4
//! artifact. Add the first chip row only after the measured values reproduce.
mod quality_common;
mod qwen2_dense_common;

const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 12.4206,
    greedy_digest: "7e0c496bbb45dda54ee61b6ffc0e9f901b10650fb2249f955031a705a4e86dde",
    sampled_digest: "6df8335ac0a03f5ae49577a78b149a18e10013aab8fe9e8a78d087a74b4ffd24",
    source: "this port, 2026-09-17, Qwen2.5-7B Instruct MLX INT4, M4 Max, AC, 8192 context",
}];

#[test]
#[ignore = "needs pinned Qwen2.5 7B MLX INT4 install and release build"]
fn qwen2_dense_quality_and_determinism() {
    quality_common::run_quality_gate_full(
        &qwen2_dense_common::install_dir(),
        BASELINES,
        "",
        8192,
        Some(8),
    );
}
