#![cfg(target_os = "macos")]
//! `qwen3_vl` quality and determinism gate for the pinned MLX INT4 artifact.
//! Add the first chip row only after the measured values reproduce.
mod quality_common;
mod qwen3vl_common;

const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 17.3463,
    greedy_digest: "ce10ae69c064b233b3a360f1144b53636206554216c2b908686ffbfac543366d",
    sampled_digest: "6f219a27c5a30606ea3a2bea71b24d05ad4d72806dea188ad616559738fab52e",
    source: "this port, 2026-09-18, Qwen3-VL-4B Instruct MLX INT4, M4 Max, AC, 4096 context",
}];

#[test]
#[ignore = "needs pinned Qwen3-VL 4B MLX INT4 install and release build"]
fn qwen3vl_quality_and_determinism() {
    quality_common::run_quality_gate_full(
        &qwen3vl_common::install_dir(),
        BASELINES,
        "",
        4096,
        None,
    );
}
