#![cfg(target_os = "macos")]
//! Dense Qwen2/Qwen2.5 quality and determinism gate for the pinned MLX INT4
//! artifact and the two GGUF artifacts. Add the first chip row only after the
//! measured values reproduce.
mod quality_common;
mod qwen2_dense_common;

const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 12.4206,
    greedy_digest: "7e0c496bbb45dda54ee61b6ffc0e9f901b10650fb2249f955031a705a4e86dde",
    sampled_digest: "6df8335ac0a03f5ae49577a78b149a18e10013aab8fe9e8a78d087a74b4ffd24",
    source: "this port, 2026-09-17, Qwen2.5-7B Instruct MLX INT4, M4 Max, AC, 8192 context",
}];

const Q3KM_BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 12.2878,
    greedy_digest: "a09ceeb8f2c6219eab179f32e509508def28de59e9ab456f29bcbb15c989dca6",
    sampled_digest: "a99a7896ac9c478b4578bf9fcbca431f82c6a0d43093e983f529b4c9161368ad",
    source: "this port, 2026-09-19, Qwen2.5-7B Instruct GGUF Q3_K_M, M4 Max, 8192 context",
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

/// The Dense Qwen2 roadmap item's artifact: the pinned official Q3_K_M GGUF,
/// whose attention and FFN projections run on the Q3_K resident kernels.
/// Each Q3_K superblock carries ~3.4 bits of weight where the MLX artifact's
/// groups carry 4, so a lower number than the MLX row would be suspicious.
#[test]
#[ignore = "needs pinned Qwen2.5 7B GGUF Q3_K_M install and release build"]
fn qwen2_dense_gguf_q3km_quality_and_determinism() {
    quality_common::run_quality_gate_full(
        &qwen2_dense_common::gguf_q3_k_m_install_dir(),
        Q3KM_BASELINES,
        "",
        8192,
        Some(8),
    );
}
