#![cfg(target_os = "macos")]
//! Dense Qwen2/Qwen2.5 memory and completion gate for the pinned MLX INT4
//! artifact. The row is frozen from the first clean run and re-asserted by
//! the same install on subsequent runs.
mod oracle_common;
mod qwen2_dense_common;

const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 660,
    tok_s_floor: 27.0,
    source: "this port, 2026-09-17, Apple M4 Max, AC, 8192 context",
}];

const UNKNOWN_CHIP_CEILING_MIB: u64 = 660;

#[test]
#[ignore = "needs pinned Qwen2.5 7B MLX INT4 install and release build"]
fn qwen2_dense_memory_and_complete_answers() {
    oracle_common::run_oracle_with_budget(
        &qwen2_dense_common::install_dir(),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
        8192,
        4096,
    );
}

/// The GGUF Q3_K_M artifact (Dense Qwen2 roadmap item). Its KV, window and
/// protocol are the MLX artifact's; only the resident payload shrinks (the
/// Q3_K/Q4_K/Q6_K mixture is ~3.5 GiB against the MLX install's ~4.0), so
/// the ceiling tracks the same 40 MiB of headroom the MLX row keeps.
const Q3KM_BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 440,
    tok_s_floor: 27.0,
    source: "this port, 2026-09-19, Apple M4 Max, AC, 8192 context",
}];

const Q3KM_UNKNOWN_CHIP_CEILING_MIB: u64 = 440;

#[test]
#[ignore = "needs pinned Qwen2.5 7B GGUF Q3_K_M install and release build"]
fn qwen2_dense_gguf_q3km_memory_and_complete_answers() {
    oracle_common::run_oracle_with_budget(
        &qwen2_dense_common::gguf_q3_k_m_install_dir(),
        Q3KM_BASELINES,
        Q3KM_UNKNOWN_CHIP_CEILING_MIB,
        8192,
        4096,
    );
}
