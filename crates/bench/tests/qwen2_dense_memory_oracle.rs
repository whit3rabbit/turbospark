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
