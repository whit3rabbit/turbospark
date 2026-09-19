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
/// protocol are the MLX artifact's, and its peak lands next to the MLX
/// install's 622: at 8192 context the KV cache dominates the footprint, so
/// shrinking the resident payload by half a gigabyte barely moves the row.
/// The ceiling keeps the MLX row's ~6% headroom. Readings and floor are from
/// a BATTERY run (AC was not available); the floor stays a valid collapse
/// sentinel on AC, where readings run higher.
const Q3KM_BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 690,
    tok_s_floor: 13.0,
    source: "this port, 2026-09-19, Apple M4 Max, on battery, 8192 context",
}];

const Q3KM_UNKNOWN_CHIP_CEILING_MIB: u64 = 690;

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

/// The second GGUF artifact. Same window and protocol; the resident payload
/// is ~880 MiB larger than the Q3_K_M one and the peak moves by about that
/// minus the KV-dominated constant the two rows share. Readings and floor
/// are from a BATTERY run (AC was not available); the ceiling keeps the
/// same ~6% headroom over the observed peak the Q3_K_M row keeps.
const Q4KM_BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 690,
    tok_s_floor: 13.0,
    source: "this port, 2026-09-19, Apple M4 Max, on battery, 8192 context",
}];

const Q4KM_UNKNOWN_CHIP_CEILING_MIB: u64 = 690;

#[test]
#[ignore = "needs the single-file Qwen2.5 7B GGUF Q4_K_M install and release build"]
fn qwen2_dense_gguf_q4km_memory_and_complete_answers() {
    oracle_common::run_oracle_with_budget(
        &qwen2_dense_common::gguf_q4_k_m_install_dir(),
        Q4KM_BASELINES,
        Q4KM_UNKNOWN_CHIP_CEILING_MIB,
        8192,
        4096,
    );
}
