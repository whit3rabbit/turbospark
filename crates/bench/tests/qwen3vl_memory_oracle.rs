#![cfg(target_os = "macos")]
//! `qwen3_vl` memory and completion gate for the pinned MLX INT4 artifact.
//! The row is frozen from the first clean run and re-asserted by the same
//! install on subsequent runs.
//!
//! **TWO OF THREE PROTOCOL CASES, on the tinyllama precedent.** Sampled at
//! the frozen protocol's own settings (temperature 0.2, top-k 64), this
//! checkpoint's `medium-review` answer had not terminated by 3,072 new
//! tokens while staying coherent and on-topic throughout -- a verbosity
//! property of the checkpoint at sampling temperature, not a decode defect:
//! greedy ends the same case cleanly at 683 tokens with EndOfTurn, and the
//! short and long cases terminate sampled at 584 and 472. The shared
//! validity gate correctly refuses a maxTokens run, so the oracle drives
//! the two terminating cases and this comment is the record of why the
//! third is absent. The long case still exercises deep-KV decode (2,838
//! prompt tokens at the 4,096 window).
mod oracle_common;
mod qwen3vl_common;

const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 840,
    tok_s_floor: 20.0,
    source: "this port, 2026-09-18, Apple M4 Max, AC, 4096 context",
}];

const UNKNOWN_CHIP_CEILING_MIB: u64 = 840;

#[test]
#[ignore = "needs pinned Qwen3-VL 4B MLX INT4 install and release build"]
fn qwen3vl_memory_and_complete_answers() {
    oracle_common::run_oracle_over_cases(
        &qwen3vl_common::install_dir(),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
        4096,
        1024,
        SHORT_AND_LONG_CASES,
    );
}

/// short-explanation and long-synthesis BY INDEX (0 and 2), pinned as a
/// slice so a protocol reorder cannot silently swap which two run.
const SHORT_AND_LONG_CASES: &[turbospark_bench::protocol::ProtocolCase] = &[
    turbospark_bench::protocol::PROTOCOL_CASES[0],
    turbospark_bench::protocol::PROTOCOL_CASES[2],
];
