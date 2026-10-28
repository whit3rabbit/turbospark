#![cfg(target_os = "macos")]
//! The memory oracle for the `lfm3` MLX 4-bit install
//! (`LiquidAI/LFM3-1.7B-MLX-4bit` @ d0ca73dc), against a REAL `.gturbo`
//! install. A separate target because the footprint assertion is a
//! whole-session peak.
//!
//!   TURBOSPARK_LFM3_INSTALL_DIR=~/models/lfm3-1.7b-mlx-4bit \
//!     cargo test -p turbospark-bench --test lfm3_memory_oracle --release -- --ignored --nocapture
//!
//! The protocol row (8,192 context, 4,096 new tokens) is DERIVED from the
//! template: the model reasons in a think block before answering, so
//! completions carry a reasoning budget. The `const` block fails the build if
//! the oracle and `protocol_parameters` drift.
//!
//! **TWO OF THREE PROTOCOL CASES, on the `qwen3_vl` precedent.** Measured
//! 2026-10-27: sampled at the protocol's own settings, `medium-review` runs
//! to the 4,096-token budget while coherent (194 of 195 sentences distinct),
//! reasoning at length before it answers. Given 7,600 tokens it stops
//! `endOfTurn` at 7,490 sampled, and greedy has not stopped at 7,600. That is
//! a verbosity property of the checkpoint, not a decode defect, and the
//! shared validity gate rightly refuses a maxTokens run, so the oracle
//! drives the two cases that terminate. The long case still exercises deep
//! KV (2,918 prompt tokens).

mod lfm3_common;
mod oracle_common;

/// Frozen from the first real run. The accounting that predicted the peak,
/// from shapes, before measuring: 24 layers, 4 kv heads at head_dim 128, 6
/// full and 5 sliding (window 256) attention layers, 13 short-conv layers
/// with no KV --
///   KV full   6 layers x 4 kv x 128 x 2 x 2 B x 8,192  =  96.0 MiB
///   KV swa    5 layers x 4 kv x 128 x 2 x 2 B x   384   =   3.75 MiB
///   conv      13 layers x 6 x 2048 x 2 B                =   0.3 MiB
///   sum                                                    ~100 MiB
/// Measured peak 348.6 MiB, so the dense-install residual is ~249 MiB, at
/// the low end of the band other dense rows show. Resident weights do not
/// count (AGENTS.md Gotcha 40).
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    // 348.6 measured; the +21% margin the spark and muse rows take.
    footprint_ceiling_mib: 420,
    // 0.73 of the SLOW STATE of short-explanation (67.8 tok/s). Four runs
    // read 198.1, 67.8, 71.0 and 180.2 on that case, and 171.6, 166.0, 163.9
    // and 166.9 on long-synthesis, on a quiet host for the last three. A hypothesis, not a
    // measurement: a 1.7B decode step is a small per-token pass, so it
    // follows the GPU's clock state, and the short case's 57-token prefill
    // never brings the device up while long-synthesis's 13 s prefill does
    // (memory: cold-device slow pass). The floor sits under the slow state so
    // it catches regressions rather than clock states.
    tok_s_floor: 50.0,
    source: "this port, 2026-10-27, Apple M4 Max, AC, 8192 context, 4096 budget",
}];

const UNKNOWN_CHIP_CEILING_MIB: u64 = 420;

/// short-explanation and long-synthesis BY INDEX (0 and 2), pinned as a
/// slice so a protocol reorder cannot silently swap which two run.
const SHORT_AND_LONG_CASES: &[turbospark_bench::protocol::ProtocolCase] = &[
    turbospark_bench::protocol::PROTOCOL_CASES[0],
    turbospark_bench::protocol::PROTOCOL_CASES[2],
];

const LFM3_MAX_CONTEXT: u32 = 8192;
const LFM3_MAX_NEW: u32 = 4096;

#[test]
#[ignore = "needs the pinned LFM3-1.7B MLX 4-bit install via TURBOSPARK_LFM3_INSTALL_DIR"]
fn real_lfm3_install_peak_footprint_and_throughput_hold() {
    const _: () = {
        let resolved =
            turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::Lfm3);
        assert!(
            resolved.max_context == LFM3_MAX_CONTEXT && resolved.max_new == LFM3_MAX_NEW,
            "this oracle's window/budget must equal protocol_parameters'"
        );
    };
    oracle_common::run_oracle_over_cases(
        &lfm3_common::install_dir(),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
        LFM3_MAX_CONTEXT,
        LFM3_MAX_NEW,
        SHORT_AND_LONG_CASES,
    );
}
