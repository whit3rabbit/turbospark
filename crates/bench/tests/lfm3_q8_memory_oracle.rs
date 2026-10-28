#![cfg(target_os = "macos")]
//! The memory oracle for the `lfm3` GGUF Q8_0 install
//! (`LiquidAI/LFM3-1.7B-GGUF` @ b3f2ae5d). Same protocol row and the same two
//! cases as `lfm3_memory_oracle`, for the reason recorded there.
//!
//!   TURBOSPARK_LFM3_Q8_INSTALL_DIR=~/models/lfm3-1.7b-q8 \
//!     cargo test -p turbospark-bench --test lfm3_q8_memory_oracle --release -- --ignored --nocapture

mod lfm3_common;
mod oracle_common;

/// Frozen from two runs. Peak 344-347 MiB, the MLX row's to within 5 MiB:
/// the KV and conv state is identical (~100 MiB at 8,192) and resident
/// weights are not charged (AGENTS.md Gotcha 40), so twice the weight bytes
/// cost nothing here. Replay +0.00 to +0.02 MiB.
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    // 346.8 measured; the MLX row's ceiling.
    footprint_ceiling_mib: 420,
    // 0.73 of the slower case (84.1 tok/s). Short case 94.7 and 91.0, long
    // 84.1 and 84.4. Neither run showed the slow state the MLX install's
    // short case reads (lfm3_memory_oracle); if it ever appears here, this
    // floor will flake before it catches a regression.
    tok_s_floor: 61.0,
    source: "this port, 2026-10-28, Apple M4 Max, AC, 8192 context, 4096 budget",
}];

const UNKNOWN_CHIP_CEILING_MIB: u64 = 420;

const LFM3_MAX_CONTEXT: u32 = 8192;
const LFM3_MAX_NEW: u32 = 4096;

const SHORT_AND_LONG_CASES: &[turbospark_bench::protocol::ProtocolCase] = &[
    turbospark_bench::protocol::PROTOCOL_CASES[0],
    turbospark_bench::protocol::PROTOCOL_CASES[2],
];

#[test]
#[ignore = "needs the pinned LFM3-1.7B Q8_0 install via TURBOSPARK_LFM3_Q8_INSTALL_DIR"]
fn real_lfm3_q8_install_peak_footprint_and_throughput_hold() {
    const _: () = {
        let resolved =
            turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::Lfm3);
        assert!(
            resolved.max_context == LFM3_MAX_CONTEXT && resolved.max_new == LFM3_MAX_NEW,
            "this oracle's window/budget must equal protocol_parameters'"
        );
    };
    oracle_common::run_oracle_over_cases(
        &lfm3_common::q8_install_dir(),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
        LFM3_MAX_CONTEXT,
        LFM3_MAX_NEW,
        SHORT_AND_LONG_CASES,
    );
}
