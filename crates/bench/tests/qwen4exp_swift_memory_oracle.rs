#![cfg(target_os = "macos")]
//! Memory oracle for the pinned Swift IQ2_XS GGUF, separate from the REAP-288
//! oracle because the quantized expert stride and resulting footprint differ.
//!
//! It runs the two Qwen4Exp protocol cases that fit the family's 2,048-token
//! quality window. `long-synthesis` is over that window before generation.
//! The M4 Max row was frozen from two fresh release-process readings on
//! 2026-09-26, on AC power, at 2,048 context and 16 expert slots.
//!
//!   TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR=/tmp/qwen4exp-swift-iq2-xs.gturbo \
//!     cargo test -p turbospark-bench --test qwen4exp_swift_memory_oracle \
//!     real_swift_iq2_xs_memory_holds --release -- --ignored --nocapture --exact

mod oracle_common;

/// Two independent readings on the pinned IQ2_XS install, both cases
/// stopping at endOfTurn with no replay growth:
///
///   short-explanation  484 new, 12.682 / 12.666 tok/s
///   medium-review      601 new, 12.367 / 12.243 tok/s
///
/// Peaks were 1,633.8 and 1,633.7 MiB. The ceiling is about 22% above the
/// higher peak; the floor is 0.73x the slowest decode, rounded down. The
/// earlier exploratory reading was lower (1,467.7 MiB) and is not used here.
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 2000,
    tok_s_floor: 8.9,
    source: "this port, 2026-09-26, Apple M4 Max, AC; two fresh-process readings, \
             2048 context, 16 slots",
}];
const UNKNOWN_CHIP_CEILING_MIB: u64 = 3000;

#[test]
#[ignore = "needs the pinned Swift IQ2_XS .gturbo install"]
fn real_swift_iq2_xs_memory_holds() {
    let Some(dir) = std::env::var_os("TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR is not set");
        return;
    };
    let max_context =
        turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::Qwen4Exp)
            .max_context;
    oracle_common::run_oracle_over_cases(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
        max_context,
        turbospark_bench::protocol::PROTOCOL_MAX_NEW,
        &turbospark_bench::protocol::PROTOCOL_CASES[..2],
    );
}

/// The catalog half of this row, checked offline on every `cargo test`.
#[test]
fn the_baselines_agree_with_the_catalogs_measured_rows() {
    oracle_common::assert_agrees_with_catalog(
        "qwen4exp-swift-iq2-xs",
        BASELINES,
        turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::Qwen4Exp)
            .max_context,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
