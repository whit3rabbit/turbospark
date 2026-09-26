#![cfg(target_os = "macos")]
//! Quality gate for the pinned Swift IQ2_XS GGUF. This is a separate row
//! from `qwen4exp_quality_gate`: same family, different quantized artifact.
//!
//! Not run by default. Run twice in separate release processes before
//! freezing the M4 Max row below:
//!
//!   TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR=/tmp/qwen4exp-swift-iq2-xs.gturbo \
//!     cargo test -p turbospark-bench --test qwen4exp_swift_quality_gate \
//!     real_swift_iq2_xs_quality_holds --release -- --ignored --nocapture --exact
//!
//! The reference-answer perplexity and both digests are this port's own
//! regression sentinels. They are not Swift-engine parity or cross-family
//! quality scores.

mod quality_common;

/// Frozen after two independent fresh processes agreed on the M4 Max.
const BASELINES: &[quality_common::ChipQuality] = &[quality_common::ChipQuality {
    brand_substr: "Apple M4 Max",
    perplexity: 4.5957,
    greedy_digest: "64e1ea5e83c754b4384ad5d4dd6f7bd544ded65ad6ead6b87673d990ccbe1408",
    sampled_digest: "cc5611889f6f772a0bd21fbbe202eaf7787c3a383d01596ae0b567be70fdb27b",
    source: "2026-09-26, M4 Max, AC; two independent fresh-process runs on pinned IQ2_XS bytes",
}];

#[test]
#[ignore = "needs the pinned Swift IQ2_XS .gturbo install"]
fn real_swift_iq2_xs_quality_holds() {
    let Some(dir) =
        std::env::var_os("TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR").map(std::path::PathBuf::from)
    else {
        eprintln!("skipping: TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR is not set");
        return;
    };
    let max_context =
        turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::Qwen4Exp)
            .max_context;
    quality_common::run_quality_gate_full(&dir, BASELINES, "", max_context, None);
}
