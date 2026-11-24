#![cfg(target_os = "macos")]
//! The memory oracle for Inkling 2 Mini (`inkling2_loop`), the original checkpoint (`inkling2-mini`), against
//! a REAL `.gturbo` install, streamed at the protocol's 16 expert-cache
//! slots. Same body and the same four assertions as `memory_oracle.rs`
//! (`oracle_common`): session peak under the ceiling, every case stops at
//! end of turn, decode at or above the floor, and a warm replay stops
//! growing. One target per install, since two ceilings cannot share one
//! process.
//!
//! **STREAMED, NOT AUTO.** On a 36 GB M4 Max `auto` residency copies the
//! whole expert table into GPU memory (`resident_experts_decision`), a peak
//! around 17.7 GiB that says nothing about running in little memory. The
//! oracle opens through `open_model_runner_with_context`, which pins
//! `Streamed`, so the peak is the slot cache the other MoE rows measure.
//!
//! Static accounting at 16 slots: the slot cache is 16 x 24 layers x 4.49 MB NVFP4 or MXFP8 experts, about 1.7 GB, plus KV of about 0.27 GB on the 8 full-attention slots at 8,192 and the 0.16 GB SWA ring. That is about 2.1 GB; the measured peak sits about 0.6 GB above it (the process, Metal pipelines and host buffers, the same offset the LAMQ install shows), while the ~3.5 GB resident core is mapped, not counted (the deepseek2 row's finding). Prefill is sequential on
//! this family (about 18 tok/s here), so the long case spends about three
//! minutes in prefill.
//!
//! Not run by default (use --release or the tok/s numbers mean nothing):
//!
//!   TURBOSPARK_INKLING2_INSTALL_DIR=~/.turbospark/models/text/inkling2-mini.gturbo \
//!     cargo test -p turbospark-bench --test inkling2_memory_oracle --release -- --ignored --nocapture

mod oracle_common;

/// Per-chip rows, frozen 2026-11-22 from two sessions on the same build
/// (release, AC, 8,192 context, 4,096 budget, 16 slots, streamed).
///
/// Session 1: peak 2,651 MiB; short-explanation 17.977, medium-review 16.917,
/// long-synthesis 14.511 tok/s. Session 2: peak 2,662 MiB; 17.308, 16.175,
/// 15.610 tok/s. Every case stopped at end of turn in both, with the same
/// token counts (greedy); warm replay +0.00 MiB. Ceiling 2,800 is the
/// higher peak + ~5%; floor 10.0 is 0.70x the slowest case (14.511).
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 2800,
    tok_s_floor: 10.0,
    source: "this port, 2026-11-22, Apple M4 Max, AC, 8192 context, 4096 budget, 16 slots streamed",
}];

/// Ceiling for an unlisted chip. The footprint is slot cache, KV and the
/// resident weights, all functions of the shapes, the window and the slot
/// count, not of the chip.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 2800;

#[test]
#[ignore = "needs a real Inkling 2 .gturbo install (TURBOSPARK_INKLING2_INSTALL_DIR)"]
fn real_inkling2_install_peak_footprint_and_throughput_hold_streamed() {
    let Some(dir) = std::env::var_os("TURBOSPARK_INKLING2_INSTALL_DIR") else {
        eprintln!("TURBOSPARK_INKLING2_INSTALL_DIR is not set; skipping. Point it at the inkling2-mini install.");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    assert_eq!(
        repack::peek_manifest_arch(&dir).unwrap(),
        model_io::inkling2_mini(),
        "the oracle measures the witnessed Inkling 2 Mini shape"
    );
    let params =
        turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::Inkling2Loop);
    oracle_common::run_oracle_over_cases_with_slots(
        &dir,
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
        params.max_context,
        params.max_new,
        &turbospark_bench::protocol::PROTOCOL_CASES,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS,
    );
}

/// The catalog half of this row, checked offline on every `cargo test`:
/// the ceiling and floor still agree with the `measured` block in
/// `models.json` they were calibrated from.
#[test]
fn the_baselines_agree_with_the_catalogs_measured_rows() {
    let params =
        turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::Inkling2Loop);
    oracle_common::assert_agrees_with_catalog(
        "inkling2-mini",
        BASELINES,
        params.max_context,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
