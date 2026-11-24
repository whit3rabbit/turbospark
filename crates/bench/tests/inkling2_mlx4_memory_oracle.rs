#![cfg(target_os = "macos")]
//! The memory oracle for Inkling 2 Mini (`inkling2_loop`), the MLX 4-bit conversion (`inkling2-mini-mlx4`), against
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
//! Static accounting at 16 slots: the experts are the original's NVFP4 and MXFP8 bytes, so the slot cache matches the original's (about 1.7 GB) and so does the peak, about 0.6 GB over slot cache plus KV; the 4-bit dense weights are mapped, not counted. KV as the original's. Prefill is sequential on
//! this family (about 18 tok/s here), so the long case spends about three
//! minutes in prefill.
//!
//! Not run by default (use --release or the tok/s numbers mean nothing):
//!
//!   TURBOSPARK_INKLING2_MLX4_INSTALL_DIR=~/.turbospark/models/text/inkling2-mini-mlx4.gturbo \
//!     cargo test -p turbospark-bench --test inkling2_mlx4_memory_oracle --release -- --ignored --nocapture

mod oracle_common;

/// Per-chip rows, frozen 2026-11-22 from two sessions on the same build
/// (release, AC, 8,192 context, 4,096 budget, 16 slots, streamed).
///
/// Session 1: peak 2,663 MiB; short-explanation 20.516, medium-review 19.552,
/// long-synthesis 17.939 tok/s. Session 2: peak 2,661 MiB; 20.359, 19.463,
/// 17.937 tok/s. Every case stopped at end of turn in both, same token
/// counts; warm replay +0.00 MiB. Ceiling 2,800 is the higher peak + ~5%;
/// floor 12.5 is 0.70x the slowest case (17.937).
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 2800,
    tok_s_floor: 12.5,
    source: "this port, 2026-11-22, Apple M4 Max, AC, 8192 context, 4096 budget, 16 slots streamed",
}];

/// Ceiling for an unlisted chip. The footprint is slot cache, KV and the
/// resident weights, all functions of the shapes, the window and the slot
/// count, not of the chip.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 2800;

#[test]
#[ignore = "needs a real Inkling 2 .gturbo install (TURBOSPARK_INKLING2_MLX4_INSTALL_DIR)"]
fn real_inkling2_mlx4_install_peak_footprint_and_throughput_hold_streamed() {
    let Some(dir) = std::env::var_os("TURBOSPARK_INKLING2_MLX4_INSTALL_DIR") else {
        eprintln!("TURBOSPARK_INKLING2_MLX4_INSTALL_DIR is not set; skipping. Point it at the inkling2-mini-mlx4 install.");
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
        "inkling2-mini-mlx4",
        BASELINES,
        params.max_context,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
