#![cfg(target_os = "macos")]
//! The memory oracle for Inkling 2 Mini (`inkling2_loop`) from unsloth's
//! dynamic UD-Q4_K_XL GGUF (`inkling2-mini-ud-q4kxl`; unsloth
//! `Inkling-2-Mini-UD-Q4_K_XL.gguf` @ `8078d149`), against a REAL `.gturbo`
//! install, streamed at the protocol's 16 expert-cache slots. Same body and
//! the same four assertions as `inkling2_memory_oracle.rs` (`oracle_common`),
//! plus the catalog half: the row's `measured` block must agree.
//!
//! **STREAMED, NOT AUTO**, as for every Inkling row: `auto` now opens a GGUF
//! install resident when it fits, a peak that says nothing about running in
//! little memory. The oracle opens through `open_model_runner_with_context`,
//! which pins `Streamed`.
//!
//! Not run by default (use --release or the tok/s numbers mean nothing):
//!
//!   TURBOSPARK_INKLING2_UD_Q4KXL_INSTALL_DIR=~/.turbospark/models/text/inkling2-mini-ud-q4kxl.gturbo \
//!     cargo test -p turbospark-bench --test inkling2_ud_q4kxl_memory_oracle --release -- --ignored --nocapture

mod oracle_common;

/// Per-chip rows, frozen 2026-11-23 from two runs in separate processes on the
/// same build (release, AC, 8,192 context, 4,096 budget, 16 slots, streamed,
/// the GGUF NVFP4 layers opened native). Peaks 2,701 and 2,705 MiB;
/// short-explanation 19.675 and 19.069, medium-review 18.148 and 17.976,
/// long-synthesis 16.360 and 16.406 tok/s; all three cases stopped at end of
/// turn with the same token counts (1,166, 2,127, 3,328) both times; warm
/// replay +0.00 MiB. Ceiling 2,870 is the higher peak + ~6%; floor 11.4 is
/// 0.70x the slowest case (16.360).
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 2870,
    tok_s_floor: 11.4,
    source: "this port, 2026-11-23, Apple M4 Max, AC, 8192 context, 4096 budget, 16 slots streamed, three protocol cases",
}];

/// Ceiling for an unlisted chip: slot cache, KV and resident weights are
/// functions of the shapes, the window and the slot count, not the chip.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 2870;

#[test]
#[ignore = "needs a real Inkling 2 .gturbo install (TURBOSPARK_INKLING2_UD_Q4KXL_INSTALL_DIR)"]
fn real_inkling2_ud_q4kxl_install_peak_footprint_and_throughput_hold_streamed() {
    let Some(dir) = std::env::var_os("TURBOSPARK_INKLING2_UD_Q4KXL_INSTALL_DIR") else {
        eprintln!("TURBOSPARK_INKLING2_UD_Q4KXL_INSTALL_DIR is not set; skipping.");
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
        "inkling2-mini-ud-q4kxl",
        BASELINES,
        params.max_context,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
