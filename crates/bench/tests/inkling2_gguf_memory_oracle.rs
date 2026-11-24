#![cfg(target_os = "macos")]
//! The memory oracle for Inkling 2 Mini (`inkling2_loop`) from a GGUF, the
//! IQ2_M file `docs/INKLING2_PHASE0.md` gates (AtomicChat
//! `Inkling-2-Mini-Uncensored-AD-IQ2_M.gguf` @ `40886ab6`, installed with
//! `repack/examples/inkling2_gguf_install.rs`), against a REAL `.gturbo`
//! install, streamed at the protocol's 16 expert-cache slots. Same body and
//! the same four assertions as `inkling2_memory_oracle.rs` (`oracle_common`).
//! A GGUF install is a local intake, not a catalog row, so there is no
//! catalog half.
//!
//! **STREAMED, NOT AUTO**, as for every Inkling row: `auto` now opens a GGUF
//! install resident when it fits, a peak that says nothing about running in
//! little memory. The oracle opens through `open_model_runner_with_context`,
//! which pins `Streamed`.
//!
//! Not run by default (use --release or the tok/s numbers mean nothing):
//!
//!   TURBOSPARK_INKLING2_GGUF_INSTALL_DIR=~/.turbospark/models/text/inkling2-mini-gguf-iq2m.gturbo \
//!     cargo test -p turbospark-bench --test inkling2_gguf_memory_oracle --release -- --ignored --nocapture

mod oracle_common;

/// Per-chip rows, frozen 2026-11-23 from two runs in separate processes on the
/// same build (release, AC, 8,192 context, 4,096 budget, 16 slots,
/// streamed). Peaks 1,669 and 1,668 MiB; short-explanation 12.919 and
/// 14.594, medium-review 13.346 and 13.829, long-synthesis 13.139 and 13.317
/// tok/s; all three cases stopped at end of turn with the same token counts
/// (1,347, 3,918, 1,631) both times; warm replay +0.00 MiB. Run 1 shared the
/// host with another session's cargo build. Ceiling 1,770 is the higher peak
/// + ~6%; floor 9.0 is 0.70x the slowest case (12.919).
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 1770,
    tok_s_floor: 9.0,
    source: "this port, 2026-11-23, Apple M4 Max, AC, 8192 context, 4096 budget, 16 slots streamed, IQ2_M GGUF, three protocol cases",
}];

/// Ceiling for an unlisted chip: slot cache, KV and resident weights are
/// functions of the shapes, the window and the slot count, not the chip.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 1770;

#[test]
#[ignore = "needs a real Inkling 2 GGUF .gturbo install (TURBOSPARK_INKLING2_GGUF_INSTALL_DIR)"]
fn real_inkling2_gguf_install_peak_footprint_and_throughput_hold_streamed() {
    let Some(dir) = std::env::var_os("TURBOSPARK_INKLING2_GGUF_INSTALL_DIR") else {
        eprintln!("TURBOSPARK_INKLING2_GGUF_INSTALL_DIR is not set; skipping.");
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
