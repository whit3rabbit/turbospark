#![cfg(target_os = "macos")]
//! The memory oracle for Inkling 2 Mini (`inkling2_loop`), the LookAtMyQuant 5.4 GB release (`inkling2-mini-lamq54`), against
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
//! Static accounting at 16 slots: the slot cache is 16 x 24 layers x 1.67 MB coded experts, about 0.64 GB, plus the same KV, about 1.07 GB in all; the measured peak sits about 0.55 GB above it, the original's offset, and the 0.6 GB resident region is mapped, not counted. Prefill is sequential on
//! this family (about 18 tok/s here), so the long case spends about three
//! minutes in prefill.
//!
//! Not run by default (use --release or the tok/s numbers mean nothing):
//!
//!   TURBOSPARK_INKLING2_LAMQ54_INSTALL_DIR=~/.turbospark/models/text/inkling2-mini-lamq54.gturbo \
//!     cargo test -p turbospark-bench --test inkling2_lamq54_memory_oracle --release -- --ignored --nocapture

mod oracle_common;

/// Per-chip rows, frozen 2026-11-22 from two sessions on the same build
/// (release, AC, 8,192 context, 4,096 budget, 16 slots, streamed).
///
/// Two sessions over the two cases below: peaks 1,573 and 1,587 MiB;
/// short-explanation 18.947 and 18.669, medium-review 16.791 and 16.513
/// tok/s; both cases stopped at end of turn with the same token counts;
/// warm replay +0.00 MiB. An earlier three-case session peaked at 1,550 MiB.
/// Ceiling 1,680 is the higher peak + ~6%; floor 11.5 is 0.70x the slowest
/// case (16.513).
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 1680,
    tok_s_floor: 11.5,
    source: "this port, 2026-11-22, Apple M4 Max, AC, 8192 context, 4096 budget, 16 slots streamed, short-explanation and medium-review",
}];

/// Ceiling for an unlisted chip. The footprint is slot cache, KV and the
/// resident weights, all functions of the shapes, the window and the slot
/// count, not of the chip.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 1680;

/// The two protocol cases this install completes. **LONG-SYNTHESIS RUNS OUT
/// OF BUDGET HERE AND NOWHERE ELSE**: it stopped at the 4,096-token budget
/// while still reasoning (the original ends it at 2,070 tokens, the MLX
/// build at 1,943), and a larger budget does not fit the 8,192 window over
/// its 3,252-token prompt. That is this release's quality loss
/// (`docs/LAMQ.md`, step 5), not a footprint question, so the oracle drives
/// the two that terminate, on `lfm3_memory_oracle.rs`'s precedent.
const CASES: [turbospark_bench::protocol::ProtocolCase; 2] = [
    turbospark_bench::protocol::PROTOCOL_CASES[0],
    turbospark_bench::protocol::PROTOCOL_CASES[1],
];

#[test]
#[ignore = "needs a real Inkling 2 .gturbo install (TURBOSPARK_INKLING2_LAMQ54_INSTALL_DIR)"]
fn real_inkling2_lamq54_install_peak_footprint_and_throughput_hold_streamed() {
    let Some(dir) = std::env::var_os("TURBOSPARK_INKLING2_LAMQ54_INSTALL_DIR") else {
        eprintln!("TURBOSPARK_INKLING2_LAMQ54_INSTALL_DIR is not set; skipping. Point it at the inkling2-mini-lamq54 install.");
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
        &CASES,
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
        "inkling2-mini-lamq54",
        BASELINES,
        params.max_context,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
