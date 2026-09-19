#![cfg(target_os = "macos")]
//! The memory oracle for `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (the
//! HADAMARD-FOLDED Bonsai-2 line, `docs/BONSAI2.md`), against a REAL
//! `qwen35` `.gturbo` install. Same body, same frozen protocol and the same
//! four assertions as its siblings (see `oracle_common`); the install and
//! the rows differ, and the WINDOW does not -- this family runs the shared
//! 4,096 / 1,024.
//!
//! A SEPARATE TARGET, not a second `#[test]`, for the same reason the others
//! are: the footprint assertion is a whole-session peak, so two models in
//! one process measure against each other's high-water mark.
//!
//! Not run by default (needs an ~8 GB install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_BONSAI2_INSTALL_DIR=~/.turbospark/models/text/bonsai2.gturbo \
//!     cargo test -p turbospark-bench --test bonsai2_memory_oracle --release -- --ignored --nocapture
//!
//! Build the install with `tests/bonsai2_checkpoint_network.rs` in
//! `turbospark-repack`.
//!
//! **THE PREDICTION FOR THIS ROW IS THE SIBLING PAIR'S NUMBER.** The dense
//! `qwen3_5` rows all read ~660 MiB at 4,096 context regardless of width --
//! ternary 661.6 on 7.6 GB of weights, qwen38 660.3 on 15.1, bonsai 660
//! on 3.9 -- because a dense install's resident weights are absent from
//! `phys_footprint` (AGENTS.md Gotcha 40) and the counted terms (KV 256.0,
//! the delta-rule state 144.0, the conv tail 7.5) are pure functions of the
//! ARCHITECTURE and the window. This install is the same architecture again
//! (plus 402 sign vectors and ~150 KiB of transform scratch), so the same
//! ~660 MiB is the prediction; a materially higher row would mean the
//! transform plan or its buffers started counting.

mod oracle_common;

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST (`memory_oracle.rs` explains
/// the lookup order).
const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen 2026-09-19 on AC, release, 16 expert-cache slots (inert on a
    // dense install), 4,096 context, from the first stream of the
    // checkpoint: peak 660.1 MiB, short 12.572 / medium 13.199 / long
    // 12.041 tok/s, all three endOfTurn, replay +0.00 MiB.
    //
    // **660 IS THE SIBLING PREDICTION LANDING EXACTLY.** ternary 661.6 on
    // 7.6 GB of weights, qwen38 660.3 on 15.1, bonsai27b 660 on 3.9, now
    // this one on 7.6 plus 402 sign vectors and ~40 KiB of transform
    // scratch -- the counted terms remain KV 256.0 + delta-rule state 144.0
    // + conv tail 7.5, pure functions of the architecture and the window
    // (AGENTS.md Gotcha 40, fourth independent reading). The Hadamard plan
    // added nothing measurable to the footprint.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 750,
        // 0.7 of the slowest reading (12.041), the ternary row's margin
        // recipe (0.73 of its slowest, rounded away from the edge).
        tok_s_floor: 8.5,
        source: "this port, 2026-09-19, Apple M4 Max, AC, 4096 context",
    },
];

/// Ceiling for an unlisted chip. Memory sizing does not depend on the chip,
/// and on this family it is KV plus a fixed recurrent state, both pure
/// functions of the architecture and the window.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 750;

#[test]
#[ignore = "needs a real Hadamard-folded Bonsai-2 install via TURBOSPARK_BONSAI2_INSTALL_DIR"]
fn real_bonsai2_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_BONSAI2_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_BONSAI2_INSTALL_DIR is not set");
        return;
    };
    oracle_common::run_oracle(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
    );
}

/// The catalog half of this row, checked offline on every `cargo test`.
///
/// NOT `#[ignore]`d and needs no install: it asserts that the ceiling and
/// floor above still agree with the `measured` block in `models.json` they
/// were calibrated from. See `oracle_common::assert_agrees_with_catalog`.
#[test]
fn the_baselines_agree_with_the_catalogs_measured_rows() {
    oracle_common::assert_agrees_with_catalog(
        "bonsai2",
        BASELINES,
        turbospark_bench::protocol::PROTOCOL_MAX_CONTEXT,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
