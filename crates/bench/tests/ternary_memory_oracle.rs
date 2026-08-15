#![cfg(target_os = "macos")]
//! The memory oracle for `prism-ml/Ternary-Bonsai-27B-mlx-2bit`, against a
//! REAL `qwen35` `.gturbo` install (ROADMAP's ternary entry). Same body, same
//! frozen protocol and the same four assertions as its siblings (see
//! `oracle_common`); the install and the baseline rows differ, and the WINDOW
//! does not -- this family runs the shared 4,096 / 1,024.
//!
//! A SEPARATE TARGET, not a second `#[test]`, for the same reason the others
//! are: the footprint assertion is a whole-session peak, so two models in one
//! process measure against each other's high-water mark.
//!
//! Not run by default (needs an ~8 GB install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
//!     cargo test -p turbospark-bench --test ternary_memory_oracle --release -- --ignored --nocapture
//!
//! Build the install with `tests/ternary_checkpoint_network.rs` in
//! `turbospark-repack`.
//!
//! **THE PREDICTION FOR THIS ROW IS QWEN3.8'S NUMBER, and that is the finding
//! rather than a shortcut.** `qwen38_memory_oracle.rs` measured 660 MiB on a
//! 15.1 GB dense install and closed the accounting on KV (256.0) plus the
//! delta-rule state (144.0) plus the conv tail (7.5) -- terms that are pure
//! functions of the ARCHITECTURE and the window, not of the quantization. This
//! install is the same architecture at 8 GB instead of 15.1, so if the
//! resident weights really are absent from `phys_footprint` (AGENTS.md
//! Gotcha 40, re-derived per install shape as it requires), halving them
//! changes nothing here. A row that came out ~7 GB lower than Qwen3.8's would
//! mean the opposite -- that the weights WERE being counted -- so this target
//! is a second, independent reading of Gotcha 40 as well as a sentinel.

mod oracle_common;

/// Per-chip rows for `prism-ml/Ternary-Bonsai-27B-mlx-2bit`, MOST SPECIFIC
/// SUBSTRING FIRST (`memory_oracle.rs` explains the lookup order).
///
/// No Swift row at any chip and there will not be one: the Swift original has
/// no `qwen3_5` support at all, so every row here is this port measuring
/// itself.
const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen protocol, release, on AC, 16 expert-cache slots (inert here --
    // a dense install has no routed experts to cache), 4,096 context, the
    // session the real checkpoint was streamed in (2026-08-15). TWO readings
    // of this oracle:
    //   peak footprint     661.6 / 657.8 MiB
    //   short-explanation  13.714 / 13.812 tok/s
    //   medium-review      13.617 / 13.155 tok/s
    //   long-synthesis     12.545 / 12.676 tok/s
    // All three cases stopped endOfTurn both times; the replay of
    // short-explanation grew +0.00 then +0.02 MiB.
    //
    // **THE PEAK IS THE INTERESTING NUMBER AND IT IS QWEN3.8'S.** That
    // install is the SAME architecture at INT4 and reads 660.3 MiB on
    // 15,132,916,736 bytes of dense weights; this one reads 661.6 on
    // 7,569,161,216. Same window, same layer graph, HALF the weights, 1.3 MiB
    // of difference -- which is AGENTS.md Gotcha 40 re-derived on a pair that
    // varies nothing else, and is the strongest evidence in the repo that a
    // dense install's resident weights are absent from `phys_footprint`. A
    // row ~7 GB below Qwen3.8's would have meant the opposite.
    //
    // The accounting that is left is the same one that closed there, and it
    // is a function of the ARCHITECTURE and the window rather than of the
    // quantization: KV 256.0 MiB at 4,096, the fixed delta-rule state 144.0
    // (it does not grow with context), the conv tail 7.5.
    //
    // So this ceiling is NOT comparable to the MoE rows: their dominant term
    // (`slots x layers x expert_stride`) is exactly zero here.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 750,
        // 0.73 of the SLOWER of the two readings of the slowest case
        // (12.545), the same margin the mistral, qwen3moe and qwen38 rows
        // take. Crate Gotcha 15's rule satisfied: two readings, floor off the
        // slower. A first draft of this row carried 12.0 -- copied from the
        // qwen38 row, where it is 0.72 of a FASTER model -- which left 4% of
        // margin here and would have flaked on machine state rather than on a
        // regression.
        tok_s_floor: 9.0,
        source: "this port, 2026-08-15, Apple M4 Max, AC, 4096 context",
    },
];

/// Ceiling for an unlisted chip. Memory sizing does not depend on the chip,
/// and on this family it is KV plus a fixed recurrent state, both pure
/// functions of the architecture and the window.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 750;

#[test]
#[ignore = "needs a real Ternary-Bonsai-27B install via TURBOSPARK_TERNARY_INSTALL_DIR"]
fn real_ternary_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_TERNARY_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_TERNARY_INSTALL_DIR is not set");
        return;
    };
    oracle_common::run_oracle(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
    );
}
