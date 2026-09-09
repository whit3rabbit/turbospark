#![cfg(target_os = "macos")]
//! The memory oracle for the `spark2_5` GGUF install (`XHToken/
//! Spark-X2.5-4B-GGUF`), against a REAL `.gturbo` install.
//!
//! A SEPARATE TARGET, not a second `#[test]`, for the same reason the others
//! are: the footprint assertion is a whole-session peak, so two models in one
//! process measure against each other's high-water mark.
//!
//! Not run by default (needs a real install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_SPARK_INSTALL_DIR=~/models/spark25.gturbo \
//!     cargo test -p turbospark-bench --test spark_memory_oracle --release -- --ignored --nocapture
//!
//! **THE PROTOCOL ROW IS DERIVED, NOT MEASURED** (crate Gotcha 12's
//! registry): `protocol_parameters(Spark25)` moved BOTH parameters on muse's
//! reasoning -- the generation prompt forces the think frame open, so
//! completions carry a reasoning budget -- and the first real run is what
//! confirms it. The `const` block below fails the BUILD if the two drift.

mod oracle_common;

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST (`memory_oracle.rs` explains
/// the lookup order).
///
/// No Swift row at any chip and there will not be one: the Swift original
/// has no `spark2_5` support at all, so every row here is this port
/// measuring itself.
///
/// Frozen from the first real run (2026-09-08). The accounting that
/// predicted the peak, computed from the shapes BEFORE it: 36 layers, 4 kv
/// heads at head_dim 256, 9 full layers and 27 sliding ones at window 512 --
///   KV full   9  layers x 4 kv x 256 x 2 x 2 B x 8,192  =  75.0 MiB
///   KV swa    27 layers x 4 kv x 256 x 2 x 2 B x   640   =  17.6 MiB
///             (ring = sliding_window 512 + 128 chunk headroom)
///   ------------------------------------------------------------------
///   sum                                                     92.6 MiB
/// Measured peak 575 MiB, so the dense-install residual (process baseline +
/// host scratch) is ~483 MiB -- between muse's ~350 and Qwen3.8's ~253 plus
/// a wider vocab scratch, the same residual shape every dense row shows.
/// The replay reproduced +0.0 MiB. All three protocol cases stopped
/// endOfTurn at the derived 8,192 / 2,048 row, which confirms the thinking
/// budget fit and freezes the row where it was derived.
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    // 575 measured; the muse row's +21% margin.
    footprint_ceiling_mib: 700,
    // 0.73 of the SLOWEST case (44.822), the margin the mistral, qwen3moe
    // and qwen38 rows take. ONE reading, which is weaker than crate Gotcha
    // 15 asks for and is stated rather than hidden; the spread across the
    // three cases (44.8 to 49.6) is tight for a first run.
    tok_s_floor: 32.0,
    source: "this port, 2026-09-08, Apple M4 Max, AC, 8192 context, 2048 budget",
}];

/// Ceiling for an unlisted chip. Memory sizing does not depend on the chip:
/// on this family it is KV plus a process baseline, both pure functions of
/// the architecture and the window.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 700;

/// The oracle's own copy of the two protocol parameters, asserted equal to
/// the resolver's in a `const` block so the binary and this row cannot drift
/// apart (crate Gotcha 16).
const SPARK_MAX_CONTEXT: u32 = 8192;
const SPARK_MAX_NEW: u32 = 2048;

#[test]
#[ignore = "needs a real spark2_5 install via TURBOSPARK_SPARK_INSTALL_DIR"]
fn real_spark_install_peak_footprint_and_throughput_hold() {
    const _: () = {
        let resolved =
            turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::Spark25);
        assert!(
            resolved.max_context == SPARK_MAX_CONTEXT && resolved.max_new == SPARK_MAX_NEW,
            "this oracle's window/budget must equal protocol_parameters'"
        );
    };
    let Some(dir) = std::env::var_os("TURBOSPARK_SPARK_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_SPARK_INSTALL_DIR is not set");
        return;
    };
    oracle_common::run_oracle_with_budget(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
        SPARK_MAX_CONTEXT,
        SPARK_MAX_NEW,
    );
}
