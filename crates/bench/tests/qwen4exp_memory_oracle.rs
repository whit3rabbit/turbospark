#![cfg(target_os = "macos")]
//! The memory oracle for `qwen4_exp` (Qwen3.8-Flash-Next), against a REAL
//! `qwen4exp` `.gturbo` install. **THE FIRST SENTINEL OF ANY KIND THIS FAMILY
//! HAS EVER HAD** (`docs/QWEN4_EXP.md`'s "What's next" section, item 1): the
//! family did not decode at all on real hardware until this session's
//! router/shared-expert-gate dtype fix, so nothing could have been measured
//! before now.
//!
//! Not run by default (needs a ~68 GB install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_QWEN4EXP_INSTALL_DIR=~/.turbospark/models/qwen4-reap288.gturbo \
//!     cargo test -p turbospark-bench --test qwen4exp_memory_oracle --release -- --ignored --nocapture
//!
//! Build the install with `turbospark-model pull --repo
//! sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit --alias qwen4-reap288`
//! (`docs/QWEN4_EXP.md`).
//!
//! **THIS ROW COVERS TWO OF THE THREE PROTOCOL CASES, NOT ALL THREE, AND ITS
//! NUMBERS ARE NOT COMPARABLE TO ANY OTHER FAMILY'S.** This frozen row uses
//! 2,048 context, the index budget recorded by the checkpoint when the row
//! was established. QSA now supports larger windows, but
//! `long-synthesis` tokenizes to 2,940 under this family's 248,320-entry
//! ChatML vocab, so it cannot fit in this row. A higher-context oracle would
//! need its own resource baseline. `run_oracle_over_cases` (see
//! `oracle_common`) lets this target keep its established two-case protocol;
//! every other family's default oracle still runs all three cases.
//!
//! A SEPARATE TARGET, not a second `#[test]`, for the reason every sibling
//! oracle is: the footprint assertion is a whole-session peak.

mod oracle_common;

/// Per-chip rows for `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`, MOST
/// SPECIFIC SUBSTRING FIRST (`memory_oracle.rs` explains the lookup order).
const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen protocol (short-explanation, medium-review only; long-synthesis
    // does not fit this family's 2,048-token window), release, on AC, 16
    // expert-cache slots, 2,048 context, first session after the
    // router/shared-expert-gate dtype fix (2026-09-04).
    //
    // TWO fresh readings (a run of this 68 GiB install costs ~8-9 minutes,
    // and this is this family's first row of any kind -- see
    // `docs/QWEN4_EXP.md`), both cases stopping `endOfTurn` on both runs:
    //   short-explanation  62 prompt / 380 new tokens, 7.150 / 7.468 tok/s
    //   medium-review      426 prompt / 605 new tokens, 7.530 / 6.870 tok/s
    // Steady-state replay grew +0.00 MiB on the second run; measured peaks
    // were 2503 / 2509 MiB.
    //
    // A THIRD reading on 2026-09-05, after the QSA indexer was wired
    // (`docs/QWEN4_EXP.md`'s "QSA wired end to end"): 2521 MiB, replay
    // +0.02 MiB, 9.386 / 9.808 tok/s. The +12 MiB over the higher 2026-09-04
    // peak is the indexer's own state at this window -- 12 QSA layers of
    // raw-key history (2,048 x 256 B) and pooled blocks (512 x 256 B) is
    // ~7.5 MiB, plus the per-layer scratch and expert-slot warming noise.
    // Every token below the budget now runs the indexer's projection, key
    // copy and block pooling, and the trunk's arithmetic did not move (the
    // quality gate reproduced its frozen row the same day).
    //
    // 288 experts at top-10, 48 layers, ~2.7648 MiB per expert blob
    // (AGENTS.md Gotcha 36): 16 slots is ~2,025.6 MiB of slot capacity alone,
    // leaving ~484 MiB for KV at 2,048 context plus the resident core and
    // process baseline -- consistent with the measured ~2,509 MiB.
    //
    // Ceiling is the higher peak (2509) + ~20% (looser than qwen38's +13%,
    // because expert-slot warming on a 288-expert table has more room to
    // vary than a 128-expert one). tok/s floor is 0.73x the slower of the
    // two readings across both runs (medium-review's 6.870), the same
    // margin `mistral`/`qwen3moe` take (crate Gotcha 15).
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 3000,
        tok_s_floor: 5.0,
        source: "this port, 2026-09-04, Apple M4 Max, AC, 16 slots, 2048 context, \
                  two of three protocol cases (see module header), two readings",
    },
];

/// Ceiling for an unlisted chip. Memory sizing does not depend on the chip;
/// see the row above for the arithmetic this comes from.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 3000;

#[test]
#[ignore = "needs a real Qwen3.8-Flash-Next-REAP-288 install via TURBOSPARK_QWEN4EXP_INSTALL_DIR"]
fn real_qwen4exp_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_QWEN4EXP_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_QWEN4EXP_INSTALL_DIR is not set");
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
///
/// NOT `#[ignore]`d and needs no install: it asserts that the ceiling and
/// floor above still agree with the `measured` block in `models.json` they
/// were calibrated from. See `oracle_common::assert_agrees_with_catalog`.
#[test]
fn the_baselines_agree_with_the_catalogs_measured_rows() {
    oracle_common::assert_agrees_with_catalog(
        "qwen4-reap288",
        BASELINES,
        turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::Qwen4Exp)
            .max_context,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
