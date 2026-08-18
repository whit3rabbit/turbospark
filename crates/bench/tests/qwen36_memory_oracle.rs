#![cfg(target_os = "macos")]
//! The memory oracle for Qwen 3.6, against a REAL 35B-A3B `.gturbo`
//! install. Same body, same frozen protocol, and the same four assertions
//! as `memory_oracle.rs` (see `oracle_common`); only the install and the
//! baseline rows differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`, because the footprint
//! assertion is against a whole-session peak and the two families do not
//! share a ceiling -- Gemma 4 26B-A4B peaks around 2,100-2,200 MiB here,
//! Qwen 3.6 around 1,610. `oracle_common`'s module doc has the full
//! reasoning.
//!
//! Not run by default (needs an ~18 GB install; use --release or the
//! tok/s numbers are meaningless):
//!
//!   TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
//!     cargo test -p turbospark-bench --test qwen36_memory_oracle --release -- --ignored --nocapture
//!
//! Build the install with `tests/qwen36_checkpoint_network.rs` in
//! `turbospark-repack`, which repacks `mlx-community/Qwen3.6-35B-A3B-4bit`
//! and drops the tokenizer sidecars in beside it.
//!
//! NOTE the protocol's prompts were chosen for Gemma 4 and are reused
//! verbatim here. That is deliberate -- the same three cases make the two
//! families' rows comparable -- but it does mean this measures Qwen on a
//! Gemma-shaped workload, not on one chosen for it.

mod oracle_common;

/// Per-chip rows for Qwen 3.6 35B-A3B, MOST SPECIFIC SUBSTRING FIRST
/// (`memory_oracle.rs` explains the lookup order).
///
/// There is no Swift row for this family at any chip: the Swift original
/// does not support Qwen 3.6, so every row here is and will stay this
/// port measuring itself.
const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen protocol, release, on AC, 16 expert-cache slots, warmup
    // discarded, first session after the real checkpoint was repacked
    // (2026-08-07). Two independent readings, `turbospark-bench --model`
    // and this oracle:
    //   peak footprint     1,610 / 1,587 MiB
    //   short-explanation  37.355 / 37.993 tok/s
    //   medium-review      35.623 / 36.219 tok/s
    //   long-synthesis     32.772 / 32.647 tok/s
    //
    // The slowest case agrees to 0.13 tok/s across the two, so the spread
    // is not what makes this row provisional -- the sample count is.
    //
    // The peak is ~500 MiB UNDER Gemma 4 26B-A4B on this machine despite
    // the larger install, and that is the hybrid, not an accounting
    // error: 30 of the 40 layers are gated-DeltaNet and hold no KV at
    // all, only ~2 MiB of fixed recurrent state each, so context growth
    // touches 10 layers instead of 30.
    //
    // Ceiling 1,700 is the measured peak + ~5%, the same rule the Gemma
    // rows use. It does NOT carry Gemma's extra headroom for expert-slot
    // warming spread, because that spread has only been characterised on
    // Gemma; widen it if a second session lands outside.
    //
    // Floor 25.0 is deliberately loose for a first row. It sits ~23%
    // under the slowest case, the same margin the Gemma row uses, but it
    // rests on two readings taken minutes apart in one session rather
    // than across sessions -- and AGENTS.md Gotcha 22 is explicit that
    // cross-session absolute numbers here have repeatedly failed to
    // reproduce. Tighten it after a run on another day, not before.
    // It is still worth having at this width: the sampler fix is worth
    // ~15 tok/s here (the same three cases read 21.4 / 21.0 / 20.0
    // immediately before it, measured on this install), so losing it
    // would fail this floor loudly.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 1700,
        tok_s_floor: 25.0,
        source: "this port, measured locally -- Swift has no Qwen 3.6 support",
    },
];

/// Unknown chip: memory sizing does not depend on the chip, so hold the
/// one measured ceiling. Throughput is reported but not asserted.
const UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB: u64 = 1700;

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_QWEN36_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~18 GB Qwen 3.6 .gturbo install (TURBOSPARK_QWEN36_INSTALL_DIR)"]
fn real_qwen36_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "qwen36_memory_oracle: TURBOSPARK_QWEN36_INSTALL_DIR is not set; skipping. \
             Point it at a repacked Qwen 3.6 .gturbo install to run the oracle."
        );
        return;
    };
    oracle_common::run_oracle(&dir, BASELINES, UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB);
}

/// The catalog half of this row, checked offline on every `cargo test`.
///
/// NOT `#[ignore]`d and needs no install: it asserts that the ceiling and
/// floor above still agree with the `measured` block in `models.json` they
/// were calibrated from. See `oracle_common::assert_agrees_with_catalog`.
#[test]
fn the_baselines_agree_with_the_catalogs_measured_rows() {
    oracle_common::assert_agrees_with_catalog(
        "qwen36",
        BASELINES,
        turbospark_bench::protocol::PROTOCOL_MAX_CONTEXT,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
