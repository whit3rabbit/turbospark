#![cfg(target_os = "macos")]
//! The memory oracle for Qwen3-30B-A3B, against a REAL `qwen3moe` `.gturbo`
//! install. Same body, same frozen protocol and the same four assertions as
//! `memory_oracle.rs` (see `oracle_common`); only the install and the
//! baseline rows differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`, for the same reason its two
//! siblings are: the footprint assertion is a whole-session peak and the
//! families do not share a ceiling.
//!
//! Not run by default (needs a ~17 GB install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
//!     cargo test -p turbospark-bench --test qwen3moe_memory_oracle --release -- --ignored --nocapture
//!
//! Build the install with `tests/gguf_qwen3moe_install_network.rs` in
//! `turbospark-repack`, which streams the published `Qwen/Qwen3-30B-A3B-GGUF`
//! Q4_K_M and drops the tokenizer sidecars in beside it.
//!
//! **THIS IS THE CHECKPOINT THAT MAKES THE ORACLE MEANINGFUL FOR THIS DECODE
//! FLOW.** The same flow's other family, Mixtral 8x7B, was deliberately never
//! oracled: 8 experts of 108.9 MiB want 54.5 GiB of slot cache at 16 slots, so
//! its "footprint" is the granularity finding rather than a ceiling worth
//! asserting (AGENTS.md Gotcha 36, ROADMAP Phase M2). At 128 experts of
//! ~2.9 MiB this one streams, and a ceiling means something again.

mod oracle_common;

/// Per-chip rows for Qwen3-30B-A3B, MOST SPECIFIC SUBSTRING FIRST
/// (`memory_oracle.rs` explains the lookup order).
///
/// No Swift row at any chip and there will not be one: the Swift original
/// has no `qwen3moe` support, so every row here is this port measuring
/// itself.
const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen protocol, release, on AC, 16 expert-cache slots, first session
    // after the real checkpoint was streamed (2026-08-10), one reading from
    // `turbospark-bench --model`:
    //   peak footprint     2,747.7 MiB
    //   short-explanation  27.335 tok/s
    //   medium-review      24.049 tok/s
    //   long-synthesis     16.036 tok/s
    // All three cases stopped endOfTurn.
    //
    // **THE PEAK IS HIGHER THAN EITHER OTHER FAMILY'S AND THAT IS ARITHMETIC,
    // NOT A REGRESSION.** The slot cache is `slots x layers x expert_stride`,
    // and this model is 48 layers deep against Gemma 4's 30, at a per-layer
    // stride of 2,654,208 or 3,063,808 bytes (its `ffn_down_exps` is Q4_K on
    // half the layers and Q6_K on the other half). That is 2,094 MiB of slot
    // capacity at 16 slots, against Gemma's ~1.5 GiB, plus a 916 MiB resident
    // core that is mapped AND PINNED (AGENTS.md Gotcha 19). So this
    // checkpoint sits ABOVE the ~1.6-2.2 GiB band the other two hold, and the
    // parity matrix should not imply otherwise. It is still a streaming
    // install -- the whole expert table is 16.36 GiB and only 2.09 of it is
    // ever resident -- which is exactly what Mixtral could not manage.
    //
    // Ceiling 2,900 is the measured peak + ~5%, the same rule the other rows
    // use. It does NOT carry Gemma's extra headroom for expert-slot warming
    // spread, because that spread has only been characterised on Gemma;
    // widen it if a second session lands outside.
    //
    // Floor 12.0 is deliberately loose for a first row: ~25% under the
    // slowest case, and it rests on ONE session. AGENTS.md Gotcha 22 is
    // explicit that cross-session absolute numbers here have repeatedly
    // failed to reproduce, so tighten it after a run on another day, not
    // before.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 2900,
        tok_s_floor: 12.0,
        source: "this port, measured locally -- Swift has no qwen3moe support",
    },
];

/// Unknown chip: memory sizing does not depend on the chip, so hold the one
/// measured ceiling. Throughput is reported but not asserted.
const UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB: u64 = 2900;

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_QWEN3MOE_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~17 GB qwen3moe .gturbo install (TURBOSPARK_QWEN3MOE_INSTALL_DIR)"]
fn real_qwen3moe_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "qwen3moe_memory_oracle: TURBOSPARK_QWEN3MOE_INSTALL_DIR is not set; skipping. \
             Point it at a streamed Qwen3-30B-A3B .gturbo install to run the oracle."
        );
        return;
    };
    oracle_common::run_oracle(&dir, BASELINES, UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB);
}
