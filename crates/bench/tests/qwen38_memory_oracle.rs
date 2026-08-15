#![cfg(target_os = "macos")]
//! The memory oracle for `Qwen/Qwen3.8-27B`, against a REAL `qwen35`
//! `.gturbo` install. Same body, same frozen protocol and the same four
//! assertions as its siblings (see `oracle_common`); the install and the
//! baseline rows differ, and the WINDOW does not -- this family runs the
//! shared 4,096 / 1,024.
//!
//! A SEPARATE TARGET, not a second `#[test]`, for the same reason the others
//! are: the footprint assertion is a whole-session peak, so two models in one
//! process measure against each other's high-water mark.
//!
//! Not run by default (needs a ~16 GB install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_QWEN38_INSTALL_DIR=~/models/qwen38-27b.gturbo \
//!     cargo test -p turbospark-bench --test qwen38_memory_oracle --release -- --ignored --nocapture
//!
//! Build the install with `tests/qwen38_checkpoint_network.rs` in
//! `turbospark-repack`, which streams `mlx-community/Qwen3.8-27B-4bit` and
//! drops the tokenizer sidecars in beside it.
//!
//! **THE FIRST MEMORY ROW THIS FAMILY HAS EVER HAD.** `qwen35` shipped with
//! Bonsai-27B and got neither an oracle nor a quality gate, so the DENSE half
//! of `families/qwen/` had no sentinel of any kind -- a regression in it was
//! visible only through the MoE half's rows, which do not exercise
//! `dense.rs` at all.
//!
//! **IT ALSO DISCHARGES A WRITTEN-DOWN UNKNOWN.** `real_model::
//! protocol_parameters` put `QwenGdnDense` in the shared-window group on the
//! TOKENIZER's evidence (`vocab_size: 248320`, Qwen 3.6's exactly, under the
//! same ChatML dialect) and said so in a comment ending "UNVERIFIED until an
//! install exists: the first memory-oracle run is what confirms it, and a
//! `long-synthesis` that stops on maxTokens is what would refute it". It is
//! confirmed: `long-synthesis` tokenizes to 2,940 and generates 637 more, so
//! 3,577 of 4,096 are used and all three cases stop `endOfTurn`. This target
//! is where that stays confirmed.

mod oracle_common;

/// Per-chip rows for `Qwen/Qwen3.8-27B` at INT4, MOST SPECIFIC SUBSTRING
/// FIRST (`memory_oracle.rs` explains the lookup order).
///
/// No Swift row at any chip and there will not be one: the Swift original
/// has no `qwen3_5` support at all, so every row here is this port measuring
/// itself.
const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen protocol, release, on AC, 16 expert-cache slots (inert here --
    // a dense install has no routed experts to cache), 4,096 context, first
    // session after the real checkpoint was streamed (2026-08-14). THREE
    // readings, two from `turbospark-bench --model` and one from this
    // oracle:
    //   peak footprint     660.3 / 660.0 / 660.2 MiB
    //   short-explanation  18.169 / 19.007 / 19.041 tok/s
    //   medium-review      17.973 / 18.606 / 18.694 tok/s
    //   long-synthesis     16.882 /    --   / 16.764 tok/s
    // All three cases stopped endOfTurn every time; the oracle's replay of
    // short-explanation grew +0.02 MiB.
    //
    // The peak spans 0.3 MiB across three readings and the slowest case
    // spans 0.7%, which is far tighter than the MoE families manage -- there
    // is no expert-slot warming here to spread either one.
    //
    // **THE 15.1 GB OF RESIDENT WEIGHTS ARE ABSENT FROM THIS NUMBER**, which
    // is the one thing to understand before reusing it. AGENTS.md Gotcha 40
    // recorded that for a dense GGUF install (Mistral 7B: 4.07 GiB of
    // weights against a 684 MiB peak) and explicitly said to re-derive it
    // per install shape rather than quote it. Re-derived here on a DENSE
    // SAFETENSORS install four times the size, and it holds: the resident
    // region is 15,132,916,736 bytes and the peak is 660.3 MiB.
    //
    // The accounting closes on the terms that are left, computed from shapes
    // before the run rather than fitted to it:
    //   KV    16 full layers x 4 kv heads x 256 head_dim x 2 x 2 B
    //         = 64 KiB/token x 4,096                        = 256.0 MiB
    //   GDN   48 linear layers x 48 v heads x 128 x 128 x 4 B = 144.0 MiB
    //         (delta-rule S; fixed, does NOT grow with context)
    //   conv  48 x 10,240 x 4 taps x 4 B                    =   7.5 MiB
    //   ---------------------------------------------------------------
    //   sum                                                   407.5 MiB
    // leaving ~253 MiB of process baseline and host scratch, which at a
    // 248,320-wide vocab is unremarkable. Predicted ~585 before the run
    // against 660 measured, i.e. the model is right and the remainder is
    // the part it does not try to predict.
    //
    // So this ceiling is NOT comparable to the MoE rows: their dominant term
    // (`slots x layers x expert_stride`) is exactly zero here, and the term
    // that dominates here is KV, which is a pure function of the
    // architecture and the window.
    //
    // Ceiling 750 is the measured peak + ~13%: loose enough that allocator
    // jitter cannot flake it (the two readings agree to 0.3 MiB, so there is
    // little to absorb), tight enough that a doubling cannot hide. There is
    // no expert-slot warming here to widen the spread, which is why the two
    // peaks agree as closely as they do.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 750,
        // 0.73 of the SLOWER of the two readings of the slowest case
        // (16.764), the same margin the mistral and qwen3moe rows take and
        // the same number those two land on. Crate Gotcha 15's rule
        // satisfied: two readings, and the floor comes off the slower.
        tok_s_floor: 12.0,
        source: "this port, 2026-08-14, Apple M4 Max, AC, 4096 context",
    },
];

/// Ceiling for an unlisted chip. Memory sizing does not depend on the chip,
/// and on this family it is KV plus a fixed recurrent state, both pure
/// functions of the architecture and the window.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 750;

#[test]
#[ignore = "needs a real Qwen3.8-27B install via TURBOSPARK_QWEN38_INSTALL_DIR"]
fn real_qwen38_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_QWEN38_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_QWEN38_INSTALL_DIR is not set");
        return;
    };
    oracle_common::run_oracle(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
    );
}
