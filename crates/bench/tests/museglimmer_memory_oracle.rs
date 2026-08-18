#![cfg(target_os = "macos")]
//! The memory oracle for `mlx-community/Muse-Glimmer-30B-4bit`, against a REAL
//! `museGlimmer` `.gturbo` install.
//!
//! A SEPARATE TARGET, not a second `#[test]`, for the same reason the others
//! are: the footprint assertion is a whole-session peak, so two models in one
//! process measure against each other's high-water mark.
//!
//! Not run by default (needs a ~15 GB install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_MUSEGLIMMER_INSTALL_DIR=~/models/museglimmer-30b.gturbo \
//!     cargo test -p turbospark-bench --test museglimmer_memory_oracle --release -- --ignored --nocapture
//!
//! Build the install with `tests/museglimmer_checkpoint_network.rs` in
//! `turbospark-repack`, which streams the checkpoint and drops the tokenizer
//! sidecars in beside it.
//!
//! **THE SECOND FAMILY TO MOVE BOTH PROTOCOL PARAMETERS, and it moves them
//! for gpt-oss's reason** (crate Gotcha 12): this model REASONS BEFORE
//! ANSWERING. Its template writes `Reasoning strength: high.` into the system
//! preamble and the model emits a `to=self` message before its `to=user` one,
//! so at the shared 1,024 budget even the SHORT case stops on `maxTokens` and
//! the validity gate refuses it. Measured greedy on the real install
//! (2026-08-15), all three stopping endOfTurn at 8,192 / 3,072:
//!   short-explanation   102 prompt + 1,054 new
//!   medium-review       464 prompt + 1,378 new
//!   long-synthesis    2,820 prompt + 1,246 new
//! Hence 2,048 (1.49x the largest completion) and 8,192 (2,820 + 2,048 does
//! not fit 4,096).

mod oracle_common;

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST (`memory_oracle.rs` explains
/// the lookup order).
///
/// No Swift row at any chip and there will not be one: the Swift original has
/// no `muse_glimmer` support at all, so every row here is this port measuring
/// itself.
const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen protocol, release, on AC, 16 expert-cache slots (inert here --
    // a dense install has no routed experts to cache), 8,192 context, 2,048
    // budget, first session after the checkpoint was streamed (2026-08-15):
    //   peak footprint     535 MiB (535.7 / 535.9 / 535.9 across the run)
    //   short-explanation  13.291 tok/s, 1,132 new tokens
    //   medium-review      15.341 tok/s, 1,498 new
    //   long-synthesis     14.483 tok/s, 1,552 new
    // All three stopped endOfTurn; the replay of short-explanation grew
    // +1.19 MiB.
    //
    // **THE 15.7 GB OF RESIDENT WEIGHTS ARE ABSENT FROM THIS NUMBER**, the
    // THIRD re-derivation of AGENTS.md Gotcha 40 (Mistral 7B: 4.07 GiB of
    // weights, 684 MiB peak; Qwen3.8-27B: 15.1 GB, 660 MiB). That gotcha says
    // to re-derive per install shape rather than quote it, and it holds again
    // on a dense safetensors install with a windowed KV cache.
    //
    // The accounting, computed from shapes BEFORE the run rather than fitted
    // to it:
    //   KV full   13 layers x 2 kv x 128 x 2 x 2 B x 8,192   = 104.0 MiB
    //   KV swa    39 layers x 2 kv x 128 x 2 x 2 B x 2,176   =  82.9 MiB
    //             (ring = sliding_window 2,048 + 128 chunk headroom)
    //   ------------------------------------------------------------------
    //   sum                                                    186.9 MiB
    // leaving ~349 MiB of process baseline and host scratch, which at a
    // 202,048-wide vocab is unremarkable and is the same residual shape
    // Qwen3.8 showed (407.5 predicted, 660 measured, ~253 left over).
    //
    // Note the KV term is SMALLER than Qwen3.8's 256 MiB despite TWICE the
    // window: this family is 2 kv heads at 128 against 4 at 256, and three
    // quarters of its layers ring at 2,176 rather than running the full
    // window. The window alone does not tell you the KV bill.
    //
    // Ceiling 650 is the measured peak + ~21%. The three readings within the
    // run agree to 0.2 MiB, so there is little jitter to absorb; the margin
    // is for allocator variation across sessions rather than for spread.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 650,
        // 0.73 of the SLOWEST case (13.291), the margin the mistral,
        // qwen3moe and qwen38 rows all take.
        //
        // **FROM ONE ORACLE READING, which is weaker than crate Gotcha 15
        // asks for and is stated rather than hidden.** That gotcha wants two
        // readings with the floor off the slower, because a dense install's
        // memory number reproduces far better than its throughput does
        // (Mistral's peak agreed to 1.6 MiB while its slowest case moved
        // 12.5%). The greedy budget probe run minutes earlier read
        // 13.907 / 14.077 / 12.303 tok/s on the same three cases, which is
        // the same neighbourhood from a DIFFERENT sampling configuration and
        // so is corroboration rather than a second reading. Re-run this
        // target and tighten if it holds.
        tok_s_floor: 9.0,
        source: "this port, 2026-08-15, Apple M4 Max, AC, 8192 context, 2048 budget",
    },
];

/// Ceiling for an unlisted chip. Memory sizing does not depend on the chip:
/// on this family it is KV plus a process baseline, both pure functions of
/// the architecture and the window.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 650;

/// The oracle's own copy of the two protocol parameters, asserted equal to
/// the resolver's in a `const` block so the binary and this row cannot drift
/// apart (crate Gotcha 16). A BUILD failure, not a runtime one in a target
/// that almost never runs.
const MUSE_MAX_CONTEXT: u32 = 8192;
const MUSE_MAX_NEW: u32 = 2048;

#[test]
#[ignore = "needs a real Muse-Glimmer-30B install via TURBOSPARK_MUSEGLIMMER_INSTALL_DIR"]
fn real_muse_glimmer_install_peak_footprint_and_throughput_hold() {
    const _: () = {
        let resolved =
            turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::MuseGlimmer);
        assert!(
            resolved.max_context == MUSE_MAX_CONTEXT && resolved.max_new == MUSE_MAX_NEW,
            "this oracle's window/budget must equal protocol_parameters'"
        );
    };
    let Some(dir) = std::env::var_os("TURBOSPARK_MUSEGLIMMER_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_MUSEGLIMMER_INSTALL_DIR is not set");
        return;
    };
    oracle_common::run_oracle_with_budget(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
        MUSE_MAX_CONTEXT,
        MUSE_MAX_NEW,
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
        "museglimmer",
        BASELINES,
        MUSE_MAX_CONTEXT,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
