#![cfg(target_os = "macos")]
//! The memory oracle for `deepseek2` (DeepSeek-V2-Lite-Chat Q8_0), against a
//! REAL `.gturbo` install. Same body, same frozen protocol and the same four
//! assertions as `memory_oracle.rs` (see `oracle_common`); only the install
//! and the baseline rows differ.
//!
//! A SEPARATE TARGET, not a second `#[test]`, for the same reason its
//! siblings are: the footprint assertion is a whole-session peak and the
//! families do not share a ceiling.
//!
//! Not run by default (needs a ~17 GB install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_DSV2_INSTALL_DIR=~/.turbospark/models/dsv2lite-16b.gturbo \
//!     cargo test -p turbospark-bench --test dsv2_memory_oracle --release -- --ignored --nocapture
//!
//! **THE PROTOCOL ROW RIDES THE COMPRESSED KV CACHE** (`real_model_params`'
//! `Deepseek2` arm): 1,152 bytes per token per layer over 27 layers is
//! 243 MiB at the 8,192 window (`docs/DEEPSEEK2_PHASE0.md`), so this family
//! takes the dense-llama window for a quarter of the KV a full-cache family
//! would pay, while the budget stays at the shared 1,024 (V2-Lite-Chat
//! answers directly, no reasoning channel). The `const` block below fails
//! the BUILD if the resolver and this row drift apart.

mod oracle_common;

/// Per-chip rows for DeepSeek-V2-Lite-Chat Q8_0, MOST SPECIFIC SUBSTRING
/// FIRST (`memory_oracle.rs` explains the lookup order).
///
/// No Swift row at any chip and there will not be one: the Swift original
/// has no `deepseek2` support at all, so every row here is this port
/// measuring itself.
///
/// Frozen from the first real run (2026-09-17), release, on AC, 16
/// expert-cache slots. A SECOND session the same day, asserting the frozen
/// row, stayed green: peak 4,076.7 MiB (within 0.7% of the first), replay
/// +0.23 MiB, tok/s 20.342 / 15.552 / 6.755 -- every case 13-18% faster
/// than the first session on identical work, the ordinary shared-machine
/// spread, with the floor margin intact. First session:
///   peak footprint     4,103.7 MiB
///   short-explanation  17.533 tok/s
///   medium-review      11.285 tok/s
///   long-synthesis      5.722 tok/s
/// All three cases stopped endOfTurn in both sessions.
///
/// **THE PEAK IS SLOT CACHE + KV PLUS ~33 MiB, AND THE RESIDENT CORE IS
/// NOT IN IT.** The arithmetic: 16 slots x 26 MoE layers x 9.19 MB is
/// 3,828 MiB of slot capacity, KV at the 8,192 window is 243 MiB
/// (`docs/DEEPSEEK2_PHASE0.md`), and the measured peak sits ~33 MiB over
/// their sum -- while the ~1.33 GiB resident core (attention, shared
/// experts, embeddings, head) contributes nothing measurable. That is the
/// dense-row phenomenon of AGENTS.md Gotcha 40 reaching an MoE install:
/// the mapped weights this flow never explicitly allocates are not
/// `phys_footprint`-counted, and the counted figure is a leak sentinel
/// (the slot cache and the KV are what it guards), not a capacity number.
/// The practical requirement is closer to the install's ~17 GB on disk.
///
/// Ceiling 4,300 is measured peak + ~5%, the qwen3moe row's margin. Floor
/// 4.0 is 0.70x the slowest case (long-synthesis at 5.722, which decodes
/// at 3,754-token context after a 375 s sequential per-token prefill --
/// the descope list's unbuilt T-row prefill kernel is what makes that
/// case the slow one). The second session reproduced every case 13-18%
/// faster on identical work, so a first floor this loose is deliberate;
/// it can tighten once a reading exists on a quiet-dedicated machine.
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 4300,
    tok_s_floor: 4.0,
    source: "this port, 2026-09-17, Apple M4 Max, AC, 8192 context, 1024 budget, 16 slots",
}];

/// Ceiling for an unlisted chip. Memory sizing does not depend on the chip:
/// on this family it is slot cache plus KV, both pure functions of the
/// architecture, the window, and the slot count.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 4300;

/// The oracle's own copy of the two protocol parameters, asserted equal to
/// the resolver's in a `const` block so the binary and this row cannot drift
/// apart (crate Gotcha 16).
const DSV2_MAX_CONTEXT: u32 = 8192;
const DSV2_MAX_NEW: u32 = 1024;

#[test]
#[ignore = "needs a real deepseek2 .gturbo install (TURBOSPARK_DSV2_INSTALL_DIR)"]
fn real_deepseek2_install_peak_footprint_and_throughput_hold() {
    const _: () = {
        let resolved =
            turbospark_bench::real_model::protocol_parameters(model_io::ModelFamily::Deepseek2);
        assert!(
            resolved.max_context == DSV2_MAX_CONTEXT && resolved.max_new == DSV2_MAX_NEW,
            "this oracle's window/budget must equal protocol_parameters'"
        );
    };
    let Some(dir) = std::env::var_os("TURBOSPARK_DSV2_INSTALL_DIR") else {
        eprintln!(
            "dsv2_memory_oracle: TURBOSPARK_DSV2_INSTALL_DIR is not set; skipping. \
             Point it at a streamed DeepSeek-V2-Lite-Chat .gturbo install to run the oracle."
        );
        return;
    };
    oracle_common::run_oracle_with_budget(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
        DSV2_MAX_CONTEXT,
        DSV2_MAX_NEW,
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
        "dsv2lite-16b",
        BASELINES,
        DSV2_MAX_CONTEXT,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
