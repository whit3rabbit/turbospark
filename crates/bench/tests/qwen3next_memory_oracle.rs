#![cfg(target_os = "macos")]
//! The memory oracle for Qwen3-Next-80B-A3B (Qwen 3.6 flow, 512 experts top-10),
//! against a REAL `.gturbo` install streamed from
//! `unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF` UD-TQ1_0. Same body and assertions
//! as its siblings (see `oracle_common`).
//!
//! Not run by default (needs the 19 GiB install; use --release):
//!
//!   TURBOSPARK_QWEN3NEXT_INSTALL_DIR=~/.turbospark/models/text/qwen3next-tq1.gturbo \
//!     cargo test -p turbospark-bench --test qwen3next_memory_oracle --release -- --ignored --nocapture
//!
//! All three protocol cases end on `endOfTurn` at the shared 4,096 window. The
//! peak is small because a streamed MoE counts the resident core, KV and the
//! slot cache, not the 19 GiB of mapped weights (Gotcha 19): at 16 slots the
//! slot cache is about 16 x 48 x 600 KiB = 450 MiB.
//!
//! PREFILL IS SLOW and is recorded as a fact, not gated: `long-synthesis`
//! prefills 2,838 tokens in 166 / 160 s (about 17 tok/s), because this install
//! takes the per-token routed path (the batched routed prefill is gated on the
//! affine expert layout).

mod oracle_common;

use turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS;

/// Apple M4 Max 36GB, release, AC power, 4,096 context, 16 expert-cache slots,
/// two back-to-back readings on 2026-11-20 (the machine was NOT idle: other
/// processes held the load average near 5 to 6):
///   peak footprint     946.6 / 947.2 MiB
///   short-explanation  20.189 / 20.210 tok/s
///   medium-review      20.807 / 18.578 tok/s
///   long-synthesis     18.566 / 18.117 tok/s
/// All three stopped endOfTurn both times; replay +0.33 / +0.00 MiB.
///
/// medium-review moved 11% between the two readings, which is why the floor is
/// taken from the slowest case's slower reading (18.117) and not from a single
/// run: 0.73 of that is 13.2. Ceiling 1,020 is the higher peak + ~8%.
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 1020,
    tok_s_floor: 13.2,
    source: "this port, 2026-11-20, Apple M4 Max, AC, 4096 context, 16 slots",
}];

const UNKNOWN_CHIP_CEILING_MIB: u64 = 1020;

#[test]
#[ignore = "needs a real Qwen3-Next install via TURBOSPARK_QWEN3NEXT_INSTALL_DIR"]
fn real_qwen3next_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_QWEN3NEXT_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_QWEN3NEXT_INSTALL_DIR is not set");
        return;
    };
    oracle_common::run_oracle(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
    );
}

/// The catalog half, offline on every `cargo test`.
#[test]
fn the_baselines_agree_with_the_catalogs_measured_rows() {
    oracle_common::assert_agrees_with_catalog(
        "qwen3next-tq1",
        BASELINES,
        turbospark_bench::protocol::PROTOCOL_MAX_CONTEXT,
        PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
