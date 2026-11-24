#![cfg(target_os = "macos")]
//! The memory oracle for gemma-2-2b-it (Gemma flow, attention-score soft cap),
//! against a REAL `.gturbo` install streamed from
//! `bartowski/gemma-2-2b-it-GGUF` Q4_K_M. Same body and assertions as its
//! siblings (see `oracle_common`).
//!
//! Not run by default (needs the 1.6 GiB install; use --release):
//!
//!   TURBOSPARK_GEMMA2_INSTALL_DIR=~/.turbospark/models/text/gemma2-2b-q4km.gturbo \
//!     cargo test -p turbospark-bench --test gemma2_memory_oracle --release -- --ignored --nocapture
//!
//! All three protocol cases end on `endOfTurn` at the shared 4,096 window
//! (long-synthesis is 2,998 tokens under this 256k vocabulary). There is no
//! catalog row for this family yet (the checkpoint is under the Gemma Terms of
//! Use), so unlike the siblings there is no `measured` block to agree with and
//! no offline agreement test: the numbers live here only.

mod oracle_common;

/// Apple M4 Max 36GB, release, AC power, 4,096 context, 16 expert-cache slots
/// (inert: dense), two back-to-back readings on 2026-11-20 (the machine was NOT
/// idle: other processes held the load average near 5 to 6):
///   peak footprint     630.7 / 631.0 MiB
///   short-explanation  57.869 / 57.774 tok/s
///   medium-review      56.055 / 55.728 tok/s
///   long-synthesis     45.897 / 45.802 tok/s
/// All three stopped endOfTurn both times; replay grew +0.00 MiB both times.
///
/// Ceiling 680 is the higher peak + ~8%. Floor 33.4 is 0.73 of the slower
/// reading of the slowest case. The decode kernel here is the BARRIER one, which
/// the attention-score cap forces (`AttentionScratch::with_softcap`).
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 680,
    tok_s_floor: 33.4,
    source: "this port, 2026-11-20, Apple M4 Max, AC, 4096 context",
}];

const UNKNOWN_CHIP_CEILING_MIB: u64 = 680;

#[test]
#[ignore = "needs a real Gemma 2 install via TURBOSPARK_GEMMA2_INSTALL_DIR"]
fn real_gemma2_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_GEMMA2_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_GEMMA2_INSTALL_DIR is not set");
        return;
    };
    oracle_common::run_oracle(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
    );
}
