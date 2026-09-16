#![cfg(target_os = "macos")]
//! The memory oracle for `prism-ml/Bonsai-27B-mlx-1bit`, against a REAL
//! `qwen35` `.gturbo` install. Same body, same frozen protocol and the same
//! assertions as its siblings (see `oracle_common`); the install and the
//! baseline rows differ, and the WINDOW does not -- this family runs the
//! shared 4,096 / 1,024.
//!
//! A SEPARATE TARGET, not a second `#[test]`, for the reason the others
//! state: the footprint assertion is a whole-session peak, so two models in
//! one process measure against each other's high-water mark.
//!
//! Not run by default (needs a ~5 GB install; use --release or the tok/s
//! numbers are meaningless):
//!
//!   TURBOSPARK_QWEN35_INSTALL_DIR=~/models/bonsai27b.gturbo \
//!     cargo test -p turbospark-bench --test bonsai_memory_oracle --release -- --ignored --nocapture
//!
//! Build the install with `tests/qwen35_checkpoint_network.rs` in
//! `turbospark-repack`, which streams `prism-ml/Bonsai-27B-mlx-1bit` and
//! drops the tokenizer sidecars in beside it.
//!
//! **THE FIRST MEMORY ROW THIS CHECKPOINT HAS HAD, and the dense half's
//! second.** `qwen35` shipped with Bonsai-27B and got neither an oracle nor
//! a quality gate (`qwen38_memory_oracle.rs`'s header records the gap);
//! ternary's rows arrived with its own install. This file closes the
//! original hole on the checkpoint that made it.
//!
//! The env var is the STREAM test's own (`TURBOSPARK_QWEN35_INSTALL_DIR`),
//! which is also what `logit_dump.rs`'s qwen35 arm and the cross-engine
//! driver read: one spelling for one install across every instrument.

mod oracle_common;

/// Per-chip rows for Bonsai-27B at INT1 group 128, MOST SPECIFIC SUBSTRING
/// FIRST. Frozen 2026-09-16 from the second run on this machine (release,
/// 4,096 context): peak 660.4 MiB -- against ternary's 657.8-661.6 on the
/// SAME architecture at the same window, which is the prediction the
/// placeholder ceiling was built on, confirmed to 4 MiB. Decode reads
/// 18.9 / 18.5 / 16.7 tok/s across the three cases; the floor is 0.73x the
/// slowest, the margin the other rows take.
///
/// **THE FIRST RUN'S NUMBERS WERE DISCARDED, AND THE REASON IS A CLASS.**
/// It ran fifth of six back-to-back GPU gates on a heat-soaked machine and
/// read medium/long prefill at 12x the cool machine's time (1.5 tok/s
/// decode against 18.5). The same night, ornith9b's oracle missed its
/// frozen tok/s floor by 17% on the same pattern and passed on the cooled
/// re-run. Numerics (perplexity, digests) are thermal-invariant and froze
/// fine from the hot runs; THROUGHPUT is not -- the interleave/cooldown
/// discipline of Gotchas 22 and 28 applies to oracle floors too.
const BASELINES: &[oracle_common::ChipBaseline] = &[oracle_common::ChipBaseline {
    brand_substr: "Apple M4 Max",
    footprint_ceiling_mib: 715,
    tok_s_floor: 12.2,
    source: "this port, 2026-09-16, Apple M4 Max, 4096 context",
}];

/// Superseded by the frozen row above; kept so the freeze history reads in
/// one file. It was ternary's neighbourhood on purpose: same architecture,
/// same window, and per Gotcha 40 the resident weights are absent from
/// `phys_footprint` -- confirmed to 4 MiB.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 700;

#[test]
#[ignore = "needs a real Bonsai-27B .gturbo install via TURBOSPARK_QWEN35_INSTALL_DIR"]
fn real_bonsai_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_QWEN35_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_QWEN35_INSTALL_DIR is not set");
        return;
    };
    oracle_common::run_oracle(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
    );
}
