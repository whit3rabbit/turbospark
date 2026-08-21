#![cfg(target_os = "macos")]
//! The memory oracle for `ornith-ai/Ornith-1.5-35B-A3B`, against the REAL
//! INT4-affine `.gturbo` install.
//!
//! A SEPARATE TARGET, not a second `#[test]`: the footprint assertion is a
//! whole-session peak, so one real model per process.
//!
//!   TURBOSPARK_ORNITH35B_INSTALL_DIR=~/models/ornith35b.gturbo \
//!     cargo test -p turbospark-bench --test ornith35b_memory_oracle --release -- --ignored --nocapture
//!
//! **THE INT4 INSTALL AND NOT THE Q8_0 ONE.** Both exist on the development
//! machine and they are the same model; the INT4 one is 1.63-1.67x faster at
//! half the expert stride and is what anything future builds on. Freezing a
//! second row for the Q8_0 install would double this gate's ten minutes to
//! sentinel a path nothing depends on. What that install is FOR is the
//! quantization A/B in `docs/BENCHMARKS.md`, which is throughput and needs no
//! frozen peak.

mod oracle_common;

const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen 2026-08-20 on AC, release, 16 expert-cache slots, 4,096 context.
    //
    // The accounting closes on terms computed from the header BEFORE the run
    // (Gotcha 36's multiplication, which is the one that decides whether a
    // checkpoint can stream at all):
    //   slots  16 x 40 layers x 1,769,472 B expert stride  = 1,080.0 MiB
    //   KV     10 full layers x 2 kv heads x 256 head_dim
    //          x 2 x 2 B = 20 KiB/token x 4,096            =    80.0 MiB
    //   GDN    30 linear layers x 32 v heads x 128 x 128 x 4 B = 180.0 MiB
    //          (delta-rule S; fixed, does NOT grow with context)
    //   conv   30 x 8,192 x 4 taps x 4 B                   =     3.8 MiB
    //   -----------------------------------------------------------------
    //   sum                                                  1,343.8 MiB
    // against 1,572.3 measured, leaving ~229 MiB of process baseline and host
    // scratch at a 248,320-wide vocab.
    //
    // **THE SLOT CACHE IS 69% OF IT**, which is the reading to carry: this
    // row is dominated by `slots x layers x expert_stride` and therefore
    // moves with the SLOT COUNT as well as the window (AGENTS.md Gotcha 58).
    // At the `auto` this machine resolves (32) it would be ~2.6 GiB rather
    // than 1.57, so a peak taken here is only comparable to another taken at
    // 16. The harness pins 16; `--expert-cache-slots auto` does not.
    //
    // Worth one comparison, because it is the same model twice: the Q8_0
    // GGUF install of this checkpoint has a 3,342,336-byte stride, so its
    // slot cache alone is 2,040 MiB and its whole peak lands above this
    // one's -- the INT4 conversion is smaller on the axis this row measures
    // as well as on disk.
    //
    // Ceiling 1,750 is the measured peak + ~11%. The three cases' peaks span
    // 1.7 MiB (1570.6 / 1570.9 / 1572.3), far tighter than the 77 MiB
    // expert-slot-warming band the gemma4 row documents, so there is little
    // for the margin to absorb; it is set to survive allocator jitter and
    // still catch a doubling.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 1750,
        // 0.73 of the slowest case's reading (long-synthesis, 32.175), the
        // same margin the mistral, qwen3moe and qwen38 rows take.
        tok_s_floor: 23.5,
        source: "this port, 2026-08-20, Apple M4 Max, AC, 4096 context, 16 slots",
    },
];

/// Ceiling for an unlisted chip. Memory sizing does not depend on the chip:
/// every term above is a function of the architecture, the window and the
/// slot count, none of which the chip decides.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 1750;

#[test]
#[ignore = "needs a real Ornith-1.5-35B INT4 install via TURBOSPARK_ORNITH35B_INSTALL_DIR"]
fn real_ornith35b_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_ORNITH35B_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_ORNITH35B_INSTALL_DIR is not set");
        return;
    };
    oracle_common::run_oracle(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
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
        "ornith35b",
        BASELINES,
        turbospark_bench::protocol::PROTOCOL_MAX_CONTEXT,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
