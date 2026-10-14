#![cfg(target_os = "macos")]
//! The memory oracle for `mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4`'s text
//! trunk (Qwen 3.6 35B-A3B, MLX mxfp4), against the REAL `.gturbo` install.
//!
//! A SEPARATE TARGET, not a second `#[test]`: the footprint assertion is a
//! whole-session peak, so one real model per process.
//!
//!   TURBOSPARK_MINICPM_V47_INSTALL_DIR=~/.turbospark/models/text/minicpm-v47-35b-mxfp4.gturbo \
//!     cargo test -p turbospark-bench --test minicpm_v47_memory_oracle --release -- --ignored --nocapture

mod oracle_common;

const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen 2026-10-13 on AC, release, 16 expert-cache slots, 4,096 context.
    // Two readings: 1,491 and 1,481 MiB peak.
    //
    // The accounting, from shapes before the run:
    //   slots  16 x 40 layers x 1,671,168 B expert stride  = 1,020.0 MiB
    //          (GGUF MXFP4 blocks re-blocked from MLX mxfp4: 102 pages)
    //   KV     10 full layers x 2 kv heads x 256 x 2 x 2 B
    //          = 20 KiB/token x 4,096                      =    80.0 MiB
    //   GDN    30 linear layers x 32 v heads x 128 x 128 x 4 B = 60.0 MiB
    //   conv   30 x 8,192 x 4 taps x 4 B                   =     3.8 MiB
    //   -----------------------------------------------------------------
    //   sum                                                  1,163.8 MiB
    // against 1,491 measured, leaving ~327 MiB of process baseline and host
    // scratch.
    //
    // **THE BF16 RESIDENT CORE IS NOT IN IT.** This checkpoint keeps embed,
    // head and every GDN projection BF16, and repack dequantizes the other
    // mxfp4 matrices to BF16, so the resident file is ~4.6 GiB, three times
    // the INT4 Qwen 3.6 install's. None of it reaches `phys_footprint`
    // (AGENTS.md Gotcha 40's measurement, again), which is why this peak sits
    // below Qwen 3.6's 1,610. The resident size shows up as decode BANDWIDTH
    // instead: 24-26 tok/s here against Qwen 3.6's 33-38.
    //
    // Ceiling 1,650 is the higher peak + ~11%, the margin the ornith35b row
    // takes for the same slot-dominated shape.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 1650,
        // 0.73 of the slowest case on the quieter second reading
        // (short-explanation and long-synthesis, 24.04), the margin the
        // other MoE rows take. The first reading (19.4-26.0) ran with another
        // session's audio tests on the host and is not the calibration.
        tok_s_floor: 17.5,
        source: "this port, 2026-10-13, Apple M4 Max, AC, 4096 context, 16 slots",
    },
];

/// Ceiling for an unlisted chip. The terms above are functions of the
/// architecture, the window and the slot count, none of which the chip sets.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 1650;

#[test]
#[ignore = "needs a real MiniCPM-V 4.7 mxfp4 install via TURBOSPARK_MINICPM_V47_INSTALL_DIR"]
fn real_minicpm_v47_install_peak_footprint_and_throughput_hold() {
    let Some(dir) = std::env::var_os("TURBOSPARK_MINICPM_V47_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_MINICPM_V47_INSTALL_DIR is not set");
        return;
    };
    oracle_common::run_oracle(
        std::path::Path::new(&dir),
        BASELINES,
        UNKNOWN_CHIP_CEILING_MIB,
    );
}

/// The catalog half of this row, checked offline on every `cargo test`.
#[test]
fn the_baselines_agree_with_the_catalogs_measured_rows() {
    oracle_common::assert_agrees_with_catalog(
        "minicpm-v47-35b-mxfp4",
        BASELINES,
        turbospark_bench::protocol::PROTOCOL_MAX_CONTEXT,
        turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS as u32,
    );
}
