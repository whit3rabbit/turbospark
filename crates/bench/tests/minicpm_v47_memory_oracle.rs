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

const BASELINES: &[oracle_common::ChipBaseline] = &[];

/// Ceiling for an unlisted chip: the accounting in the row's comment, which
/// no chip changes.
const UNKNOWN_CHIP_CEILING_MIB: u64 = 7000;

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
