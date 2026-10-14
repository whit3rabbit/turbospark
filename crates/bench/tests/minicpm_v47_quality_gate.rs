#![cfg(target_os = "macos")]
//! The quality gate for `mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4`'s text
//! trunk, against the REAL `.gturbo` install. Same body and measurements as
//! `quality_gate.rs` (see `quality_common`); only the install and rows differ.
//!
//! A SEPARATE TARGET, one real model per process.
//!
//!   TURBOSPARK_MINICPM_V47_INSTALL_DIR=~/.turbospark/models/text/minicpm-v47-35b-mxfp4.gturbo \
//!     cargo test -p turbospark-bench --test minicpm_v47_quality_gate --release -- --ignored --nocapture

mod quality_common;

const BASELINES: &[quality_common::ChipQuality] = &[];

#[test]
#[ignore = "needs a real ~21 GB MiniCPM-V 4.7 mxfp4 install (TURBOSPARK_MINICPM_V47_INSTALL_DIR)"]
fn real_minicpm_v47_install_quality_holds() {
    let Some(dir) = std::env::var_os("TURBOSPARK_MINICPM_V47_INSTALL_DIR") else {
        eprintln!("minicpm_v47_quality_gate: TURBOSPARK_MINICPM_V47_INSTALL_DIR is not set; skipping");
        return;
    };
    quality_common::run_quality_gate(std::path::Path::new(&dir), BASELINES);
}
