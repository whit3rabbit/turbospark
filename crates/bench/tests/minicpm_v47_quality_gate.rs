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

const BASELINES: &[quality_common::ChipQuality] = &[
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // Frozen 2026-10-13 on AC, release, 16 expert-cache slots. Three fresh
    // processes agreed on the perplexity and both digests, and the 8-slot
    // digest equals the 16-slot one (Gotcha 27: routed slots dispatch in
    // router rank).
    //
    // 7.5693 sits above Qwen 3.6's 6.2536 on the same corpus and framing.
    // That is not a ranking: this is a different fine-tune (the MiniCPM-V
    // trunk) at a different quantization (OCP FP4 rather than affine INT4),
    // and each row is only a sentinel against its own past. Nothing external
    // has checked this install's numerics (no cross-engine KL).
    //
    // The constrained arm's throughput ratio read 0.49x and 1.20x on a
    // loaded host and 0.68x on a quieter one; only the last is evidence.
    quality_common::ChipQuality {
        brand_substr: "Apple M4 Max",
        perplexity: 7.5693,
        greedy_digest: "cfa82442be9368dacce457197b7e6e5a4f4eeec14c63853bb44b9d09af140ac9",
        sampled_digest: "68850b0931646d77571c99148b99ba6d4975154b88105169a46b7148990832b9",
        source: "this port, 2026-10-13, Apple M4 Max, AC, 16 slots",
    },
];

#[test]
#[ignore = "needs a real ~21 GB MiniCPM-V 4.7 mxfp4 install (TURBOSPARK_MINICPM_V47_INSTALL_DIR)"]
fn real_minicpm_v47_install_quality_holds() {
    let Some(dir) = std::env::var_os("TURBOSPARK_MINICPM_V47_INSTALL_DIR") else {
        eprintln!(
            "minicpm_v47_quality_gate: TURBOSPARK_MINICPM_V47_INSTALL_DIR is not set; skipping"
        );
        return;
    };
    quality_common::run_quality_gate(std::path::Path::new(&dir), BASELINES);
}
