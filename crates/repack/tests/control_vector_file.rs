//! Reads a control vector off DISK and reports what it holds.
//!
//! `crates/repack/src/control_vector.rs`'s unit tests round-trip the reader
//! against the writer beside it, which cannot say whether either matches the
//! format anyone else writes -- a reader and a writer by one author agree
//! whenever they share a mistake (AGENTS.md Gotcha 48, the same argument
//! `mtp_install_fidelity_network.rs` makes about the MTP head).
//!
//! This one takes a file produced by something ELSE. Point it at
//! `scripts/extract_direction.py`'s output, or at a published `repeng` /
//! llama.cpp vector, and it either parses or names what it could not read.
//!
//! ```sh
//! TURBOSPARK_CONTROL_VECTOR=/tmp/steer/d.gguf \
//!   cargo test -p turbospark-repack --test control_vector_file -- --ignored --nocapture
//! ```

use turbospark_repack::control_vector::load_control_vector;

#[test]
#[ignore = "needs a control vector on disk (TURBOSPARK_CONTROL_VECTOR)"]
fn a_control_vector_on_disk_parses_and_reports_its_shape() {
    let Some(path) = std::env::var_os("TURBOSPARK_CONTROL_VECTOR") else {
        eprintln!("TURBOSPARK_CONTROL_VECTOR unset; skipping");
        return;
    };
    let path = std::path::PathBuf::from(path);
    let set = match load_control_vector(&path) {
        Ok(s) => s,
        Err(e) => panic!("{}: {e}", path.display()),
    };

    println!("control vector: {}", path.display());
    println!("  architecture : {:?}", set.declared_arch);
    println!("  declared mode: {:?}", set.declared_mode);
    println!("  hidden       : {}", set.hidden);
    println!(
        "  layers       : {} covered of {} spanned",
        set.covered_layers(),
        set.layers.len()
    );

    assert!(set.covered_layers() > 0, "no layer carries a direction");

    // Report the per-layer norm rather than asserting a floor on it. A
    // near-zero layer is a real outcome (the extraction found no separation
    // there) and thresholding it here would turn a finding into a failure.
    // What IS asserted is finiteness: a non-finite direction multiplies the
    // residual stream and arrives downstream as NaN, which reads as a PERFECT
    // score on any rank instrument (AGENTS.md Gotcha 59).
    let mut worst = f32::INFINITY;
    let mut best: f32 = 0.0;
    for (l, dir) in set.layers.iter().enumerate() {
        let Some(d) = dir else { continue };
        assert!(
            d.values.iter().all(|v| v.is_finite()),
            "layer {l} carries a non-finite direction"
        );
        assert!(
            d.inv_norm.is_finite(),
            "layer {l} has a non-finite inv_norm"
        );
        let norm = if d.inv_norm > 0.0 {
            1.0 / d.inv_norm
        } else {
            0.0
        };
        worst = worst.min(norm);
        best = best.max(norm);
    }
    println!("  norm range   : {worst:.4} to {best:.4}");
    println!(
        "  NOTE the raw norm rises with the residual stream's own scale and is NOT\n\
         \x20      comparable across layers; extract_direction.py's `sep` column is."
    );
}
