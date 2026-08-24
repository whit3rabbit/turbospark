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

//! A SECOND target below takes a file from the wider ecosystem specifically
//! and asserts where its directions land, which is the question this port got
//! wrong until 2026-08-24.
//!
//! ```sh
//! TURBOSPARK_FOREIGN_CONTROL_VECTOR=/tmp/steer-interop/pub/....gguf \
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

/// A vector written by something that is not this port, read for WHERE its
/// directions land rather than merely whether it parses.
///
/// This is the interop question `docs/OBLITERATION.md` carried as open for
/// five sessions, and the answer it was carrying was wrong: `direction.N`
/// names llama.cpp's block N, and this port read it as block `N - 1`.
///
/// The load-bearing assertion is that BLOCK 0 IS UNCOVERED. A foreign vector
/// skips it -- llama.cpp refuses `direction.0` by name and its apply loop
/// starts at 1, so `repeng`'s exporter never writes one -- and under the old
/// mapping that same file covered block 0 and left the model's LAST block
/// unsteered. Both readings parse, both steer, and only one is right.
///
/// Measured 2026-08-24 against
/// `jukofyork/creative-writing-control-vectors-v3.0`,
/// `Meta-Llama-3-8B-Instruct/llama-3:8b-optimism_vs_nihilism__optimism.gguf`:
/// hidden 4096, **31 covered of 32 spanned**, which is exactly Llama-3-8B's
/// 32 blocks minus the one llama.cpp cannot reach. The old reader made the
/// same bytes 31 of 31.
#[test]
#[ignore = "needs a published control vector (TURBOSPARK_FOREIGN_CONTROL_VECTOR)"]
fn a_foreign_control_vector_lands_on_llama_cpps_own_apply_range() {
    let Some(path) = std::env::var_os("TURBOSPARK_FOREIGN_CONTROL_VECTOR") else {
        eprintln!("TURBOSPARK_FOREIGN_CONTROL_VECTOR unset; skipping");
        return;
    };
    let path = std::path::PathBuf::from(path);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));

    // The fixture has to BE foreign, or this proves nothing: a file this port
    // wrote before the correction carries `turbospark.layer_base = 0`, is read
    // under the old convention on purpose, and would fail the block-0 check
    // below for a reason that is not a regression. Checked first so the
    // message says which mistake was made.
    let header = turbospark_repack::parse_gguf_header(&bytes, bytes.len() as u64)
        .unwrap_or_else(|e| panic!("{}: not a readable GGUF: {e:?}", path.display()));
    assert!(
        header.metadata_u64("turbospark.layer_base").is_none(),
        "{} carries turbospark.layer_base, so it was written by this port rather \
         than by the ecosystem; point this at a published repeng/llama.cpp vector",
        path.display()
    );

    let set = load_control_vector(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    println!("foreign control vector: {}", path.display());
    println!("  architecture : {:?}", set.declared_arch);
    println!("  hidden       : {}", set.hidden);
    println!(
        "  blocks       : {} covered of {} spanned",
        set.covered_layers(),
        set.layers.len()
    );

    assert!(set.covered_layers() > 0, "no block carries a direction");
    assert!(
        set.layer(0).is_none(),
        "block 0 is covered, so this port is reading direction.1 as block 0 again; \
         llama.cpp never applies a direction at block 0"
    );

    // repeng writes a direction for every block it hooked, which is the whole
    // apply range and therefore contiguous. A gap would be a real (and legal)
    // outcome for a hand-built vector, so it is REPORTED rather than asserted;
    // what is asserted is that the covered set starts at 1.
    let first = set
        .layers
        .iter()
        .position(Option::is_some)
        .expect("covered_layers > 0");
    assert_eq!(first, 1, "the lowest direction must be block 1");
    let spanned = set.layers.len();
    if set.covered_layers() != spanned - 1 {
        println!(
            "  NOTE sparse: {} covered across blocks 1..{}",
            set.covered_layers(),
            spanned - 1
        );
    }

    // A non-finite direction multiplies the residual stream and arrives
    // downstream as NaN, which reads as a PERFECT score on any rank
    // instrument (AGENTS.md Gotcha 59).
    for (l, dir) in set.layers.iter().enumerate() {
        let Some(d) = dir else { continue };
        assert!(
            d.values.iter().all(|v| v.is_finite()) && d.inv_norm.is_finite(),
            "block {l} carries a non-finite direction"
        );
        assert_eq!(d.values.len(), set.hidden, "block {l} width");
    }
}
